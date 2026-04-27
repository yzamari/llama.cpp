// ggml-custom-turboquant.c — bridge from llama.cpp's graph to a TurboQuant
// (PolarQuant + 1-bit QJL) attention provider, registered through the C-API
// hook in llama.h: llama_set_turboquant_attn_fn / llama_get_turboquant_attn_fn.
//
// This file is part of our private llama.cpp fork (yzamari/llama.cpp,
// branch tq-main). It is wired into the graph by llama-graph.cpp via
// ggml_custom_4d when the active KV cache is a llama_kv_cache_turboquant
// instance and flash-attn is off — see docs/path2-algorithm-playbook.md
// §Step 3 in the parent turboQuantPlayground repo.
//
// Layering note: the file lives under ggml/src/ for symmetry with other
// ggml-* sources but is compiled into libllama (see src/CMakeLists.txt),
// not libggml-base — it depends on a llama.h symbol and we don't want
// libggml to depend on libllama.

#include "ggml.h"

#include <stdlib.h>
#include <string.h>

// Forward-declared from llama.h (Step 2 of the playbook). We keep a private
// extern here so this translation unit links on its own once the Step 2
// patch lands. The full include happens in llama-graph.cpp; the symbol
// resolves at link time against libllama.
typedef void (*llama_turboquant_attn_fn)(
        const void * session, int layer_il,
        const float * q, const float * k, const float * v,
        int BH, int n_q, int n_kv, int D,
        float scale, const float * mask, float * out);

extern llama_turboquant_attn_fn llama_get_turboquant_attn_fn(void);

// Userdata layout. Allocated by llama-graph.cpp via ggml_new_buffer so it
// lives as long as the graph context (i.e. until compute completes).
//
// session is the kv_cache_turboquant instance pointer — stable across
// decode calls for the same model, used by the provider to cache
// prepared K-state on GPU (P2.1c Phase A2). layer_il is the layer
// index (0..n_layer-1).
//
// Mask data is NOT captured here — it lives at dst->src[3]->data and is
// only valid at execute time, after the graph allocator has run. Capturing
// it at graph-build time gives a stale / null pointer.
struct ggml_turboquant_userdata {
    float        scale;
    const void * session;     // opaque: kv_cache_turboquant *
    int          layer_il;    // 0..n_layer-1
};

// Forward declaration for -Wmissing-prototypes; the real consumer is the
// extern "C" block in src/llama-graph.cpp.
void ggml_custom_op_turboquant_attn(
        struct ggml_tensor * dst,
        int ith, int nth, void * userdata);

// Read one F32 / F16 element at byte offset `off` from a tensor's data.
// Tensors here may be non-contiguous views of the KV cache, so we always
// index via nb[] strides rather than assuming a packed layout.
static inline float tq_read_elem(const struct ggml_tensor * t, size_t off_bytes) {
    const char * p = (const char *) t->data + off_bytes;
    if (t->type == GGML_TYPE_F32) {
        return *(const float *) p;
    }
    if (t->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
    }
    GGML_ABORT("TurboQuant op: unsupported tensor type %d", (int) t->type);
}

// Custom ggml_custom_4d callback — generic ggml_custom_op_t signature.
// Source tensors come from dst->src[0..3]:
//
//   dst->src[0] = q       : Q activation, [D, n_head, n_tokens, 1] (F16/F32)
//   dst->src[1] = k       : K cache view, [D, n_head_kv, n_kv, 1]  (F16/F32)
//   dst->src[2] = v       : V cache view, layout depends on v_trans:
//                             v_trans=true  : [n_kv, n_head_kv, D, 1]
//                             v_trans=false : [D, n_head_kv, n_kv, 1]
//                           (v_trans is set when cparams.flash_attn=false,
//                            which is the only path that reaches this op.)
//   dst->src[3] = kq_mask : [n_kv, n_q, 1, 1]                       F32
//
// dst is allocated with q's exact shape (see ggml_custom_4d call in
// llama-graph.cpp). It's contiguous F32 — write through default strides.
//
// The provider expects packed F32 buffers with shapes:
//   q   : [BH, n_q,  D] row-major   (BH = batch * n_head)
//   k   : [BH, n_kv, D] row-major
//   v   : [BH, n_kv, D] row-major
//   out : [BH, n_q,  D] row-major
//
// We therefore (a) cast F16 -> F32 inline, (b) permute via nb-aware reads
// to the packed layout, (c) replicate K/V across the GQA group.
//
// **Multi-threaded by BH.** ggml's CPU dispatcher calls this function
// once per worker thread with ith ∈ [0, nth). Each thread handles a
// disjoint slice of BH, packs Q/K/V for its slice, calls the provider
// with the slice's BH count, and writes its slice of dst. dst slices
// are disjoint along the n_head axis, so no synchronization is needed.
void ggml_custom_op_turboquant_attn(
        struct ggml_tensor * dst,
        int ith, int nth, void * userdata) {
    const llama_turboquant_attn_fn fn = llama_get_turboquant_attn_fn();
    if (fn == NULL) {
        // No provider registered. Only ith=0 zeroes the full output —
        // other workers no-op so we don't double-write.
        if (ith == 0) {
            memset(dst->data, 0, ggml_nbytes(dst));
        }
        return;
    }

    const struct ggml_turboquant_userdata * ud =
        (const struct ggml_turboquant_userdata *) userdata;

    const struct ggml_tensor * q       = dst->src[0];
    const struct ggml_tensor * k       = dst->src[1];
    const struct ggml_tensor * v       = dst->src[2];
    const struct ggml_tensor * kq_mask = dst->src[3]; // may be NULL

    GGML_ASSERT(q && k && v);
    GGML_ASSERT((q->type == GGML_TYPE_F32 || q->type == GGML_TYPE_F16) && "TurboQuant op: q must be F32 or F16");
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16) && "TurboQuant op: k must be F32 or F16");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v->type == GGML_TYPE_F16) && "TurboQuant op: v must be F32 or F16");

    // Q layout: [D, n_head, n_q, 1]
    const int D      = (int) q->ne[0];
    const int n_head = (int) q->ne[1];
    const int n_q    = (int) q->ne[2];

    // K cache layout: [D, n_head_kv, n_kv, 1]
    const int n_head_kv = (int) k->ne[1];
    const int n_kv      = (int) k->ne[2];

    GGML_ASSERT(D == (int) k->ne[0] && "TurboQuant op: q and k must share D");

    // V layout depends on v_trans (set by attn_v_trans = !cparams.flash_attn
    // at cache construction; always true on our path since we force flash off).
    const int v_trans = (v->nb[1] > v->nb[2]) ? 1 : 0;
    if (v_trans) {
        GGML_ASSERT((int) v->ne[0] == n_kv);
        GGML_ASSERT((int) v->ne[1] == n_head_kv);
        GGML_ASSERT((int) v->ne[2] == D);
    } else {
        GGML_ASSERT((int) v->ne[0] == D);
        GGML_ASSERT((int) v->ne[1] == n_head_kv);
        GGML_ASSERT((int) v->ne[2] == n_kv);
    }

    GGML_ASSERT(n_head_kv > 0 && n_head % n_head_kv == 0 && "TurboQuant op: GQA ratio must divide");
    const int gqa = n_head / n_head_kv;
    const int BH  = n_head_kv * gqa;  // == n_head

    // ---- BH partition strategy ----
    //
    // Two cases:
    //
    //   Stateless (ud->session == NULL): each worker handles a disjoint
    //   slice of BH, calls the provider with bh_count, writes its slice
    //   of dst. Maximum CPU parallelism for the data shuffle.
    //
    //   Session-cached (ud->session != NULL, Phase A2): the libturboquant
    //   session cache holds one TurboQuantKVCache per (session, layer)
    //   covering the FULL BH. If we partitioned by ith, thread 0 would
    //   prefill the cache with its BH slice (e.g. heads [0..8)) and
    //   thread 1 would then call append() on that cache with a different
    //   slice (heads [8..16)) — the cache class throws on the BH/shape
    //   mismatch (SIGILL via __cxa_throw on Android, since exceptions
    //   escape the noexcept ggml callback). Solution: only ith=0 runs;
    //   it processes the full BH; other workers no-op. We lose the 4×
    //   shuffle parallelism, but the session cache amortizes the
    //   rotate+quantize cost across all decode steps so it's a net win.
    int bh_start, bh_end, bh_count;
    if (ud->session != NULL) {
        if (ith != 0) return;
        bh_start = 0;
        bh_end   = BH;
        bh_count = BH;
    } else {
        bh_start = (BH * ith) / nth;
        bh_end   = (BH * (ith + 1)) / nth;
        bh_count = bh_end - bh_start;
        if (bh_count <= 0) {
            return; // nth > BH: extra threads no-op
        }
    }

    // Mask. F32, contiguous, shared read-only across threads.
    const float * mask_data = NULL;
    if (kq_mask != NULL && kq_mask->data != NULL) {
        GGML_ASSERT(kq_mask->type == GGML_TYPE_F32 && "TurboQuant op: kq_mask must be F32");
        mask_data = (const float *) kq_mask->data;
    }

    // ---- Pack Q slice: q[D, n_head, n_q] -> q_buf[bh_count, n_q, D] ----
    // For GQA, the bh index IS the head index, so this is straightforward.
    float * q_buf = (float *) malloc((size_t) bh_count * n_q * D * sizeof(float));
    GGML_ASSERT(q_buf && "TurboQuant op: q_buf alloc failed");
    for (int t = 0; t < n_q; ++t) {
        for (int bh = bh_start; bh < bh_end; ++bh) {
            const int h = bh; // 1:1 for GQA folded layout
            float * row = q_buf + ((size_t) (bh - bh_start) * n_q + t) * D;
            for (int d = 0; d < D; ++d) {
                const size_t off = (size_t) d * q->nb[0]
                                 + (size_t) h * q->nb[1]
                                 + (size_t) t * q->nb[2];
                row[d] = tq_read_elem(q, off);
            }
        }
    }

    // ---- Pack K slice: k[D, n_head_kv, n_kv] -> k_buf[bh_count, n_kv, D] ----
    // bh -> hkv = bh / gqa. Multiple bh values within the same hkv group
    // share the same K vector (GQA replication).
    float * k_buf = (float *) malloc((size_t) bh_count * n_kv * D * sizeof(float));
    GGML_ASSERT(k_buf && "TurboQuant op: k_buf alloc failed");
    {
        int last_hkv = -1;
        float src[1024];
        GGML_ASSERT(D <= (int) (sizeof(src) / sizeof(src[0])) && "TurboQuant op: D exceeds local buf");
        for (int kt = 0; kt < n_kv; ++kt) {
            // Read K vector once per hkv-kt; replicate across the bh values
            // in our slice that map to that hkv. We walk bh in our slice in
            // order; an hkv can repeat for `gqa` consecutive bh values.
            last_hkv = -1;
            for (int bh = bh_start; bh < bh_end; ++bh) {
                const int hkv = bh / gqa;
                if (hkv != last_hkv) {
                    for (int d = 0; d < D; ++d) {
                        const size_t off = (size_t) d   * k->nb[0]
                                         + (size_t) hkv * k->nb[1]
                                         + (size_t) kt  * k->nb[2];
                        src[d] = tq_read_elem(k, off);
                    }
                    last_hkv = hkv;
                }
                memcpy(k_buf + ((size_t) (bh - bh_start) * n_kv + kt) * D,
                       src,
                       (size_t) D * sizeof(float));
            }
        }
    }

    // ---- Pack V slice: layout depends on v_trans ----
    float * v_buf = (float *) malloc((size_t) bh_count * n_kv * D * sizeof(float));
    GGML_ASSERT(v_buf && "TurboQuant op: v_buf alloc failed");
    {
        float src[1024];
        for (int kt = 0; kt < n_kv; ++kt) {
            int last_hkv = -1;
            for (int bh = bh_start; bh < bh_end; ++bh) {
                const int hkv = bh / gqa;
                if (hkv != last_hkv) {
                    for (int d = 0; d < D; ++d) {
                        size_t off;
                        if (v_trans) {
                            off = (size_t) kt  * v->nb[0]
                                + (size_t) hkv * v->nb[1]
                                + (size_t) d   * v->nb[2];
                        } else {
                            off = (size_t) d   * v->nb[0]
                                + (size_t) hkv * v->nb[1]
                                + (size_t) kt  * v->nb[2];
                        }
                        src[d] = tq_read_elem(v, off);
                    }
                    last_hkv = hkv;
                }
                memcpy(v_buf + ((size_t) (bh - bh_start) * n_kv + kt) * D,
                       src,
                       (size_t) D * sizeof(float));
            }
        }
    }

    // ---- Provider call for this thread's BH slice ----
    float * out_buf = (float *) malloc((size_t) bh_count * n_q * D * sizeof(float));
    GGML_ASSERT(out_buf && "TurboQuant op: out_buf alloc failed");

    fn(ud->session, ud->layer_il,
       q_buf, k_buf, v_buf,
       bh_count, n_q, n_kv, D,
       ud->scale, mask_data,
       out_buf);

    // ---- Write back to dst slice [D, n_head, n_q] ----
    // dst is contiguous F32 with q's shape, so dst[d, h, t] is at
    // ((t * n_head + h) * D + d). Each thread writes only h ∈ [bh_start, bh_end),
    // which is disjoint from other threads' slices.
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && "TurboQuant op: dst must be F32");
    GGML_ASSERT(ggml_is_contiguous(dst) && "TurboQuant op: dst must be contiguous");
    float * dst_data = (float *) dst->data;
    for (int t = 0; t < n_q; ++t) {
        for (int bh = bh_start; bh < bh_end; ++bh) {
            const int h = bh;
            const float * src = out_buf + ((size_t) (bh - bh_start) * n_q + t) * D;
            float       * row = dst_data + ((size_t) t * n_head + h) * D;
            memcpy(row, src, (size_t) D * sizeof(float));
        }
    }

    free(q_buf);
    free(k_buf);
    free(v_buf);
    free(out_buf);
}
