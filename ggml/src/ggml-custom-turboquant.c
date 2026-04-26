// ggml-custom-turboquant.c — bridge from llama.cpp's graph to a TurboQuant
// (PolarQuant + 1-bit QJL) attention provider, registered through the C-API
// hook in llama.h: llama_set_turboquant_attn_fn / llama_get_turboquant_attn_fn.
//
// This file is part of our private llama.cpp fork (yzamari/llama.cpp,
// branch tq-main). It is wired into the graph by llama-graph.cpp via
// ggml_map_custom3 when the active KV cache is a llama_kv_cache_turboquant
// instance and flash-attn is off — see docs/path2-algorithm-playbook.md
// §Step 3 in the parent turboQuantPlayground repo.
//
// Layering note: the file lives under ggml/src/ for symmetry with other
// ggml-* sources but is compiled into libllama (see src/CMakeLists.txt),
// not libggml-base — it depends on a llama.h symbol and we don't want
// libggml to depend on libllama.

#include "ggml.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Forward-declared from llama.h (Step 2 of the playbook). We keep a private
// extern here so this translation unit links on its own once the Step 2
// patch lands. The full include happens in llama-graph.cpp; the symbol
// resolves at link time against libllama.
typedef void (*llama_turboquant_attn_fn)(
        const float * q, const float * k, const float * v,
        int BH, int n_q, int n_kv, int D,
        float scale, const float * mask, float * out);

extern llama_turboquant_attn_fn llama_get_turboquant_attn_fn(void);

// Userdata layout. Allocated by llama-graph.cpp via ggml_new_buffer so it
// lives as long as the graph context (i.e. until compute completes).
struct ggml_turboquant_userdata {
    float scale;
    const float * mask;   // [n_q * n_kv] additive, may be NULL
    int n_q_mask;
    int n_kv_mask;
};

// Forward declaration for -Wmissing-prototypes; the real consumer is the
// extern "C" block in src/llama-graph.cpp.
void ggml_custom_op_turboquant_attn(
        struct ggml_tensor * dst,
        const struct ggml_tensor * q,
        const struct ggml_tensor * k,
        const struct ggml_tensor * v,
        int ith, int nth, void * userdata);

// Custom ggml_map_custom3 callback. Invoked once per graph compute step
// per node; the dispatcher calls it nth times (once per worker), so we
// guard ith != 0 and run the whole op single-threaded for the v1 cut.
//
// Tensor layout contract (un-permuted, as built by build_attn before the
// (0,2,1,3) permute in build_attn_mha):
//
//   q : [D, n_head,    n_tokens]   -- contiguous, F32
//   k : [D, n_kv,      n_head_kv]  -- contiguous, F32
//   v : [D, n_kv,      n_head_kv]  -- contiguous, F32
//
// dst is a dup of q, so its shape is [D, n_head, n_tokens] as well, which
// matches what the rest of build_attn_mha expects after the permute is
// undone — we write back in q's layout so the downstream `cur = ggml_permute
// (kqv, 0, 2, 1, 3)` keeps working.
void ggml_custom_op_turboquant_attn(
        struct ggml_tensor * dst,
        const struct ggml_tensor * q,
        const struct ggml_tensor * k,
        const struct ggml_tensor * v,
        int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) {
        return;
    }

    const llama_turboquant_attn_fn fn = llama_get_turboquant_attn_fn();
    if (fn == NULL) {
        // No provider registered: zero output and return. The graph builder
        // is supposed to guarantee one is set before this op is emitted, but
        // we fail safe rather than crash.
        memset(dst->data, 0, ggml_nbytes(dst));
        return;
    }

    const struct ggml_turboquant_userdata * ud =
        (const struct ggml_turboquant_userdata *) userdata;

    GGML_ASSERT(q->type == GGML_TYPE_F32 && "TurboQuant op expects F32 q");
    GGML_ASSERT(k->type == GGML_TYPE_F32 && "TurboQuant op expects F32 k");
    GGML_ASSERT(v->type == GGML_TYPE_F32 && "TurboQuant op expects F32 v");

    const int D         = (int) q->ne[0];
    const int n_head    = (int) q->ne[1];
    const int n_q       = (int) q->ne[2];
    const int n_kv      = (int) k->ne[1];
    const int n_head_kv = (int) k->ne[2];

    GGML_ASSERT(D == (int) k->ne[0]);
    GGML_ASSERT(D == (int) v->ne[0]);
    GGML_ASSERT(n_kv == (int) v->ne[1]);
    GGML_ASSERT(n_head_kv == (int) v->ne[2]);
    GGML_ASSERT(n_head_kv > 0 && n_head % n_head_kv == 0 && "GQA ratio must divide");

    const int gqa = n_head / n_head_kv;
    const int BH  = n_head_kv * gqa;  // == n_head

    // Q is laid out as q[t][h][d] (D fastest-changing). The provider expects
    // [BH, n_q, D] row-major, i.e. q'[bh][q_idx][d] = q[t=q_idx][h=bh][d].
    // For GQA, BH == n_head so this is just a (h, t) -> (t, h) swap.
    float * q_buf = (float *) malloc((size_t) BH * n_q * D * sizeof(float));
    GGML_ASSERT(q_buf && "TurboQuant op: q_buf alloc failed");
    for (int t = 0; t < n_q; ++t) {
        for (int h = 0; h < n_head; ++h) {
            const float * src     = (const float *) q->data + ((size_t) t * n_head + h) * D;
            float       * dst_row = q_buf + ((size_t) h * n_q + t) * D;
            memcpy(dst_row, src, (size_t) D * sizeof(float));
        }
    }

    // K/V are stored once per KV head; the provider takes BH-aligned
    // tensors so we replicate each KV head `gqa` times along the BH axis.
    // K layout is k[hkv][kt][d] -> k_buf[bh = hkv*gqa + g][kt][d].
    float * k_buf = (float *) malloc((size_t) BH * n_kv * D * sizeof(float));
    float * v_buf = (float *) malloc((size_t) BH * n_kv * D * sizeof(float));
    GGML_ASSERT(k_buf && v_buf && "TurboQuant op: k/v_buf alloc failed");
    for (int hkv = 0; hkv < n_head_kv; ++hkv) {
        for (int kt = 0; kt < n_kv; ++kt) {
            const float * ksrc = (const float *) k->data + ((size_t) hkv * n_kv + kt) * D;
            const float * vsrc = (const float *) v->data + ((size_t) hkv * n_kv + kt) * D;
            for (int g = 0; g < gqa; ++g) {
                const int bh = hkv * gqa + g;
                memcpy(k_buf + ((size_t) bh * n_kv + kt) * D, ksrc, (size_t) D * sizeof(float));
                memcpy(v_buf + ((size_t) bh * n_kv + kt) * D, vsrc, (size_t) D * sizeof(float));
            }
        }
    }

    // Output buffer in provider layout [BH, n_q, D]. We write into a temp,
    // then permute back to dst's [D, n_head, n_tokens] layout.
    float * out_buf = (float *) malloc((size_t) BH * n_q * D * sizeof(float));
    GGML_ASSERT(out_buf && "TurboQuant op: out_buf alloc failed");

    fn(q_buf, k_buf, v_buf,
       BH, n_q, n_kv, D,
       ud->scale, ud->mask,
       out_buf);

    // Permute provider output [BH, n_q, D] back to dst layout [D, n_head, n_tokens]
    // i.e. dst[t][h][d] = out_buf[h][t][d].
    for (int t = 0; t < n_q; ++t) {
        for (int h = 0; h < n_head; ++h) {
            const float * src     = out_buf + ((size_t) h * n_q + t) * D;
            float       * dst_row = (float *) dst->data + ((size_t) t * n_head + h) * D;
            memcpy(dst_row, src, (size_t) D * sizeof(float));
        }
    }

    free(q_buf);
    free(k_buf);
    free(v_buf);
    free(out_buf);
}
