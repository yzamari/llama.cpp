#pragma once

// llama-kv-cache-turboquant — fork-only KV cache implementation that routes
// K/V through TurboQuant (PolarQuant + 1-bit QJL, ICLR 2026).
//
// This is the scaffold from Path 2.1 of the TurboQuant fork. The class is
// instantiated when the caller sets llama_context_params::kv_turboquant=true
// (the JNI shim wires kvType=3 to that). Today the class is a thin tag over
// llama_kv_cache so the integration shape — cparams flag, memory_params field,
// model factory branch, derived class — is in place end-to-end, ready for the
// algorithmic substitution to land in a follow-up commit without further
// plumbing churn.
//
// Roadmap inside this class:
//   - Override the K/V write methods (cpy_k / cpy_v in build_attn) to feed
//     turboquant::TurboQuantKVCache from cpp/include/turboquant/api.hpp
//     instead of allocating ggml tensors, and
//   - Substitute the standard Q@K.T → softmax → attn@V triple in
//     llm_graph_context::build_attn_mha with a custom ggml op that calls
//     attention_scores() + attend() on the per-layer TurboQuantKVCache.
// Both will live in this fork's branch (yzamari/llama.cpp, tq-main); see
// turboQuantPlayground/.claude/plans/hashed-squishing-hoare.md for the plan.

#include "llama-kv-cache.h"

class llama_kv_cache_turboquant : public llama_kv_cache {
public:
    using llama_kv_cache::llama_kv_cache;  // inherit constructors

    // Tag for callers that need to detect TurboQuant mode without RTTI in
    // hot paths (e.g. graph builder when the algorithmic substitution lands).
    bool is_turboquant() const { return true; }
};
