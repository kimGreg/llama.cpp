#include "models.h"

llm_build_qwen3moe::llm_build_qwen3moe(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            if (model.layers[il].wo_s) {
                cur = ggml_mul(ctx0, cur, model.layers[il].wo_s);
            }
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // MoE branch
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out = nullptr;
        if (model.streamllm_executor != nullptr) {
            // ─── Mode A (StreamLLM milestone 1, S5).
            //
            // Bypass build_moe_ffn entirely: emit the router/topk via
            // the shared helper, then a single per-layer sentinel
            // node that the runtime's pre_op_hook routes to
            // forward_moe_layer. Managed Qwen3-MoE no longer surfaces
            // as `MUL_MAT_ID` in the cgraph for these layers.
            //
            // Per-expert scale tensors (the `_s` siblings) aren't
            // handled by forward_moe_layer in M1 — Qwen3-30B-A3B
            // AnyBCQ doesn't carry them, but a future variant might.
            // Fail loudly rather than silently dropping the scale.
            if (model.layers[il].ffn_up_exps_s   != nullptr ||
                model.layers[il].ffn_gate_exps_s != nullptr ||
                model.layers[il].ffn_down_exps_s != nullptr) {
                GGML_ABORT(
                    "streamllm-ext / Qwen3-MoE Mode A (L=%d): per-expert "
                    "scale tensors (`_s`) are not supported by "
                    "forward_moe_layer in M1. Either remove the scales "
                    "from the artifact or extend the executor to honour "
                    "them.", il);
            }

            // logits = build_lora_mm(gate_inp, cur). Matches what
            // build_moe_ffn does internally for the simple-routing
            // path (Qwen3-MoE: SOFTMAX, no exp_probs_b, no LLAMA4
            // quirks, no expert groups).
            ggml_tensor * logits =
                build_lora_mm(model.layers[il].ffn_gate_inp, cur);
            cb(logits, "ffn_moe_logits", il);

            auto router = llm_build_moe_routing_softmax_topk(
                ctx0, logits,
                /*n_expert=*/      (int64_t) n_expert,
                /*n_expert_used=*/ (int64_t) n_expert_used,
                /*norm_w=*/        true);

            // Sentinel: ggml_dup(cur) gives an F32 [n_embd, n_tokens]
            // op node whose default backend executor would memcpy
            // src[0] → dst. The runtime's pre_op_hook claims this
            // node by name and writes layer_out into dst from
            // forward_moe_layer instead. src[1..3] are manually
            // wired so the graph allocator schedules ids/probs/
            // weights ahead of the sentinel — by the time the hook
            // fires, all three are resident on the compute stream.
            ggml_tensor * sentinel = ggml_dup(ctx0, cur);
            sentinel->src[1] = router.ids;
            sentinel->src[2] = router.probs;
            sentinel->src[3] = router.weights;
            {
                char name[64];
                std::snprintf(name, sizeof(name),
                              "streamllm.moe_layer_%d", il);
                ggml_set_name(sentinel, name);
            }
            moe_out = sentinel;
            // Deliberately do NOT call `cb(moe_out, "ffn_moe_out", il)`
            // here. cb() renames the tensor via ggml_format_name and
            // would overwrite the "streamllm.moe_layer_<il>" sentinel
            // name that the pre_op_hook matches on. The sentinel's
            // semantic role IS the moe_out — keep its identifying name.
        } else {
            moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr, nullptr,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
            cb(moe_out, "ffn_moe_out", il);
        }
        cur = moe_out;

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
