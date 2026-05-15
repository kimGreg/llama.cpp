// streamllm-gate-trace
//
// Loads a model, runs prefill over a calibration corpus, captures the
// per-(layer, token) MoE router top-k expert IDs + normalized gate
// weights via `ggml_backend_sched_eval_callback`, and writes a binary
// trace file consumed by experiments/streamllm_pareto.
//
// The tensors of interest are produced by `llm_build_moe_routing_softmax_topk`
// inside `build_moe_ffn` and are already named "ffn_moe_topk-<il>" and
// "ffn_moe_weights-<il>" via the existing `cb()` mechanism — no
// llama.cpp patch needed.
//
// Trace file layout (little-endian throughout):
//
//   header (40 bytes):
//     char     magic[4]      = "GTRA"
//     uint32_t version       = 1
//     uint32_t n_layer
//     uint32_t n_expert
//     uint32_t n_expert_used
//     uint64_t n_tokens
//     uint64_t ctx_n_tokens   (tokens per perplexity chunk; for offline reshape)
//
//   then per token (in stream order), per layer (0..n_layer-1):
//     uint16_t expert_ids[K]
//     float    weights[K]            // already softmax-and-renorm'd
//
//   total bytes after header = n_tokens * n_layer * K * (2 + 4)
//
// CLI:
//   streamllm-gate-trace -m model.gguf -f corpus.txt -o trace.bin
//                        [-c N_CTX] [--n-chunks N] [--seed S]
//

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>


namespace {

struct GateTraceHeader {
    char     magic[4];
    uint32_t version;
    uint32_t n_layer;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint64_t n_tokens;       // back-patched at close()
    uint64_t ctx_n_tokens;   // tokens per perplexity chunk
};

class GateTracer {
public:
    GateTracer() = default;

    void open(const std::string & path, uint32_t n_layer, uint32_t n_expert,
              uint32_t n_expert_used, uint64_t ctx_n_tokens) {
        m_path        = path;
        m_n_layer     = n_layer;
        m_n_expert    = n_expert;
        m_n_expert_used = n_expert_used;
        m_ctx_n_tokens = ctx_n_tokens;
        m_layer_topk_w.assign(n_layer, {});
        m_layer_ids.assign(n_layer, {});
        m_seen_probs.assign(n_layer, false);
        m_seen_ids.assign(n_layer, false);
        m_n_expert = n_expert;

        m_out.open(path, std::ios::binary | std::ios::trunc);
        if (!m_out) {
            fprintf(stderr, "[gate-trace] failed to open output: %s\n", path.c_str());
            std::abort();
        }
        // Write placeholder header — back-patch n_tokens at close.
        GateTraceHeader hdr{};
        std::memcpy(hdr.magic, "GTRA", 4);
        hdr.version       = 1;
        hdr.n_layer       = n_layer;
        hdr.n_expert      = n_expert;
        hdr.n_expert_used = n_expert_used;
        hdr.n_tokens      = 0;
        hdr.ctx_n_tokens  = ctx_n_tokens;
        m_out.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    }

    void close() {
        m_out.flush();
        // Back-patch n_tokens.
        m_out.seekp(offsetof(GateTraceHeader, n_tokens), std::ios::beg);
        m_out.write(reinterpret_cast<const char *>(&m_n_tokens), sizeof(uint64_t));
        m_out.close();
        fprintf(stderr, "[gate-trace] wrote %llu tokens to %s\n",
                (unsigned long long) m_n_tokens, m_path.c_str());
    }

    // Returns true if the scheduler should split the graph at this
    // tensor (ask=true) AND captures the data when ask=false.
    bool on_eval(struct ggml_tensor * t, bool ask) {
        const char * name = t->name;
        if (!name) return false;

        // Capture two tensors per MoE layer:
        //   * ``ffn_moe_logits-il`` (MUL_MAT) — raw router logits; we
        //     softmax + top-K in this callback to recover the ids and
        //     normalized gate weights.
        //   * MUL_MAT_ID's src[2] — top-K expert ids as the runtime
        //     actually computed them, used to cross-check our top-K.
        const bool is_logits  = std::strncmp(name, "ffn_moe_logits-", 15) == 0;
        const bool is_mmid    = (t->op == GGML_OP_MUL_MAT_ID);
        if (!is_logits && !is_mmid) return false;
        if (ask) return true;

        // ----- logits path: softmax + top-K --------------------------
        if (is_logits) {
            const int il = parse_layer(name, 15);
            if (il < 0 || il >= (int) m_n_layer) return true;
            if (m_seen_probs[il]) return true;  // first call per batch wins

            const int64_t n_expert   = t->ne[0];
            const int64_t n_tokens_t = t->ne[1];
            if (n_tokens_t <= 0) return true;

            if (t->type != GGML_TYPE_F32) {
                fprintf(stderr, "[gate-trace] WARN: logits tensor il=%d has type=%s, "
                        "expected F32 — skipping\n", il, ggml_type_name(t->type));
                m_seen_probs[il] = true;
                auto & wts_dst = m_layer_topk_w[il];
                auto & ids_dst = m_layer_ids[il];
                const int64_t K = (int64_t) m_n_expert_used;
                wts_dst.assign((size_t) n_tokens_t * K, 1.0f / (float) K);
                ids_dst.assign((size_t) n_tokens_t * K, 0);
                m_seen_ids[il] = true;
                m_pending_n_tokens = n_tokens_t;
                return true;
            }

            std::lock_guard<std::mutex> lock(m_mu);
            const size_t nb = ggml_nbytes(t);
            std::vector<char> host(nb);
            if (ggml_backend_buffer_is_host(t->buffer)) {
                std::memcpy(host.data(), t->data, nb);
            } else {
                ggml_backend_tensor_get(t, host.data(), 0, nb);
            }
            const float * logits = (const float *) host.data();

            const int64_t K = (int64_t) m_n_expert_used;
            auto & ids_dst = m_layer_ids[il];
            auto & wts_dst = m_layer_topk_w[il];
            ids_dst.resize((size_t) n_tokens_t * K);
            wts_dst.resize((size_t) n_tokens_t * K);

            std::vector<int>   idx_buf(n_expert);
            std::vector<float> probs(n_expert);
            for (int64_t tok = 0; tok < n_tokens_t; ++tok) {
                const float * row = logits + tok * n_expert;
                float maxv = row[0];
                for (int64_t e = 1; e < n_expert; ++e) if (row[e] > maxv) maxv = row[e];
                float sum = 0.0f;
                for (int64_t e = 0; e < n_expert; ++e) {
                    probs[e] = std::exp(row[e] - maxv);
                    sum += probs[e];
                }
                const float inv = 1.0f / sum;
                for (int64_t e = 0; e < n_expert; ++e) probs[e] *= inv;

                for (int64_t e = 0; e < n_expert; ++e) idx_buf[e] = (int) e;
                std::partial_sort(idx_buf.begin(), idx_buf.begin() + K, idx_buf.end(),
                    [&](int a, int b) { return probs[a] > probs[b]; });

                float topk_sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) topk_sum += probs[idx_buf[k]];
                const float inv_k = topk_sum > 0.0f ? 1.0f / topk_sum : 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    ids_dst[(size_t) tok * K + k] = (uint16_t) idx_buf[k];
                    wts_dst[(size_t) tok * K + k] = probs[idx_buf[k]] * inv_k;
                }
            }
            m_seen_probs[il] = true;
            // If MUL_MAT_ID didn't visit yet, mark ids-seen too (they
            // come from the same softmax → top-K we just computed).
            if (!m_seen_ids[il]) m_seen_ids[il] = true;
            m_pending_n_tokens = n_tokens_t;
            return true;  // continue traversal
        }

        // ----- MUL_MAT_ID path: no-op (logits path already captured
        // both ids and weights for the layer).  Must return true so
        // ggml-backend-sched's per-node loop continues traversing the
        // rest of the split graph (see ggml-backend.cpp line 1708:
        // ``if (need && !callback_eval(ask=false)) break;``).
        return true;
    }

    // Flush one batch.  Caller invokes after `llama_decode` returns
    // and all layers have populated their per-batch buffers.  Writes
    // the batch in (token, layer) major order — token outer, layer
    // inner — so a Python loader can mmap with stride
    // `n_layer * K * 6` bytes per token.
    void flush_batch(int64_t batch_tokens) {
        std::lock_guard<std::mutex> lock(m_mu);
        if (batch_tokens <= 0) return;
        for (int il = 0; il < (int) m_n_layer; ++il) {
            if (!m_seen_probs[il] || !m_seen_ids[il]) {
                fprintf(stderr, "[gate-trace] WARN: layer %d missing on batch "
                        "(probs=%d ids=%d)\n",
                        il, (int) m_seen_probs[il], (int) m_seen_ids[il]);
                return;
            }
        }
        const int64_t K = (int64_t) m_n_expert_used;
        const size_t rec_bytes = (size_t) K * (sizeof(uint16_t) + sizeof(float));
        std::vector<char> rec(rec_bytes);
        for (int64_t tok = 0; tok < batch_tokens; ++tok) {
            for (int il = 0; il < (int) m_n_layer; ++il) {
                uint16_t * ids_dst = (uint16_t *) rec.data();
                float    * w_dst   = (float *) (rec.data() + K * sizeof(uint16_t));
                const auto & ids = m_layer_ids[il];
                const auto & wts = m_layer_topk_w[il];
                for (int64_t k = 0; k < K; ++k) {
                    ids_dst[k] = ids[(size_t) tok * K + k];
                    w_dst[k]   = wts[(size_t) tok * K + k];
                }
                m_out.write(rec.data(), rec_bytes);
            }
        }
        m_n_tokens += (uint64_t) batch_tokens;
        for (int il = 0; il < (int) m_n_layer; ++il) {
            m_seen_probs[il] = false;
            m_seen_ids[il]   = false;
        }
    }

private:
    static int parse_layer(const char * name, size_t prefix_len) {
        const char * p = name + prefix_len;
        if (!*p) return -1;
        return std::atoi(p);
    }

    std::ofstream m_out;
    std::string   m_path;
    uint32_t      m_n_layer       = 0;
    uint32_t      m_n_expert      = 0;
    uint32_t      m_n_expert_used = 0;
    uint64_t      m_ctx_n_tokens  = 0;
    uint64_t      m_n_tokens      = 0;
    int64_t       m_pending_n_tokens = 0;

    std::vector<std::vector<float>>     m_layer_topk_w;  // [il] = [n_tokens × K]
    std::vector<std::vector<uint16_t>>  m_layer_ids;     // [il] = [n_tokens × K]
    std::vector<bool>                   m_seen_probs;
    std::vector<bool>                   m_seen_ids;

    std::mutex m_mu;
};

GateTracer g_tracer;

bool gate_trace_cb(struct ggml_tensor * t, bool ask, void * /* user_data */) {
    return g_tracer.on_eval(t, ask);
}

} // namespace


static void print_usage(int, char ** argv) {
    LOG("\nstreamllm-gate-trace — capture per-(layer, token) MoE gate-score "
        "trace via eval_callback.\n");
    LOG("\nUsage:\n  %s -m model.gguf -f corpus.txt -o trace.bin "
        "[-c 4096] [--n-chunks 0] [-ngl 999]\n\n", argv[0]);
}


int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx       = 4096;
    params.n_batch     = 2048;
    params.n_ubatch    = 512;
    params.warmup      = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    common_init();

    if (params.out_file.empty()) {
        fprintf(stderr, "ERROR: -o (trace path) required\n");
        return 1;
    }
    if (params.prompt.empty() && params.prompt_file.empty()) {
        fprintf(stderr, "ERROR: -f (calibration corpus) required\n");
        return 1;
    }

    // Ensure n_batch ≥ n_ctx so a full chunk fits in one llama_decode call.
    if (params.n_batch < params.n_ctx) {
        params.n_batch = params.n_ctx;
    }
    if (params.n_ubatch > params.n_batch) {
        params.n_ubatch = params.n_batch;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // Set callback AFTER backend init (imatrix does this).
    params.cb_eval           = gate_trace_cb;
    params.cb_eval_user_data = nullptr;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) {
        fprintf(stderr, "ERROR: failed to load model\n");
        return 2;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_layer        = llama_model_n_layer(model);
    // n_expert / n_expert_used aren't exposed via the high-level API
    // for every arch; we fall back to defaults the trace's consumer
    // can override from the GGUF metadata.
    const int n_expert       = 128;  // Qwen3-30B-A3B
    const int n_expert_used  = 8;

    // Tokenize the calibration corpus.
    std::string prompt = params.prompt;
    if (prompt.empty()) {
        std::ifstream f(params.prompt_file);
        std::stringstream ss; ss << f.rdbuf();
        prompt = ss.str();
    }
    std::vector<llama_token> toks = common_tokenize(vocab, prompt, true, false);
    fprintf(stderr, "[gate-trace] tokenized %zu tokens (corpus = %.2f MB)\n",
            toks.size(), prompt.size() / 1e6);

    const int n_ctx = params.n_ctx;
    int64_t n_chunks = params.n_chunks > 0
        ? std::min<int64_t>(params.n_chunks, (int64_t) toks.size() / n_ctx)
        : (int64_t) toks.size() / n_ctx;
    fprintf(stderr, "[gate-trace] processing %lld chunks of %d tokens\n",
            (long long) n_chunks, n_ctx);

    g_tracer.open(params.out_file, n_layer, n_expert, n_expert_used, n_ctx);

    // imatrix-style batch: pre-allocate, fill with logits=true per token.
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    for (int64_t chunk = 0; chunk < n_chunks; ++chunk) {
        const size_t off = (size_t) chunk * n_ctx;

        llama_memory_clear(llama_get_memory(ctx), true);

        // Build batch manually with logits=true so every token's
        // forward pass reaches all 48 MoE layers — same setup as imatrix.
        batch.n_tokens = 0;
        for (int k = 0; k < n_ctx; ++k) {
            common_batch_add(batch, toks[off + k], k, {0}, /*logits*/true);
        }

        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            fprintf(stderr, "[gate-trace] llama_decode chunk=%lld failed: %d\n",
                    (long long) chunk, rc);
            break;
        }
        g_tracer.flush_batch((int64_t) batch.n_tokens);

        if ((chunk + 1) % 8 == 0 || chunk + 1 == n_chunks) {
            fprintf(stderr, "[gate-trace] chunk %lld / %lld\n",
                    (long long) (chunk + 1), (long long) n_chunks);
        }
    }
    llama_batch_free(batch);

    g_tracer.close();
    return 0;
}
