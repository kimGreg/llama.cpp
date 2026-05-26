#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <unordered_map>
#include <vector>

// Forward declaration of the dp-moe-ext KBar setter so this
// file doesn't take a hard include dependency on the dp-moe-ext tree.
// Resolved at final link via the `llama` library's private dep on
// `dp_moe_ext` when the CUDA extension is enabled. Weak fallbacks keep
// stock and CPU-only tool builds linkable; a missing runtime is a no-op.
extern "C" __attribute__((weak)) bool dp_moe_set_kbar(float, int) { return false; }
extern "C" __attribute__((weak)) bool dp_moe_schedule_active(void) { return false; }
namespace dp_moe_ext {
__attribute__((weak)) bool dp_moe_schedule_get(
    std::vector<int> *                out_thresholds,
    std::vector<float> *              out_kbars,
    int *                             out_allocator_mode) {
    if (out_thresholds) out_thresholds->clear();
    if (out_kbars) out_kbars->clear();
    if (out_allocator_mode) *out_allocator_mode = 0;
    return false;
}
}

static bool parse_dp_moe_float(const char * s, float & out) {
    if (!s || !*s) return false;
    char * end = nullptr;
    out = std::strtof(s, &end);
    return end != s && out >= 0.0f;
}

static bool parse_dp_moe_int(const char * s, int & out) {
    if (!s || !*s) return false;
    char * end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || v < 0 || v > INT_MAX) return false;
    out = (int) v;
    return true;
}

static const char * dp_moe_getenv2(const char * primary, const char * fallback) {
    const char * v = std::getenv(primary);
    if (v && *v) return v;
    v = std::getenv(fallback);
    return (v && *v) ? v : nullptr;
}

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    bool                          phase_aware_enabled = false;
    float                         kbar_reasoning = 0.0f;
    float                         kbar_generation = 0.0f;
    common_reasoning_budget_state prev_rbudget_state  = REASONING_BUDGET_IDLE;

    bool                          schedule_enabled = false;
    std::vector<int>              schedule_thresholds;
    std::vector<float>            schedule_kbars;
    int                           schedule_allocator_mode = 0;
    int                           n_tokens_generated = 0;
    int                           schedule_next_idx  = 0;
    bool                          schedule_generation_started = false;
    bool                          schedule_linear_decay = false;
    float                         schedule_pp_budget = 0.0f;
    float                         schedule_tg_high_budget = 0.0f;
    float                         schedule_tg_low_budget = 0.0f;
    float                         schedule_last_applied_budget = -1.0f;
    int                           schedule_tg_decay_start = 0;
    int                           schedule_tg_decay_end = 0;

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);

        if (schedule_enabled) {
            schedule_generation_started = false;
            n_tokens_generated = 0;
            schedule_next_idx = 0;
            schedule_last_applied_budget = -1.0f;
            const float kbar0 = schedule_linear_decay
                ? schedule_pp_budget
                : (!schedule_kbars.empty() ? schedule_kbars[0] : -1.0f);
            if (kbar0 >= 0.0f) {
                const bool ok = dp_moe_set_kbar(kbar0, schedule_allocator_mode);
                schedule_last_applied_budget = kbar0;
                LOG_INF("dp_moe kbar schedule: reset, primed kbar[0]=%s\n",
                        ok ? "ok" : "no-op (no runtime)");
            }
        }
    }

    void set_logits(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
        } else {
            const auto * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

struct common_sampler * common_sampler_init(const struct llama_model * model, struct common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty() && common_grammar_needs_prefill(params.grammar)) {
        GGML_ASSERT(vocab != nullptr);
        prefill_tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        if (!prefill_tokens.empty()) {
            std::string first_token = common_token_to_piece(vocab, prefill_tokens[0], true);
            if (std::isspace(first_token[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to remove
                prefill_tokens = std::vector<llama_token>(prefill_tokens.begin() + 1, prefill_tokens.end());
            }
        }

        if (grmr && !params.grammar_lazy) {
            try {
                for (const auto & token : prefill_tokens) {
                    llama_sampler_accept(grmr, token);
                    LOG_DBG("%s: accepted prefill token (%d)\n", __func__, token);
                }
            } catch (std::exception &e) {
                LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                    common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
                throw e;
            }
        }
    }

    // DP_MoE phase-aware extension: when the caller (typically a
    // non-chat-template path like llama-completion or llama-perplexity)
    // hasn't populated reasoning_budget_start/end from a chat template,
    // honour direct token-id envs so the reasoning-budget sampler can
    // still attach. Required for the phase-aware dial observer below
    // to fire on <think>/</think> transitions.
    if (params.reasoning_budget_start.empty() || params.reasoning_budget_end.empty()) {
        auto parse_token_id_csv = [](const char * s) -> std::vector<llama_token> {
            std::vector<llama_token> out;
            if (!s || !*s) return out;
            std::stringstream ss(s);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                try { out.push_back((llama_token) std::stoi(tok)); }
                catch (...) { return {}; }
            }
            return out;
        };
        const char * open_csv  = std::getenv("DP_MOE_PHASE_OPEN_TOKEN_IDS");
        const char * close_csv = std::getenv("DP_MOE_PHASE_CLOSE_TOKEN_IDS");
        auto open_ids  = parse_token_id_csv(open_csv);
        auto close_ids = parse_token_id_csv(close_csv);
        if (!open_ids.empty() && !close_ids.empty()) {
            params.reasoning_budget_start = std::move(open_ids);
            params.reasoning_budget_end   = std::move(close_ids);
            if (params.reasoning_budget_tokens < 0) {
                params.reasoning_budget_tokens = INT_MAX;
            }
            LOG_INF("dp_moe phase-aware: reasoning_budget seeded from env "
                    "(start_ids=%zu, end_ids=%zu, tokens=%d)\n",
                    params.reasoning_budget_start.size(),
                    params.reasoning_budget_end.size(),
                    params.reasoning_budget_tokens);
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            params.reasoning_budget_start,
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens,
            prefill_tokens);
    }

    if (params.has_logit_bias()) {
        samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), params.logit_bias.size(), params.logit_bias.data()));
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, llama_model_n_ctx_train(model), params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // default: sample from distribution
            samplers.push_back(llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    if (grmr && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if (rbudget && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

        params.backend_sampling = false;
    }

    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
        /* .phase_aware_enabled = */ false,
        /* .kbar_reasoning = */ 0.0f,
        /* .kbar_generation = */ 0.0f,
        /* .prev_rbudget_state = */ REASONING_BUDGET_IDLE,
        /* .schedule_enabled = */ false,
        /* .schedule_thresholds = */ {},
        /* .schedule_kbars = */ {},
        /* .schedule_allocator_mode = */ 0,
        /* .n_tokens_generated = */ 0,
        /* .schedule_next_idx = */ 0,
        /* .schedule_generation_started = */ false,
        /* .schedule_linear_decay = */ false,
        /* .schedule_pp_budget = */ 0.0f,
        /* .schedule_tg_high_budget = */ 0.0f,
        /* .schedule_tg_low_budget = */ 0.0f,
        /* .schedule_last_applied_budget = */ -1.0f,
        /* .schedule_tg_decay_start = */ 0,
        /* .schedule_tg_decay_end = */ 0,
        /* .t_total_us = */ 0,
    };

    bool sch_from_api = false;
    std::vector<int> sch_api_thr;
    std::vector<float> sch_api_kbars;
    int sch_api_allocator = 0;
    if (dp_moe_schedule_active()) {
        sch_from_api = dp_moe_ext::dp_moe_schedule_get(
            &sch_api_thr, &sch_api_kbars, &sch_api_allocator);
    }
    if (sch_from_api) {
        result->schedule_enabled    = true;
        result->schedule_thresholds = std::move(sch_api_thr);
        result->schedule_kbars      = std::move(sch_api_kbars);
        result->schedule_allocator_mode = sch_api_allocator;
        result->n_tokens_generated  = 0;
        result->schedule_next_idx   = 0;
        const bool ok = dp_moe_set_kbar(
            result->schedule_kbars[0], result->schedule_allocator_mode);
        LOG_INF("dp_moe kbar schedule (HTTP API): %zu transition(s); "
                "primed kbar[0]=%s\n",
                result->schedule_thresholds.size(),
                ok ? "ok" : "no-op (no runtime)");
    }
    const char * sch_pp_env        = !sch_from_api ? dp_moe_getenv2("DP_MOE_KBAR_PP_BUDGET", "DP_MOE_PP_BUDGET") : nullptr;
    const char * sch_tg_high_env   = !sch_from_api ? dp_moe_getenv2("DP_MOE_KBAR_TG_HIGH_BUDGET", "DP_MOE_TG_HIGH_BUDGET") : nullptr;
    const char * sch_tg_low_env    = !sch_from_api ? dp_moe_getenv2("DP_MOE_KBAR_TG_LOW_BUDGET", "DP_MOE_TG_LOW_BUDGET") : nullptr;
    const char * sch_decay_start_env = !sch_from_api ? dp_moe_getenv2("DP_MOE_KBAR_TG_DECAY_START", "DP_MOE_TG_DECAY_START") : nullptr;
    const char * sch_decay_end_env   = !sch_from_api ? dp_moe_getenv2("DP_MOE_KBAR_TG_DECAY_END", "DP_MOE_TG_DECAY_END") : nullptr;
    float sch_pp = 0.0f;
    float sch_tg_high = 0.0f;
    float sch_tg_low = 0.0f;
    int sch_decay_start = 0;
    int sch_decay_end = 0;
    const bool sch_linear_present =
        sch_pp_env || sch_tg_high_env || sch_tg_low_env ||
        sch_decay_start_env || sch_decay_end_env;
    if (!sch_from_api && sch_linear_present) {
        const bool valid =
            parse_dp_moe_float(sch_pp_env,      sch_pp) &&
            parse_dp_moe_float(sch_tg_high_env, sch_tg_high) &&
            parse_dp_moe_float(sch_tg_low_env,  sch_tg_low) &&
            parse_dp_moe_int  (sch_decay_start_env, sch_decay_start) &&
            parse_dp_moe_int  (sch_decay_end_env,   sch_decay_end) &&
            sch_decay_end > sch_decay_start;
        if (valid) {
            result->schedule_enabled          = true;
            result->schedule_linear_decay     = true;
            result->schedule_pp_budget        = sch_pp;
            result->schedule_tg_high_budget   = sch_tg_high;
            result->schedule_tg_low_budget    = sch_tg_low;
            result->schedule_tg_decay_start   = sch_decay_start;
            result->schedule_tg_decay_end     = sch_decay_end;
            result->n_tokens_generated        = 0;
            result->schedule_next_idx         = 0;
            const bool ok = dp_moe_set_kbar(
                result->schedule_pp_budget, result->schedule_allocator_mode);
            result->schedule_last_applied_budget = result->schedule_pp_budget;
            LOG_INF("dp_moe linear kbar schedule: pp=%.3f, tg_high=%.3f, "
                    "tg_low=%.3f, decay=[%d,%d] (primed pp=%s)\n",
                    sch_pp, sch_tg_high, sch_tg_low,
                    sch_decay_start, sch_decay_end,
                    ok ? "ok" : "no-op (no runtime)");
        } else {
            LOG_WRN("dp_moe linear kbar schedule: env vars present but "
                    "malformed — expected PP/TG_HIGH/TG_LOW budgets and "
                    "TG_DECAY_END > TG_DECAY_START — disabled\n");
        }
    }

    const char * sch_prefill_env = (!sch_from_api && !result->schedule_enabled) ? std::getenv("DP_MOE_KBAR_PREFILL") : nullptr;
    const char * sch_front_env   = !sch_from_api ? std::getenv("DP_MOE_KBAR_FRONT") : nullptr;
    const char * sch_rear_env    = !sch_from_api ? std::getenv("DP_MOE_KBAR_REAR") : nullptr;
    const char * sch_front_n_env = !sch_from_api ? std::getenv("DP_MOE_KBAR_FRONT_TOKENS") : nullptr;
    float sch_prefill = 0.0f;
    float sch_front   = 0.0f;
    float sch_rear    = 0.0f;
    int sch_front_tokens = 0;
    const bool sch_three_present =
        sch_prefill_env || sch_front_env || sch_rear_env || sch_front_n_env;
    if (!sch_from_api && !result->schedule_enabled && sch_three_present) {
        const bool valid =
            parse_dp_moe_float(sch_prefill_env, sch_prefill) &&
            parse_dp_moe_float(sch_front_env,   sch_front) &&
            parse_dp_moe_float(sch_rear_env,    sch_rear) &&
            parse_dp_moe_int  (sch_front_n_env, sch_front_tokens) &&
            sch_front_tokens > 0;
        if (valid) {
            result->schedule_enabled    = true;
            result->schedule_thresholds = { 0, sch_front_tokens };
            result->schedule_kbars      = { sch_prefill, sch_front, sch_rear };
            result->n_tokens_generated  = 0;
            result->schedule_next_idx   = 0;
            const bool ok = dp_moe_set_kbar(
                result->schedule_kbars[0], result->schedule_allocator_mode);
            LOG_INF("dp_moe 3-level kbar schedule: prefill=%.3f, "
                    "front=%.3f for %d generated token(s), rear=%.3f "
                    "(primed prefill=%s)\n",
                    sch_prefill, sch_front, sch_front_tokens, sch_rear,
                    ok ? "ok" : "no-op (no runtime)");
        } else {
            LOG_WRN("dp_moe 3-level kbar schedule: env vars present but "
                    "malformed — expected DP_MOE_KBAR_PREFILL, "
                    "DP_MOE_KBAR_FRONT, DP_MOE_KBAR_REAR, and "
                    "DP_MOE_KBAR_FRONT_TOKENS>0 — disabled\n");
        }
    }
    const char * sch_thr_csv  = (!sch_from_api && !result->schedule_enabled) ? std::getenv("DP_MOE_KBAR_SCHEDULE_TOKENS") : nullptr;
    const char * sch_dial_csv = (!sch_from_api && !result->schedule_enabled) ? std::getenv("DP_MOE_KBAR_SCHEDULE_VALUES") : nullptr;
    // Treat empty strings as unset — the orchestrator may clear env
    // explicitly when using the HTTP API path.
    if (sch_thr_csv  && !sch_thr_csv[0])  sch_thr_csv  = nullptr;
    if (sch_dial_csv && !sch_dial_csv[0]) sch_dial_csv = nullptr;
    if (sch_thr_csv && sch_dial_csv) {
        auto parse_int_csv = [](const char * s) {
            std::vector<int> out;
            std::stringstream ss(s ? s : "");
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                try { out.push_back(std::stoi(tok)); }
                catch (...) { return std::vector<int>{}; }
            }
            return out;
        };
        auto parse_kbars = [](const char * s) {
            std::vector<float> out;
            std::stringstream ss(s ? s : "");
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                try {
                    float v = std::stof(tok);
                    if (v < 0.0f) return std::vector<float>{};
                    out.push_back(v);
                } catch (...) {
                    return std::vector<float>{};
                }
            }
            return out;
        };
        auto sch_thr   = parse_int_csv(sch_thr_csv);
        auto sch_kbars = parse_kbars(sch_dial_csv);
        bool valid = (!sch_thr.empty() && sch_kbars.size() == sch_thr.size() + 1);
        for (size_t i = 1; valid && i < sch_thr.size(); ++i) {
            if (sch_thr[i] <= sch_thr[i - 1]) valid = false;
        }
        if (valid) {
            result->schedule_enabled    = true;
            result->schedule_thresholds = std::move(sch_thr);
            result->schedule_kbars      = std::move(sch_kbars);
            result->n_tokens_generated  = 0;
            result->schedule_next_idx   = 0;
            const bool ok = dp_moe_set_kbar(
                result->schedule_kbars[0], result->schedule_allocator_mode);
            LOG_INF("dp_moe kbar schedule: enabled with %zu transition(s); "
                    "primed kbar[0]=%s\n",
                    result->schedule_thresholds.size(),
                    ok ? "ok" : "no-op (no runtime)");
        } else {
            LOG_WRN("dp_moe kbar schedule: env vars present but malformed "
                    "(thresholds=%zu, kbars=%zu — expected kbars=thresholds+1, "
                    "thresholds strictly ascending) — disabled\n",
                    sch_thr.size(), sch_kbars.size());
        }
    }

    if (rbudget && !result->schedule_enabled) {
        const char * thr_r_csv = std::getenv("DP_MOE_PHASE_REASONING_KBAR");
        const char * thr_g_csv = std::getenv("DP_MOE_PHASE_GENERATION_KBAR");
        if (thr_r_csv && thr_g_csv) {
            float kbar_r = 0.0f;
            float kbar_g = 0.0f;
            if (parse_dp_moe_float(thr_r_csv, kbar_r) &&
                parse_dp_moe_float(thr_g_csv, kbar_g)) {
                result->phase_aware_enabled = true;
                result->kbar_reasoning      = kbar_r;
                result->kbar_generation     = kbar_g;
                result->prev_rbudget_state  = REASONING_BUDGET_IDLE;
                const bool ok = dp_moe_set_kbar(result->kbar_reasoning, 0);
                LOG_INF("dp_moe phase-aware dial: enabled, primed reasoning=%s\n",
                        ok ? "ok" : "no-op (no runtime)");
            } else {
                LOG_WRN("dp_moe phase-aware dial: env vars present but "
                        "KBar values malformed — disabled\n");
            }
        }
    }

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

static float dp_moe_linear_budget_for_token(const common_sampler * gsmpl, int token_idx) {
    if (token_idx <= gsmpl->schedule_tg_decay_start) {
        return gsmpl->schedule_tg_high_budget;
    }
    if (token_idx >= gsmpl->schedule_tg_decay_end) {
        return gsmpl->schedule_tg_low_budget;
    }
    const float denom = (float) (gsmpl->schedule_tg_decay_end - gsmpl->schedule_tg_decay_start);
    const float alpha = (float) (token_idx - gsmpl->schedule_tg_decay_start) / denom;
    return gsmpl->schedule_tg_high_budget +
           alpha * (gsmpl->schedule_tg_low_budget - gsmpl->schedule_tg_high_budget);
}

static bool dp_moe_apply_scheduled_kbar(common_sampler * gsmpl, float kbar) {
    if (gsmpl->schedule_last_applied_budget >= 0.0f &&
        std::fabs(gsmpl->schedule_last_applied_budget - kbar) < 1.0e-6f)
    {
        return true;
    }
    const bool ok = dp_moe_set_kbar(kbar, gsmpl->schedule_allocator_mode);
    gsmpl->schedule_last_applied_budget = kbar;
    return ok;
}

void common_sampler_dp_moe_begin_generation(struct common_sampler * gsmpl) {
    if (!gsmpl || !gsmpl->schedule_enabled || gsmpl->schedule_generation_started) {
        return;
    }

    gsmpl->schedule_generation_started = true;
    gsmpl->n_tokens_generated = 0;

    if (gsmpl->schedule_linear_decay) {
        const bool ok = dp_moe_apply_scheduled_kbar(
            gsmpl, gsmpl->schedule_tg_high_budget);
        LOG_INF("dp_moe linear kbar schedule: generation begin, "
                "tg_high=%.3f (set_kbar=%s)\n",
                gsmpl->schedule_tg_high_budget, ok ? "ok" : "no-op");
        return;
    }

    while (gsmpl->schedule_next_idx <
           (int) gsmpl->schedule_thresholds.size() &&
           gsmpl->schedule_thresholds[gsmpl->schedule_next_idx] == 0)
    {
        const int next_dial = gsmpl->schedule_next_idx + 1;
        const float kbar = gsmpl->schedule_kbars[next_dial];
        const bool ok = dp_moe_set_kbar(
            kbar, gsmpl->schedule_allocator_mode);
        LOG_INF("dp_moe kbar schedule: generation begin, "
                "swap to kbar[%d] (set_kbar=%s)\n",
                next_dial, ok ? "ok" : "no-op");
        gsmpl->schedule_next_idx++;
    }
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool accept_grammar) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();
    const bool is_generation_accept = accept_grammar;

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    accept_grammar = accept_grammar && grammar_should_apply(gsmpl);

    llama_sampler_accept(gsmpl->rbudget, token);

    if (gsmpl->schedule_enabled && !gsmpl->schedule_generation_started && is_generation_accept) {
        common_sampler_dp_moe_begin_generation(gsmpl);
    }

    if (gsmpl->schedule_enabled && gsmpl->schedule_generation_started) {
        gsmpl->n_tokens_generated++;
        if (gsmpl->schedule_linear_decay) {
            const int next_token_idx = gsmpl->n_tokens_generated + 1;
            const float next_kbar =
                dp_moe_linear_budget_for_token(gsmpl, next_token_idx);
            const bool ok = dp_moe_apply_scheduled_kbar(gsmpl, next_kbar);
            if (gsmpl->n_tokens_generated == gsmpl->schedule_tg_decay_start ||
                gsmpl->n_tokens_generated == gsmpl->schedule_tg_decay_end - 1 ||
                gsmpl->n_tokens_generated == gsmpl->schedule_tg_decay_end)
            {
                LOG_INF("dp_moe linear kbar schedule: at token %d, "
                        "next_token=%d kbar=%.3f (set_kbar=%s)\n",
                        gsmpl->n_tokens_generated, next_token_idx,
                        next_kbar, ok ? "ok" : "no-op");
            }
        } else {
            while (gsmpl->schedule_next_idx <
               (int) gsmpl->schedule_thresholds.size() &&
               gsmpl->n_tokens_generated >=
               gsmpl->schedule_thresholds[gsmpl->schedule_next_idx])
            {
                const int next_dial = gsmpl->schedule_next_idx + 1;
                const float kbar = gsmpl->schedule_kbars[next_dial];
                const bool ok = dp_moe_set_kbar(
                    kbar, gsmpl->schedule_allocator_mode);
                LOG_INF("dp_moe kbar schedule: at token %d, "
                        "swap to kbar[%d] (set_kbar=%s)\n",
                        gsmpl->n_tokens_generated, next_dial,
                        ok ? "ok" : "no-op");
                gsmpl->schedule_next_idx++;
            }
        }
    }

    // Phase-aware adaptive dial: observe the reasoning-budget state
    // *after* the accept above has updated it. Flip the dp_moe score
    // table on any transition into/out of the reasoning phase.
    if (gsmpl->phase_aware_enabled && gsmpl->rbudget) {
        const auto cur = common_reasoning_budget_get_state(gsmpl->rbudget);
        if (cur != gsmpl->prev_rbudget_state) {
            const bool in_reasoning =
                (cur == REASONING_BUDGET_IDLE         ||
                 cur == REASONING_BUDGET_COUNTING     ||
                 cur == REASONING_BUDGET_WAITING_UTF8 ||
                 cur == REASONING_BUDGET_FORCING);
            const float kbar = in_reasoning ? gsmpl->kbar_reasoning
                                            : gsmpl->kbar_generation;
            const bool ok = dp_moe_set_kbar(kbar, 0);
            LOG_INF("dp_moe phase-aware dial: %s -> %s (set_kbar=%s)\n",
                    in_reasoning ? "non-reasoning" : "reasoning",
                    in_reasoning ? "reasoning"     : "generation",
                    ok ? "ok" : "no-op");
            gsmpl->prev_rbudget_state = cur;
        }
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    return new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
        /* .phase_aware_enabled = */ gsmpl->phase_aware_enabled,
        /* .kbar_reasoning = */ gsmpl->kbar_reasoning,
        /* .kbar_generation = */ gsmpl->kbar_generation,
        /* .prev_rbudget_state = */ gsmpl->prev_rbudget_state,
        /* .schedule_enabled = */ gsmpl->schedule_enabled,
        /* .schedule_thresholds = */ gsmpl->schedule_thresholds,
        /* .schedule_kbars = */ gsmpl->schedule_kbars,
        /* .schedule_allocator_mode = */ gsmpl->schedule_allocator_mode,
        /* .n_tokens_generated = */ gsmpl->n_tokens_generated,
        /* .schedule_next_idx = */ gsmpl->schedule_next_idx,
        /* .schedule_generation_started = */ gsmpl->schedule_generation_started,
        /* .schedule_linear_decay = */ gsmpl->schedule_linear_decay,
        /* .schedule_pp_budget = */ gsmpl->schedule_pp_budget,
        /* .schedule_tg_high_budget = */ gsmpl->schedule_tg_high_budget,
        /* .schedule_tg_low_budget = */ gsmpl->schedule_tg_low_budget,
        /* .schedule_last_applied_budget = */ gsmpl->schedule_last_applied_budget,
        /* .schedule_tg_decay_start = */ gsmpl->schedule_tg_decay_start,
        /* .schedule_tg_decay_end = */ gsmpl->schedule_tg_decay_end,
        /* .t_total_us = */ gsmpl->t_total_us,
    };
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    llama_synchronize(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            GGML_ASSERT(!gsmpl->grmr    && "using grammar in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");

            // TODO: simplify
            gsmpl->cur.resize(1);
            gsmpl->cur[0] = { id, 0.0f, 1.0f };
            cur_p = { gsmpl->cur.data(), gsmpl->cur.size(), 0, true };

            return id;
        }
    }

    gsmpl->set_logits(ctx, idx);

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names, bool allow_alt_names) {
    std::unordered_map<std::string, common_sampler_type> sampler_canonical_name_map {
        { "dry",         COMMON_SAMPLER_TYPE_DRY },
        { "top_k",       COMMON_SAMPLER_TYPE_TOP_K },
        { "top_p",       COMMON_SAMPLER_TYPE_TOP_P },
        { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "min_p",       COMMON_SAMPLER_TYPE_MIN_P },
        { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
        { "xtc",         COMMON_SAMPLER_TYPE_XTC },
        { "infill",      COMMON_SAMPLER_TYPE_INFILL },
        { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES },
        { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    // since samplers names are written multiple ways
    // make it ready for both system names and input names
    std::unordered_map<std::string, common_sampler_type> sampler_alt_name_map {
        { "top-k",       COMMON_SAMPLER_TYPE_TOP_K },
        { "top-p",       COMMON_SAMPLER_TYPE_TOP_P },
        { "top-n-sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { "nucleus",     COMMON_SAMPLER_TYPE_TOP_P },
        { "typical-p",   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "typical",     COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "typ-p",       COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "typ",         COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "min-p",       COMMON_SAMPLER_TYPE_MIN_P },
        { "temp",        COMMON_SAMPLER_TYPE_TEMPERATURE },
        { "adaptive-p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        auto sampler = sampler_canonical_name_map.find(name);
        if (sampler != sampler_canonical_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        if (allow_alt_names) {
            sampler = sampler_alt_name_map.find(name);
            if (sampler != sampler_alt_name_map.end()) {
                samplers.push_back(sampler->second);
                continue;
            }
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}
