// DPMoE schedule fetch unit test.
//
// Covers dp_moe_schedule_get(), the bulk fetch path used by request-time
// sampler setup to read the active runtime KBar schedule.

#include "moe/common/runtime_glue.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const char * msg)
{
    if (cond) return;
    std::fprintf(stderr, "FAIL: %s\n", msg);
    ++g_failures;
}

bool same_ints(const std::vector<int> & a, const std::vector<int> & b)
{
    return a == b;
}

bool same_floats(const std::vector<float> & a, const std::vector<float> & b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > 1.0e-6f) return false;
    }
    return true;
}

bool snapshot_is_a_or_b(const std::vector<int> & thr,
                        const std::vector<float> & kbars,
                        int mode)
{
    static const std::vector<int> a_thr{16, 32};
    static const std::vector<float> a_kbars{7.0f, 4.0f, 2.0f};
    static const std::vector<int> b_thr{8};
    static const std::vector<float> b_kbars{3.0f, 2.5f};

    const bool a =
        mode == 11 && same_ints(thr, a_thr) && same_floats(kbars, a_kbars);
    const bool b =
        mode == 22 && same_ints(thr, b_thr) && same_floats(kbars, b_kbars);
    return a || b;
}

} // namespace

int main()
{
    using dp_moe_ext::dp_moe_kbar_schedule_set;
    using dp_moe_ext::dp_moe_schedule_active;
    using dp_moe_ext::dp_moe_schedule_clear;
    using dp_moe_ext::dp_moe_schedule_get;

    dp_moe_schedule_clear();
    check(!dp_moe_schedule_active(), "clear should make schedule inactive");

    std::vector<int> out_thr{123};
    std::vector<float> out_kbars{9.0f};
    int out_mode = 77;
    check(!dp_moe_schedule_get(&out_thr, &out_kbars, &out_mode),
          "fetch should fail when no schedule is active");
    check(same_ints(out_thr, {123}) &&
              same_floats(out_kbars, {9.0f}) &&
              out_mode == 77,
          "inactive fetch must not mutate output args");

    const int thr_a[] = {16, 32};
    const float kbars_a[] = {7.0f, 4.0f, 2.0f};
    check(!dp_moe_kbar_schedule_set(nullptr, 0, kbars_a, 1, 1),
          "null thresholds must be rejected");
    check(!dp_moe_kbar_schedule_set(thr_a, 2, nullptr, 3, 1),
          "null kbars must be rejected");
    check(!dp_moe_kbar_schedule_set(thr_a, 2, kbars_a, 2, 1),
          "n_kbars must equal n_thresholds + 1");
    const int bad_thr[] = {32, 16};
    check(!dp_moe_kbar_schedule_set(bad_thr, 2, kbars_a, 3, 1),
          "thresholds must be strictly ascending");
    const float bad_kbars[] = {7.0f, -1.0f, 2.0f};
    check(!dp_moe_kbar_schedule_set(thr_a, 2, bad_kbars, 3, 1),
          "negative kbar must be rejected");
    check(!dp_moe_schedule_active(),
          "invalid sets must not activate the schedule");

    check(dp_moe_kbar_schedule_set(thr_a, 2, kbars_a, 3, 11),
          "valid schedule set should succeed");
    check(dp_moe_schedule_active(), "valid set should activate schedule");

    out_thr.clear();
    out_kbars.clear();
    out_mode = -1;
    check(dp_moe_schedule_get(&out_thr, &out_kbars, &out_mode),
          "fetch should succeed when schedule is active");
    check(same_ints(out_thr, {16, 32}) &&
              same_floats(out_kbars, {7.0f, 4.0f, 2.0f}) &&
              out_mode == 11,
          "fetch should return the full active schedule");

    out_mode = -1;
    check(dp_moe_schedule_get(nullptr, nullptr, &out_mode),
          "fetch should allow skipped vector outputs");
    check(out_mode == 11, "mode-only fetch should return allocator mode");

    out_thr[0] = 999;
    out_kbars[0] = 999.0f;
    out_mode = -1;
    check(dp_moe_schedule_get(&out_thr, &out_kbars, &out_mode),
          "second fetch should succeed");
    check(same_ints(out_thr, {16, 32}) &&
              same_floats(out_kbars, {7.0f, 4.0f, 2.0f}) &&
              out_mode == 11,
          "fetch should return copies, not aliases to internal state");

    const int thr_b[] = {8};
    const float kbars_b[] = {3.0f, 2.5f};
    check(dp_moe_kbar_schedule_set(thr_b, 1, kbars_b, 2, 22),
          "overwrite schedule set should succeed");
    check(dp_moe_schedule_get(&out_thr, &out_kbars, &out_mode),
          "fetch after overwrite should succeed");
    check(same_ints(out_thr, {8}) &&
              same_floats(out_kbars, {3.0f, 2.5f}) &&
              out_mode == 22,
          "fetch should return the latest complete schedule");

    // Stress the "same atomic snapshot" contract. The reader must never
    // observe thresholds from one schedule with kbars/mode from another.
    std::atomic<bool> start{false};
    std::atomic<bool> failed{false};
    std::thread writer([&] {
        while (!start.load(std::memory_order_acquire)) {
        }
        for (int i = 0; i < 10000; ++i) {
            if ((i & 1) == 0) {
                if (!dp_moe_kbar_schedule_set(thr_a, 2, kbars_a, 3, 11)) {
                    failed.store(true, std::memory_order_release);
                }
            } else {
                if (!dp_moe_kbar_schedule_set(thr_b, 1, kbars_b, 2, 22)) {
                    failed.store(true, std::memory_order_release);
                }
            }
        }
    });

    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r) {
        readers.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) {
            }
            for (int i = 0; i < 10000; ++i) {
                std::vector<int> thr;
                std::vector<float> kbars;
                int mode = 0;
                if (!dp_moe_schedule_get(&thr, &kbars, &mode) ||
                    !snapshot_is_a_or_b(thr, kbars, mode)) {
                    failed.store(true, std::memory_order_release);
                    return;
                }
            }
        });
    }
    start.store(true, std::memory_order_release);
    writer.join();
    for (auto & t : readers) {
        t.join();
    }
    check(!failed.load(std::memory_order_acquire),
          "concurrent fetch must return a complete schedule snapshot");

    dp_moe_schedule_clear();
    check(!dp_moe_schedule_active(), "final clear should make schedule inactive");
    check(!dp_moe_schedule_get(&out_thr, &out_kbars, &out_mode),
          "fetch after final clear should fail");

    if (g_failures != 0) {
        std::fprintf(stderr, "dp_moe-schedule-fetch-test: %d failure(s)\n",
                     g_failures);
        return 1;
    }

    std::fprintf(stderr, "dp_moe-schedule-fetch-test: OK\n");
    return 0;
}
