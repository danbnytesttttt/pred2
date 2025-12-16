#pragma once

#include "sdk.hpp"
#include <array>
#include <unordered_map>
#include <algorithm>

namespace HybridPred
{
    struct OpportunityDecision
    {
        bool should_cast;
        float final_hit_chance;
        const char* reason;
    };

    class HitChanceHistory
    {
    public:
        enum class Trend { Stable, Rising, Falling };

    private:
        static constexpr int BUFFER_SIZE = 32; // ~0.25s history at 120 FPS
        std::array<float, BUFFER_SIZE> buffer_{};
        int head_ = 0;
        int count_ = 0;

        float start_time_ = 0.f;
        float last_update_time_ = 0.f;
        float peak_hc_ = 0.f;

        Trend current_trend_ = Trend::Stable;

    public:
        void reset(float game_time)
        {
            head_ = 0;
            count_ = 0;
            peak_hc_ = 0.f;
            start_time_ = game_time;
            last_update_time_ = game_time;
            current_trend_ = Trend::Stable;
            buffer_.fill(0.f);
        }

        void add_sample(float hc, float game_time)
        {
            // Reset if target was out of range/vision for > 0.5s
            if (count_ > 0 && (game_time - last_update_time_ > 0.5f))
            {
                reset(game_time);
            }
            last_update_time_ = game_time;

            buffer_[head_] = hc;
            head_ = (head_ + 1) % BUFFER_SIZE;
            if (count_ < BUFFER_SIZE) count_++;

            if (hc > peak_hc_) peak_hc_ = hc;

            update_trend();
        }

        void update_trend()
        {
            if (count_ < 4)
            {
                current_trend_ = Trend::Stable;
                return;
            }

            // CONSTANT: 1.5% change over 3 frames (approx 21ms)
            // This filters out micro-jitter but catches human jukes.
            constexpr float SENSITIVITY = 0.015f;

            int idx_curr = (head_ - 1 + BUFFER_SIZE) % BUFFER_SIZE;
            float val_curr = buffer_[idx_curr];

            int idx_old = (head_ - 4 + BUFFER_SIZE) % BUFFER_SIZE;
            float val_old = buffer_[idx_old];

            float delta = val_curr - val_old;

            if (delta > SENSITIVITY) current_trend_ = Trend::Rising;
            else if (delta < -SENSITIVITY) current_trend_ = Trend::Falling;
            else current_trend_ = Trend::Stable;
        }

        Trend get_trend() const { return current_trend_; }
        float get_peak() const { return peak_hc_; }
        float get_start_time() const { return start_time_; }
        int get_sample_count() const { return count_; }

        float get_average(int n_samples) const
        {
            if (count_ == 0) return 0.f;
            int samples_to_check = std::min(n_samples, count_);
            float sum = 0.f;
            for (int i = 0; i < samples_to_check; ++i)
            {
                int idx = (head_ - 1 - i + BUFFER_SIZE) % BUFFER_SIZE;
                sum += buffer_[idx];
            }
            return sum / samples_to_check;
        }
    };

    class OpportunityManager
    {
    private:
        std::unordered_map<uint32_t, HitChanceHistory> histories_;

        // THE GOLDEN RATIO CONSTANTS (DO NOT EXPOSE TO USER)
        static constexpr float BYPASS_HC = 0.85f;          // 85% - Just fire, don't be greedy
        static constexpr float MIN_PEAK_QUALITY = 0.60f;   // 60% - Minimum acceptable peak
        static constexpr float PLATEAU_THRESHOLD = 0.75f;  // 75% - Fire if stable here
        static constexpr float MAX_WAIT_TIME = 0.25f;      // 250ms - Max wait (prevents feeling "laggy")

    public:
        static OpportunityManager& get()
        {
            static OpportunityManager instance;
            return instance;
        }

        static void clear()
        {
            get().histories_.clear();
        }

        OpportunityDecision evaluate(
            game_object* target,
            float hit_chance,
            bool is_urgent,
            float game_time)
        {
            try
            {
                // CRITICAL: Validate game object before any access
                if (!target || !target->is_valid()) return { false, 0.f, "INVALID_TARGET" };

                // 1. HARD BYPASS (Speed)
                if (is_urgent) return { true, hit_chance, "URGENT_STATE" };
                if (hit_chance >= BYPASS_HC) return { true, hit_chance, "HIGH_CONFIDENCE" };

                // 2. DATA UPDATE
                uint32_t id = 0;
                try
                {
                    id = target->get_network_id();
                }
                catch (...)
                {
                    // get_network_id() crashed - use pointer as ID (truncate to 32-bit)
                    id = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(target));
                }

                auto& history = histories_[id];

                auto prev_trend = history.get_trend();
                history.add_sample(hit_chance, game_time);
                auto curr_trend = history.get_trend();

                // 3. COLD START (First 35ms)
                if (history.get_sample_count() < 5)
                {
                    if (hit_chance > 0.70f) return { true, hit_chance, "COLD_START_GOOD" };
                    return { false, hit_chance, "GATHERING_DATA" };
                }

                // 4. PEAK DETECTION (Accuracy)
                bool was_rising = (prev_trend == HitChanceHistory::Trend::Rising);
                bool is_dropping = (curr_trend == HitChanceHistory::Trend::Falling);

                if (was_rising && is_dropping && hit_chance > MIN_PEAK_QUALITY)
                {
                    return { true, hit_chance, "PEAK_DETECTED" };
                }

                // 5. PLATEAU DETECTION (Consistency)
                if (curr_trend == HitChanceHistory::Trend::Stable)
                {
                    if (history.get_average(5) > PLATEAU_THRESHOLD)
                    {
                        return { true, hit_chance, "PLATEAU_FIRE" };
                    }
                }

                // 6. TIMEOUT (Feel)
                if (game_time - history.get_start_time() > MAX_WAIT_TIME)
                {
                    if (hit_chance > 0.50f) return { true, hit_chance, "TIMEOUT_ACCEPT" };

                    if (hit_chance < 0.30f) history.reset(game_time);
                }

                return { false, hit_chance, "WAITING" };
            }
            catch (...)
            {
                // CRITICAL: Exception in OpportunityManager - fail safe by allowing cast
                return { true, hit_chance, "EXCEPTION_BYPASS" };
            }
        }
    };
}
