/**
 * =============================================================================
 * PathStability.h
 * =============================================================================
 *
 * Path stability tracking for prediction calibration.
 *
 * Treats path stability as a probability feature rather than a hard gate.
 * Tracks intent changes over time and adjusts hit chance based on path persistence.
 *
 * Priority 1 (Minimal Version):
 * - Intent change detection via first-segment direction + short-horizon point
 * - Time-since-meaningful-change tracking
 * - Knee-curve persistence scoring tied to t_impact
 * - Contextual floor (fast vs slow spells)
 * - Windup-aware damping (not freezing)
 *
 * =============================================================================
 */

#pragma once

#include <unordered_map>
#include <cmath>
#include "sdk.h"

namespace PathStability
{
    /**
     * Intent signature for robust path change detection.
     * Uses first-segment direction + short-horizon prediction instead of raw waypoints.
     */
    struct IntentSignature
    {
        math::vector3 first_segment_dir;  // Direction of first meaningful path segment
        math::vector3 short_horizon_pos;  // Predicted position at short horizon (~0.2s)
        float timestamp;                  // When this was calculated
        bool is_valid;                    // Whether we have enough data

        IntentSignature()
            : first_segment_dir(0.f, 0.f, 0.f)
            , short_horizon_pos(0.f, 0.f, 0.f)
            , timestamp(0.f)
            , is_valid(false)
        {}

        /**
         * Check if intent has changed meaningfully (with hysteresis + cumulative drift detection).
         * Uses both direction change and short-horizon drift.
         *
         * @param prev - Previous frame's intent (for frame-to-frame comparison)
         * @param ref - Reference intent from last reset (for cumulative drift detection)
         */
        bool has_changed_meaningfully(const IntentSignature& prev, const IntentSignature& ref) const
        {
            if (!is_valid || !prev.is_valid)
                return false;

            // ===================================================================
            // FRAME-TO-FRAME CHANGE (for immediate direction shifts)
            // ===================================================================
            float dot_prev = first_segment_dir.dot(prev.first_segment_dir);
            dot_prev = std::clamp(dot_prev, -1.f, 1.f);
            float angle_change_prev = std::acos(dot_prev);

            float pos_drift_prev = short_horizon_pos.distance(prev.short_horizon_pos);

            // Hysteresis thresholds
            constexpr float ANGLE_SMALL = 25.f * (3.14159f / 180.f);  // 25 degrees
            constexpr float ANGLE_BIG = 50.f * (3.14159f / 180.f);    // 50 degrees
            constexpr float DRIFT_SMALL = 30.f;  // units
            constexpr float DRIFT_BIG = 80.f;    // units

            // ===================================================================
            // CUMULATIVE DRIFT CHECK (prevents hysteresis lock-in on gradual rotation)
            // ===================================================================
            if (ref.is_valid)
            {
                float dot_ref = first_segment_dir.dot(ref.first_segment_dir);
                dot_ref = std::clamp(dot_ref, -1.f, 1.f);
                float angle_change_ref = std::acos(dot_ref);

                float pos_drift_ref = short_horizon_pos.distance(ref.short_horizon_pos);

                // If cumulative drift from reference exceeds BIG threshold, reset
                // This catches gradual rotation (e.g., 0° -> 35° -> 70° over time)
                if (angle_change_ref > ANGLE_BIG || pos_drift_ref > DRIFT_BIG)
                    return true;
            }

            // ===================================================================
            // FRAME-TO-FRAME HYSTERESIS (filters micro-jitter)
            // ===================================================================

            // Big immediate change = definitely new intent
            if (angle_change_prev > ANGLE_BIG || pos_drift_prev > DRIFT_BIG)
                return true;

            // Small immediate change = definitely same intent
            if (angle_change_prev < ANGLE_SMALL && pos_drift_prev < DRIFT_SMALL)
                return false;

            // Medium change = hysteresis (stay with previous decision)
            // NOTE: Cumulative check above prevents lock-in
            return false;
        }
    };

    /**
     * Per-target behavior tracker.
     * Maintains intent signatures and stability timing.
     */
    class TargetBehaviorTracker
    {
    public:
        IntentSignature current_intent;
        IntentSignature previous_intent;
        IntentSignature reference_intent;   // Intent at last reset (for cumulative drift detection)

        float time_since_meaningful_change;  // Time intent has been stable
        float last_update_time;
        float last_update_frame_time;       // Frame time of last update (to prevent multi-update per frame)

        // State change tracking (for resets)
        bool was_visible;                    // Previous visibility state
        bool was_alive;                      // Previous alive state
        float last_seen_time;                // Last time target was updated (for gap detection)

        // Windup state tracking
        bool was_in_windup;
        IntentSignature pre_windup_intent;

        TargetBehaviorTracker()
            : time_since_meaningful_change(0.f)
            , last_update_time(-999.f)
            , last_update_frame_time(-999.f)
            , was_visible(true)
            , was_alive(true)
            , last_seen_time(-999.f)
            , was_in_windup(false)
        {}

        /**
         * Update with current game state.
         * Handles windup-aware damping and intent change detection.
         */
        void update(game_object* target, float current_time)
        {
            if (!target || !target->is_valid())
                return;

            // Skip if already updated this frame (prevent multi-call inflation)
            // Using 1ms epsilon (covers up to 1000 FPS)
            constexpr float FRAME_EPSILON = 0.001f;
            if (current_time - last_update_frame_time < FRAME_EPSILON)
                return;

            last_update_frame_time = current_time;

            float delta_time = (last_update_time > 0.f)
                ? (current_time - last_update_time)
                : 0.05f; // Default 50ms if first update

            // Clamp delta_time to prevent negative (time rollback) or huge (freeze) values
            // Max 0.25s = 4 FPS minimum, prevents massive jumps from pauses
            delta_time = std::clamp(delta_time, 0.f, 0.25f);

            last_update_time = current_time;

            // =====================================================================
            // STATE CHANGE RESETS (Critical for data integrity)
            // =====================================================================

            bool is_visible_now = target->is_visible();
            bool is_alive_now = !target->is_dead();

            // Reset on visibility loss (target entered fog)
            if (!is_visible_now && was_visible)
            {
                time_since_meaningful_change = 0.f;
                current_intent = IntentSignature();  // Clear signature
                previous_intent = IntentSignature();
                reference_intent = IntentSignature();
            }

            // Reset on death (includes respawn detection)
            if (!is_alive_now || (!was_alive && is_alive_now))
            {
                time_since_meaningful_change = 0.f;
                current_intent = IntentSignature();
                previous_intent = IntentSignature();
                reference_intent = IntentSignature();
            }

            // Reset on large gap (>2s since last update = target was out of range/vision)
            constexpr float MAX_GAP = 2.0f;
            if (last_seen_time > 0.f && (current_time - last_seen_time) > MAX_GAP)
            {
                time_since_meaningful_change = 0.f;
                current_intent = IntentSignature();
                previous_intent = IntentSignature();
                reference_intent = IntentSignature();
            }

            // Update state tracking
            was_visible = is_visible_now;
            was_alive = is_alive_now;
            last_seen_time = current_time;

            // =====================================================================
            // INTENT CHANGE DETECTION
            // =====================================================================

            // Calculate current intent signature
            previous_intent = current_intent;
            current_intent = calculate_intent_signature(target, current_time);

            // Windup state machine
            bool in_windup_now = target->is_winding_up() || target->is_channeling();
            bool just_exited_windup = was_in_windup && !in_windup_now;

            if (just_exited_windup)
            {
                // Check if direction changed meaningfully post-windup
                if (current_intent.has_changed_meaningfully(pre_windup_intent, reference_intent))
                {
                    // Meaningful intent change after windup - reset stability
                    time_since_meaningful_change = 0.f;
                    reference_intent = current_intent;  // Update reference to new intent
                }
                // else: direction consistent, don't reset
            }
            else if (!was_in_windup && in_windup_now)
            {
                // Windup started - save pre-windup intent
                pre_windup_intent = previous_intent;
            }

            // During windup: dampen stability accumulation (don't freeze, don't ramp aggressively)
            if (in_windup_now)
            {
                // Don't reset on micro path changes (auto-pathing noise)
                // But don't let stability ramp aggressively either
                constexpr float WINDUP_DAMPING = 0.3f; // 30% normal rate
                time_since_meaningful_change += (delta_time * WINDUP_DAMPING);
            }
            else
            {
                // Normal stability accumulation
                if (current_intent.has_changed_meaningfully(previous_intent, reference_intent))
                {
                    // Intent changed - reset timer and update reference
                    time_since_meaningful_change = 0.f;
                    reference_intent = current_intent;  // Update reference to new intent
                }
                else
                {
                    // Intent stable - accumulate time
                    time_since_meaningful_change += delta_time;
                }
            }

            was_in_windup = in_windup_now;
        }

        /**
         * Calculate intent signature from current game state.
         */
        IntentSignature calculate_intent_signature(game_object* target, float current_time) const
        {
            IntentSignature sig;
            sig.timestamp = current_time;

            if (!target || !target->is_valid())
                return sig;

            auto path = target->get_path();
            math::vector3 current_pos = target->get_server_position();
            math::vector3 current_vel = target->get_velocity();

            // First-segment direction (robust to path noise)
            if (path.size() > 0)
            {
                // Find first non-trivial segment
                for (size_t i = 0; i < path.size(); ++i)
                {
                    float seg_length = path[i].distance(current_pos);
                    if (seg_length > 50.f) // Non-trivial segment
                    {
                        math::vector3 dir = path[i] - current_pos;
                        float mag = std::sqrt(dir.x * dir.x + dir.z * dir.z);
                        if (mag > 1.f)
                        {
                            sig.first_segment_dir = math::vector3(dir.x / mag, 0.f, dir.z / mag);
                            sig.is_valid = true;
                        }
                        break;
                    }
                }
            }

            // Fallback: use velocity direction if no path
            if (!sig.is_valid)
            {
                float vel_mag = std::sqrt(current_vel.x * current_vel.x + current_vel.z * current_vel.z);
                if (vel_mag > 10.f)
                {
                    sig.first_segment_dir = math::vector3(current_vel.x / vel_mag, 0.f, current_vel.z / vel_mag);
                    sig.is_valid = true;
                }
            }

            // Short-horizon predicted position (0.3s ahead, simple linear)
            // Tuned to balance fast spells (~0.2s) and slow spells (~0.8s)
            // Longer horizon captures intent better for slow spells without excessive noise for fast spells
            constexpr float SHORT_HORIZON = 0.3f;
            sig.short_horizon_pos = current_pos + current_vel * SHORT_HORIZON;

            return sig;
        }

        /**
         * Get intent change metrics for telemetry.
         */
        float get_delta_theta() const
        {
            if (!current_intent.is_valid || !previous_intent.is_valid)
                return 0.f;

            float dot = current_intent.first_segment_dir.dot(previous_intent.first_segment_dir);
            dot = std::clamp(dot, -1.f, 1.f);
            return std::acos(dot);
        }

        float get_delta_short_horizon() const
        {
            if (!current_intent.is_valid || !previous_intent.is_valid)
                return 0.f;

            return current_intent.short_horizon_pos.distance(previous_intent.short_horizon_pos);
        }
    };

    // Global per-target trackers
    static std::unordered_map<uint32_t, TargetBehaviorTracker> g_target_trackers;

    /**
     * Compute persistence score using knee-curve (smoothstep).
     *
     * @param time_stable - How long intent has been stable
     * @param t_impact - Time until spell impact
     * @return Persistence score [0, 1] where 1 = high confidence in path persistence
     */
    inline float compute_persistence(float time_stable, float t_impact)
    {
        // Required stability scales with t_impact
        // Shorter spells need less confirmation
        // Longer spells need more evidence
        float required = std::clamp(0.6f * t_impact, 0.10f, 0.40f);

        float ratio = std::clamp(time_stable / required, 0.f, 1.f);

        // Knee-curve: early stability is almost meaningless, then rapid reward
        // smoothstep(0.55, 0.95, ratio) creates harsh early penalty, fast reward near stable
        auto smoothstep = [](float edge0, float edge1, float x) {
            float t = std::clamp((x - edge0) / (edge1 - edge0), 0.f, 1.f);
            return t * t * (3.f - 2.f * t);
        };

        return smoothstep(0.55f, 0.95f, ratio);
    }

    /**
     * Apply path stability calibration to base hit chance.
     * Uses contextual floor based on t_impact.
     *
     * @param base_hc - Base hit chance from geometric prediction
     * @param persistence - Path stability score [0, 1]
     * @param t_impact - Time to impact
     * @return Calibrated hit chance
     */
    inline float apply_calibrator(float base_hc, float persistence, float t_impact)
    {
        // Contextual floor: fast spells are less sensitive to path changes
        // Smooth transition from fast (0.65) to slow (0.45) over 0.20s-0.30s range
        constexpr float FLOOR_FAST = 0.65f;
        constexpr float FLOOR_SLOW = 0.45f;
        constexpr float TRANSITION_START = 0.20f;
        constexpr float TRANSITION_DURATION = 0.10f;

        float t = std::clamp((t_impact - TRANSITION_START) / TRANSITION_DURATION, 0.f, 1.f);
        float floor = FLOOR_FAST + (FLOOR_SLOW - FLOOR_FAST) * t;

        // Apply calibration: finalHC = baseHC * (floor + (1-floor) * persistence)
        float calibrated = base_hc * (floor + (1.f - floor) * persistence);

        return calibrated;
    }

    /**
     * Update tracker for target and compute persistence.
     * Main entry point called from prediction system.
     *
     * @return Persistence score [0, 1]
     */
    inline float update_and_get_persistence(game_object* target, float t_impact, float current_time)
    {
        if (!target || !target->is_valid())
            return 1.f; // Default to full confidence if invalid

        uint32_t target_id = target->get_network_id();
        auto& tracker = g_target_trackers[target_id];

        // Update tracker with current state
        tracker.update(target, current_time);

        // Compute persistence based on time stable
        float persistence = compute_persistence(tracker.time_since_meaningful_change, t_impact);

        return persistence;
    }

    /**
     * Get current tracker for telemetry (read-only).
     */
    inline const TargetBehaviorTracker* get_tracker(game_object* target)
    {
        if (!target || !target->is_valid())
            return nullptr;

        uint32_t target_id = target->get_network_id();
        auto it = g_target_trackers.find(target_id);
        if (it == g_target_trackers.end())
            return nullptr;

        return &it->second;
    }

} // namespace PathStability
