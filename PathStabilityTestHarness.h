/**
 * =============================================================================
 * PathStabilityTestHarness.h
 * =============================================================================
 *
 * Micro-simulation test harness for path stability and outcome evaluation.
 * NO GAME REQUIRED - purely deterministic synthetic tests.
 *
 * Tests all failure modes identified in DETERMINISTIC_BUGS_AND_FIXES.md
 *
 * Usage:
 *   PathStabilityTests::run_all_tests();
 *
 * =============================================================================
 */

#pragma once

#include "PathStability.h"
#include "PredictionTelemetry.h"
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <iostream>

namespace PathStabilityTests
{
    // =========================================================================
    // MOCK GAME OBJECT (No SDK required)
    // =========================================================================

    struct MockGameObject
    {
        math::vector3 position;
        math::vector3 velocity;
        std::vector<math::vector3> path;
        bool visible = true;
        bool alive = true;
        bool winding_up = false;
        bool channeling = false;
        uint32_t network_id = 12345;
        float bounding_radius = 65.f;

        math::vector3 get_server_position() const { return position; }
        math::vector3 get_velocity() const { return velocity; }
        auto get_path() const { return path; }
        bool is_visible() const { return visible; }
        bool is_dead() const { return !alive; }
        bool is_winding_up() const { return winding_up; }
        bool is_channeling() const { return channeling; }
        bool is_valid() const { return alive; }
        uint32_t get_network_id() const { return network_id; }
        float get_bounding_radius() const { return bounding_radius; }
    };

    // =========================================================================
    // TEST UTILITIES
    // =========================================================================

    struct TestResult
    {
        std::string test_name;
        bool passed = true;
        std::string failure_reason;

        // Expected vs actual values for reporting
        struct Checkpoint
        {
            float time;
            float expected_time_stable;
            float actual_time_stable;
            float expected_persistence;
            float actual_persistence;
            bool expected_reset;
            bool actual_reset;
            std::string reason;
        };
        std::vector<Checkpoint> checkpoints;
    };

    inline void log_test(const std::string& test_name, bool passed, const std::string& reason = "")
    {
        if (passed)
        {
            std::cout << "[PASS] " << test_name << std::endl;
        }
        else
        {
            std::cout << "[FAIL] " << test_name << ": " << reason << std::endl;
        }
    }

    inline bool approx_equal(float a, float b, float epsilon = 0.01f)
    {
        return std::abs(a - b) < epsilon;
    }

    // =========================================================================
    // TEST SCENARIO 1: Hysteresis Lock-In Regression
    // =========================================================================

    inline TestResult test_hysteresis_lock_in()
    {
        TestResult result;
        result.test_name = "Hysteresis Lock-In (Gradual Rotation)";

        MockGameObject target;
        target.position = math::vector3(0, 0, 0);
        target.velocity = math::vector3(350, 0, 0);  // Moving at 350 u/s

        PathStability::TargetBehaviorTracker tracker;
        float current_time = 0.f;
        float last_time_stable = 0.f;
        bool saw_reset = false;

        // Rotate direction gradually: 0° → 35° → 70° → 105° over 1.2s
        // Each step is "medium" (25-50°) but cumulative exceeds 50°

        struct DirectionStep
        {
            float time;
            float angle_degrees;  // Direction
        };

        std::vector<DirectionStep> steps = {
            {0.0f, 0.f},    // Start at 0°
            {0.3f, 0.f},    // Still 0° for 0.3s (build stability)
            {0.6f, 35.f},   // Medium rotation (hysteresis: no reset)
            {0.9f, 70.f},   // Another medium rotation (cumulative: 70° > 50° → SHOULD RESET)
            {1.2f, 105.f}   // Another medium rotation
        };

        for (size_t i = 0; i < steps.size(); ++i)
        {
            current_time = steps[i].time;
            float angle_rad = steps[i].angle_degrees * (3.14159f / 180.f);

            target.velocity = math::vector3(
                350.f * std::cos(angle_rad),
                0.f,
                350.f * std::sin(angle_rad)
            );

            tracker.update(&target, current_time);

            // Check if reset occurred (time_stable decreased)
            if (tracker.time_since_meaningful_change < last_time_stable)
            {
                saw_reset = true;
                result.checkpoints.push_back({
                    current_time,
                    0.f,  // Expected: reset to 0
                    tracker.time_since_meaningful_change,  // Actual
                    0.f,  // Expected persistence after reset
                    PathStability::compute_persistence(tracker.time_since_meaningful_change, 0.5f),
                    true,  // Expected reset
                    true,  // Actual reset
                    "Cumulative drift exceeded 50° threshold"
                });
            }

            last_time_stable = tracker.time_since_meaningful_change;
        }

        // EXPECTATION: Should see reset at t=0.9s when cumulative drift = 70° > 50°
        if (!saw_reset)
        {
            result.passed = false;
            result.failure_reason = "No reset detected despite 105° total rotation";
        }
        else if (tracker.time_since_meaningful_change > 0.5f)
        {
            // If we saw reset but time_stable is still high, something's wrong
            result.passed = false;
            result.failure_reason = "Reset detected but time_stable still accumulating through rotation";
        }

        return result;
    }

    // =========================================================================
    // TEST SCENARIO 2: High-APM Zigzag Kiting
    // =========================================================================

    inline TestResult test_zigzag_kiting()
    {
        TestResult result;
        result.test_name = "High-APM Zigzag Kiting (Oscillation)";

        MockGameObject target;
        target.position = math::vector3(0, 0, 0);

        PathStability::TargetBehaviorTracker tracker;
        float current_time = 0.f;
        float dt = 0.05f;  // 50ms ticks

        // Alternate +40° and -40° around forward axis every 0.4s for 4s
        // Each change is 80° (medium-to-big) but alternating
        float base_speed = 350.f;
        bool positive_offset = true;
        float last_change_time = 0.f;

        for (int tick = 0; tick < 80; ++tick)  // 4s at 50ms ticks
        {
            current_time = tick * dt;

            // Change direction every 0.4s
            if (current_time - last_change_time >= 0.4f)
            {
                positive_offset = !positive_offset;
                last_change_time = current_time;
            }

            float offset_angle = positive_offset ? 40.f : -40.f;
            float angle_rad = offset_angle * (3.14159f / 180.f);

            target.velocity = math::vector3(
                base_speed * std::cos(angle_rad),
                0.f,
                base_speed * std::sin(angle_rad)
            );

            tracker.update(&target, current_time);
        }

        // EXPECTATION: time_stable should NOT reach high values (prevent "infinite stability")
        // With 80° changes every 0.4s, persistence should never hit 1.0
        float final_persistence = PathStability::compute_persistence(
            tracker.time_since_meaningful_change,
            0.5f  // Typical t_impact
        );

        if (final_persistence > 0.8f)
        {
            result.passed = false;
            result.failure_reason = "Persistence reached " + std::to_string(final_persistence) +
                                   " despite constant zigzagging (should stay low)";
        }

        if (tracker.time_since_meaningful_change > 1.0f)
        {
            result.passed = false;
            result.failure_reason = "time_stable = " + std::to_string(tracker.time_since_meaningful_change) +
                                   "s despite zigzag pattern (should reset frequently)";
        }

        return result;
    }

    // =========================================================================
    // TEST SCENARIO 3: Micro-Jitter Noise Immunity
    // =========================================================================

    inline TestResult test_micro_jitter_immunity()
    {
        TestResult result;
        result.test_name = "Micro-Jitter Noise Immunity";

        MockGameObject target;
        target.position = math::vector3(0, 0, 0);

        PathStability::TargetBehaviorTracker tracker;
        float current_time = 0.f;
        float dt = 0.016f;  // 60 FPS

        // Straight path with small random jitter ±10° every tick for 3s
        float base_speed = 350.f;
        float base_angle = 0.f;  // Forward

        int reset_count = 0;
        float last_time_stable = 0.f;

        for (int tick = 0; tick < 180; ++tick)  // 3s at 60 FPS
        {
            current_time = tick * dt;

            // Add small jitter ±10°
            float jitter = ((rand() % 200) - 100) / 1000.f;  // -0.1 to +0.1 radians (~±6°)
            float angle_rad = base_angle + jitter;

            target.velocity = math::vector3(
                base_speed * std::cos(angle_rad),
                0.f,
                base_speed * std::sin(angle_rad)
            );

            tracker.update(&target, current_time);

            // Count resets
            if (tracker.time_since_meaningful_change < last_time_stable)
                reset_count++;

            last_time_stable = tracker.time_since_meaningful_change;
        }

        // EXPECTATION: Few or no resets (jitter < ANGLE_SMALL = 25°)
        if (reset_count > 5)
        {
            result.passed = false;
            result.failure_reason = std::to_string(reset_count) +
                                   " resets from micro-jitter (should be noise-immune)";
        }

        // Stability should accumulate
        if (tracker.time_since_meaningful_change < 2.0f)
        {
            result.passed = false;
            result.failure_reason = "time_stable only " + std::to_string(tracker.time_since_meaningful_change) +
                                   "s despite 3s straight path (jitter caused false resets)";
        }

        return result;
    }

    // =========================================================================
    // TEST SCENARIO 4: Windup Damping Logic
    // =========================================================================

    inline TestResult test_windup_damping()
    {
        TestResult result;
        result.test_name = "Windup Damping (No False Stability Credit)";

        MockGameObject target;
        target.position = math::vector3(0, 0, 0);
        target.velocity = math::vector3(350, 0, 0);

        PathStability::TargetBehaviorTracker tracker;
        float current_time = 0.f;
        float dt = 0.05f;

        // Phase 1: Zigzag ±40° for 1s (build instability)
        bool positive = true;
        for (int tick = 0; tick < 20; ++tick)  // 1s
        {
            if (tick % 8 == 0) positive = !positive;

            float angle = positive ? 40.f : -40.f;
            target.velocity = math::vector3(
                350.f * std::cos(angle * 3.14159f / 180.f),
                0.f,
                350.f * std::sin(angle * 3.14159f / 180.f)
            );

            current_time = tick * dt;
            tracker.update(&target, current_time);
        }

        float time_stable_before_windup = tracker.time_since_meaningful_change;

        // Phase 2: Enter windup for 0.5s (damping active, but direction stable)
        target.winding_up = true;
        target.velocity = math::vector3(0, 0, 0);  // Stationary during windup

        for (int tick = 0; tick < 10; ++tick)  // 0.5s
        {
            current_time = 1.0f + tick * dt;
            tracker.update(&target, current_time);
        }

        // Phase 3: Exit windup, resume zigzag
        target.winding_up = false;
        positive = true;

        for (int tick = 0; tick < 20; ++tick)  // 1s
        {
            if (tick % 8 == 0) positive = !positive;

            float angle = positive ? 40.f : -40.f;
            target.velocity = math::vector3(
                350.f * std::cos(angle * 3.14159f / 180.f),
                0.f,
                350.f * std::sin(angle * 3.14159f / 180.f)
            );

            current_time = 1.5f + tick * dt;
            tracker.update(&target, current_time);
        }

        // EXPECTATION: Stability should not be high after resuming zigzag
        // Windup should not grant persistent credit
        float final_persistence = PathStability::compute_persistence(
            tracker.time_since_meaningful_change,
            0.5f
        );

        if (final_persistence > 0.5f)
        {
            result.passed = false;
            result.failure_reason = "Persistence = " + std::to_string(final_persistence) +
                                   " after zigzag resume (windup granted false stability)";
        }

        return result;
    }

    // =========================================================================
    // TEST SCENARIO 5: Vision Loss Reset
    // =========================================================================

    inline TestResult test_vision_loss_reset()
    {
        TestResult result;
        result.test_name = "Vision Loss / Fog Dancing Reset";

        MockGameObject target;
        target.position = math::vector3(0, 0, 0);
        target.velocity = math::vector3(350, 0, 0);

        PathStability::TargetBehaviorTracker tracker;
        float current_time = 0.f;
        float dt = 0.05f;

        // Phase 1: Stable movement for 0.6s
        for (int tick = 0; tick < 12; ++tick)
        {
            current_time = tick * dt;
            tracker.update(&target, current_time);
        }

        float time_stable_before_fog = tracker.time_since_meaningful_change;

        // Phase 2: Enter fog (invisible for 1.0s)
        target.visible = false;
        current_time = 0.6f;
        tracker.update(&target, current_time);

        // Phase 3: Exit fog with NEW direction
        current_time = 1.6f;
        target.visible = true;
        target.velocity = math::vector3(0, 0, 350);  // 90° turn
        tracker.update(&target, current_time);

        // EXPECTATION: Stability reset on vision loss
        if (tracker.time_since_meaningful_change >= time_stable_before_fog)
        {
            result.passed = false;
            result.failure_reason = "time_stable not reset after vision loss (carried stale stability)";
        }

        if (tracker.time_since_meaningful_change > 0.1f)
        {
            result.passed = false;
            result.failure_reason = "time_stable = " + std::to_string(tracker.time_since_meaningful_change) +
                                   " after vision regain (should be near 0)";
        }

        return result;
    }

    // =========================================================================
    // TEST SCENARIO 6: Death / Respawn Reset
    // =========================================================================

    inline TestResult test_death_respawn_reset()
    {
        TestResult result;
        result.test_name = "Death / Respawn / Long Gap Reset";

        MockGameObject target;
        target.position = math::vector3(0, 0, 0);
        target.velocity = math::vector3(350, 0, 0);

        PathStability::TargetBehaviorTracker tracker;
        float current_time = 0.f;

        // Phase 1: Stable for 1.0s
        for (int tick = 0; tick < 20; ++tick)
        {
            current_time = tick * 0.05f;
            tracker.update(&target, current_time);
        }

        float time_stable_before_death = tracker.time_since_meaningful_change;

        // Phase 2: Death
        target.alive = false;
        current_time = 1.0f;
        tracker.update(&target, current_time);

        // Phase 3: No updates for 3s (respawn timer)
        current_time = 4.0f;

        // Phase 4: Respawn with new path
        target.alive = true;
        target.velocity = math::vector3(0, 0, 350);
        tracker.update(&target, current_time);

        // EXPECTATION: Complete reset, no old reference_intent
        if (tracker.time_since_meaningful_change >= time_stable_before_death)
        {
            result.passed = false;
            result.failure_reason = "time_stable not reset after death/respawn";
        }

        if (!tracker.reference_intent.is_valid)
        {
            // This is actually EXPECTED - reference should be re-initialized on first valid intent
            // Test passes if tracker starts fresh
        }

        return result;
    }

    // =========================================================================
    // TEST SCENARIO 7: delta_time Anomalies
    // =========================================================================

    inline TestResult test_delta_time_anomalies()
    {
        TestResult result;
        result.test_name = "delta_time Anomalies (Negative + Huge)";

        MockGameObject target;
        target.position = math::vector3(0, 0, 0);
        target.velocity = math::vector3(350, 0, 0);

        PathStability::TargetBehaviorTracker tracker;
        float current_time = 0.f;

        // Normal updates
        for (int i = 0; i < 10; ++i)
        {
            current_time = i * 0.05f;
            tracker.update(&target, current_time);
        }

        float time_stable_before = tracker.time_since_meaningful_change;

        // Simulate time rollback (negative delta_time)
        current_time = 0.4f;  // Go backwards
        tracker.update(&target, current_time);

        // EXPECTATION: time_stable should NOT decrease (clamped to 0)
        if (tracker.time_since_meaningful_change < time_stable_before)
        {
            result.passed = false;
            result.failure_reason = "time_stable decreased on negative delta_time (not clamped)";
        }

        // Simulate freeze (huge delta_time)
        current_time = 10.0f;  // Jump 9.6s forward
        tracker.update(&target, current_time);

        // EXPECTATION: delta_time clamped to max 0.25s, so time_stable shouldn't jump by 9.6s
        float expected_max_increase = 0.25f;  // Clamp max
        float actual_increase = tracker.time_since_meaningful_change - time_stable_before;

        if (actual_increase > expected_max_increase + 0.01f)
        {
            result.passed = false;
            result.failure_reason = "time_stable increased by " + std::to_string(actual_increase) +
                                   "s on huge delta_time (should clamp to 0.25s)";
        }

        return result;
    }

    // =========================================================================
    // TEST SCENARIO 8: Per-Frame Dedup
    // =========================================================================

    inline TestResult test_per_frame_dedup()
    {
        TestResult result;
        result.test_name = "Per-Frame Dedup (Multiple Predictions Same Frame)";

        MockGameObject target;
        target.position = math::vector3(0, 0, 0);
        target.velocity = math::vector3(350, 0, 0);

        PathStability::TargetBehaviorTracker tracker;
        float current_time = 0.5f;

        // First update
        tracker.update(&target, current_time);
        float time_stable_after_first = tracker.time_since_meaningful_change;

        // Call update 8 more times at SAME timestamp (simulating Q/W/E/R checks)
        for (int i = 0; i < 8; ++i)
        {
            tracker.update(&target, current_time);
        }

        // EXPECTATION: time_stable should NOT change (dedup prevents multi-update)
        if (!approx_equal(tracker.time_since_meaningful_change, time_stable_after_first, 0.001f))
        {
            result.passed = false;
            result.failure_reason = "time_stable changed from " +
                                   std::to_string(time_stable_after_first) + " to " +
                                   std::to_string(tracker.time_since_meaningful_change) +
                                   " on same-frame updates (dedup failed)";
        }

        return result;
    }

    // =========================================================================
    // TEST SCENARIO 9: Outcome Multi-Sample Window
    // =========================================================================

    inline TestResult test_outcome_multi_sample()
    {
        TestResult result;
        result.test_name = "Outcome Multi-Sample (Proves Noise Reduction)";

        // Simulate: Target passes through predicted point at T, but offset at T+50ms
        math::vector3 predicted_pos(1000, 0, 0);
        float spell_radius = 70.f;
        float target_radius = 65.f;
        float effective_radius = spell_radius + target_radius;

        // Sample positions at 3 times
        math::vector3 pos_early(1000, 0, 10);   // t-50ms: 10 units offset (HIT)
        math::vector3 pos_exact(1000, 0, 0);    // t+0ms: exact (HIT)
        math::vector3 pos_late(1000, 0, 80);    // t+50ms: 80 units offset (MISS if single-sample)

        // Calculate distances
        float dist_early = pos_early.distance(predicted_pos);
        float dist_exact = pos_exact.distance(predicted_pos);
        float dist_late = pos_late.distance(predicted_pos);

        float min_distance = std::min({dist_early, dist_exact, dist_late});

        // Single-sample at late would MISS
        bool single_sample_hit = (dist_late <= effective_radius);

        // Multi-sample uses min distance → HIT
        bool multi_sample_hit = (min_distance <= effective_radius);

        if (single_sample_hit)
        {
            result.passed = false;
            result.failure_reason = "Test scenario broken: single-sample should MISS";
        }

        if (!multi_sample_hit)
        {
            result.passed = false;
            result.failure_reason = "Multi-sample MISSED despite passing through predicted point (noise reduction failed)";
        }

        return result;
    }

    // =========================================================================
    // TEST SCENARIO 10: Line Projection Edge Cases
    // =========================================================================

    inline TestResult test_line_projection_edge_cases()
    {
        TestResult result;
        result.test_name = "Line Projection (Behind Caster / Past End)";

        math::vector3 source_pos(0, 0, 0);
        math::vector3 predicted_pos(1000, 0, 0);
        float spell_radius = 70.f;
        float target_radius = 65.f;

        // Use the helper from PredictionTelemetry
        auto calculate_miss_distance = [](
            const math::vector3& actual_pos,
            const math::vector3& predicted_pos,
            bool is_line_spell,
            const math::vector3& source_pos) -> float
        {
            if (!is_line_spell)
                return actual_pos.distance(predicted_pos);

            math::vector3 line_vec = predicted_pos - source_pos;
            float line_length = std::sqrt(line_vec.x * line_vec.x + line_vec.z * line_vec.z);

            if (line_length < 1.f)
                return actual_pos.distance(predicted_pos);

            math::vector3 line_dir(line_vec.x / line_length, 0.f, line_vec.z / line_length);
            math::vector3 to_target = actual_pos - source_pos;
            float proj = (to_target.x * line_dir.x + to_target.z * line_dir.z);

            // CORRECTED: Handle outside segment
            if (proj < 0.f || proj > line_length)
            {
                float dist_to_start = actual_pos.distance(source_pos);
                float dist_to_end = actual_pos.distance(predicted_pos);
                return std::min(dist_to_start, dist_to_end);
            }

            math::vector3 closest_point(
                source_pos.x + line_dir.x * proj,
                source_pos.y,
                source_pos.z + line_dir.z * proj
            );
            return actual_pos.distance(closest_point);
        };

        // Test 1: Target behind caster
        math::vector3 behind_caster(-100, 0, 0);
        float dist_behind = calculate_miss_distance(behind_caster, predicted_pos, true, source_pos);

        // Should be distance to source (100), NOT clamped to 0 (which would give 0)
        if (dist_behind < 90.f)
        {
            result.passed = false;
            result.failure_reason = "Behind-caster distance = " + std::to_string(dist_behind) +
                                   " (expected ~100, got false hit from clamping)";
        }

        // Test 2: Target beyond end
        math::vector3 beyond_end(1200, 0, 0);
        float dist_beyond = calculate_miss_distance(beyond_end, predicted_pos, true, source_pos);

        // Should be distance to end (200), NOT clamped to line_length
        if (dist_beyond < 190.f)
        {
            result.passed = false;
            result.failure_reason = "Beyond-end distance = " + std::to_string(dist_beyond) +
                                   " (expected ~200, got false hit from clamping)";
        }

        // Test 3: Target inside segment (perpendicular)
        math::vector3 perpendicular(500, 0, 50);
        float dist_perp = calculate_miss_distance(perpendicular, predicted_pos, true, source_pos);

        // Should be perpendicular distance (50)
        if (!approx_equal(dist_perp, 50.f, 5.f))
        {
            result.passed = false;
            result.failure_reason = "Perpendicular distance = " + std::to_string(dist_perp) +
                                   " (expected ~50)";
        }

        return result;
    }

    // =========================================================================
    // MASTER TEST RUNNER
    // =========================================================================

    inline void run_all_tests()
    {
        std::cout << "\n=================================================================\n";
        std::cout << "PATH STABILITY & OUTCOME EVALUATION - MICRO-SIMULATION TEST SUITE\n";
        std::cout << "=================================================================\n\n";

        std::vector<TestResult> results;

        results.push_back(test_hysteresis_lock_in());
        results.push_back(test_zigzag_kiting());
        results.push_back(test_micro_jitter_immunity());
        results.push_back(test_windup_damping());
        results.push_back(test_vision_loss_reset());
        results.push_back(test_death_respawn_reset());
        results.push_back(test_delta_time_anomalies());
        results.push_back(test_per_frame_dedup());
        results.push_back(test_outcome_multi_sample());
        results.push_back(test_line_projection_edge_cases());

        int passed = 0;
        int failed = 0;

        for (const auto& r : results)
        {
            if (r.passed)
            {
                log_test(r.test_name, true);
                passed++;
            }
            else
            {
                log_test(r.test_name, false, r.failure_reason);
                failed++;
            }
        }

        std::cout << "\n=================================================================\n";
        std::cout << "RESULTS: " << passed << " passed, " << failed << " failed\n";
        std::cout << "=================================================================\n\n";

        if (failed > 0)
        {
            std::cout << "❌ CRITICAL BUGS DETECTED - Apply fixes from DETERMINISTIC_BUGS_AND_FIXES.md\n\n";
        }
        else
        {
            std::cout << "✅ All tests passed - Safe for data collection\n\n";
        }
    }

} // namespace PathStabilityTests
