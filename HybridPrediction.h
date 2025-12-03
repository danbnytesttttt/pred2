#pragma once

#include "sdk.hpp"
#include "StandalonePredictionSDK.h"  // MUST be included AFTER sdk.hpp for compatibility
#include "EdgeCaseDetection.h"
#include "PredictionTelemetry.h"
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <memory>

// Enable telemetry: Set to 1 to track pattern detection stats (prints on game end)
#define ENABLE_PATTERN_TELEMETRY 0

/**
 * =============================================================================
 * HYBRID PROJECTILE PREDICTION SYSTEM
 * =============================================================================
 *
 * A mathematically rigorous prediction system combining:
 * 1. Deterministic Physics (Kinematic Reachability Analysis)
 * 2. Probabilistic Behavior Modeling (Adaptive Learning)
 * 3. Bayesian Fusion (Optimal Aim Point Selection)
 *
 * Mathematical Foundation:
 * -----------------------
 *
 * 1. PHYSICS COMPONENT (Deterministic):
 *    - Computes the reachable region R(t) at projectile arrival time t
 *    - R(t) = {p : ||p - p₀|| ≤ v_max * t + ½ * a_max * t²}
 *    - Accounts for turn rate: θ_max = ω * t
 *    - Hit Probability: P_physics = Area(projectile ∩ R(t)) / Area(R(t))
 *
 * 2. BEHAVIOR COMPONENT (Probabilistic):
 *    - Builds a probability density function from movement history
 *    - P(direction | context) using exponential decay weighting
 *    - Factors: dodge patterns, CS timing, HP pressure, animation locks
 *    - Hit Probability: P_behavior = ∫∫ PDF(x,y) dx dy over hit region
 *
 * 3. HYBRID FUSION (Bayesian):
 *    - HitChance = P_physics × P_behavior × Confidence
 *    - Confidence = f(distance, latency, mobility, animation_lock)
 *    - Optimal aim point: arg max_{p} HitChance(p)
 *
 * =============================================================================
 */

namespace HybridPred
{
    // =========================================================================
    // MATHEMATICAL CONSTANTS AND CONFIGURATION
    // =========================================================================

    constexpr float PI = 3.14159265358979323846f;
    constexpr float EPSILON = 1e-6f;

    // Movement tracking parameters
    constexpr int MOVEMENT_HISTORY_SIZE = 100;      // Track last N positions
    constexpr float MOVEMENT_SAMPLE_RATE = 0.05f;   // Sample every 50ms
    constexpr float BEHAVIOR_DECAY_RATE = 0.95f;    // Exponential decay factor
    constexpr int MIN_SAMPLES_FOR_BEHAVIOR = 35;    // Minimum 1.75 seconds of observation before trusting behavior patterns

    // Tracker cleanup parameters
    constexpr float TRACKER_TIMEOUT = 30.0f;        // Remove trackers after 30s when target doesn't exist

    // Physics parameters (REVERTED: Use battle-tested values until empirically verified)
    constexpr float DEFAULT_TURN_RATE = 2.0f * PI;  // radians/second
    constexpr float DEFAULT_ACCELERATION = 1200.0f; // units/s² (standard, tested)
    constexpr float DEFAULT_DECELERATION = 2000.0f; // units/s² (standard, tested)

    // Human reaction time parameters (CRITICAL for realistic predictions)
    constexpr float HUMAN_REACTION_TIME = 0.20f;    // Balanced reaction time (200ms - pros are ~150ms, avg ~250ms)
    constexpr float MIN_REACTION_TIME = 0.15f;      // Fast reactions (pros, expecting the spell)
    constexpr float MAX_REACTION_TIME = 0.35f;      // Slow reactions (distracted, teamfight)

    // Confidence parameters
    constexpr float CONFIDENCE_DISTANCE_DECAY = 0.00005f;  // Per-unit distance penalty (REDUCED 10x from 0.0005)
    constexpr float CONFIDENCE_LATENCY_FACTOR = 0.01f;    // Per-ms latency penalty
    constexpr float ANIMATION_LOCK_CONFIDENCE_BOOST = 0.3f; // Boost when immobile

    // =========================================================================
    // HELPER FUNCTIONS
    // =========================================================================

    /**
     * Get effective move speed accounting for CC and animation locks
     *
     * CRITICAL FIX: Now considers prediction_time to calculate partial freedom.
     * If target is locked for 0.1s but spell lands in 0.5s, they can move for 0.4s.
     * Returns scaled speed based on the % of time they're free to move.
     *
     * @param target The target to check
     * @param prediction_time Time until spell arrives (0 = instant check, returns binary)
     * @return Effective move speed (scaled by free time ratio)
     */
    inline float get_effective_move_speed(game_object* target, float prediction_time = 0.f)
    {
        if (!target || !target->is_valid())
            return 0.f;

        float base_speed = target->get_move_speed();

        // 1. HARD CC CHECK (Stun, Snare, etc.)
        bool is_immobile = target->has_buff_of_type(buff_type::stun) ||
            target->has_buff_of_type(buff_type::snare) ||
            target->has_buff_of_type(buff_type::charm) ||
            target->has_buff_of_type(buff_type::fear) ||
            target->has_buff_of_type(buff_type::taunt) ||
            target->has_buff_of_type(buff_type::suppression) ||
            target->has_buff_of_type(buff_type::knockup) ||
            target->has_buff_of_type(buff_type::knockback);

        if (is_immobile)
        {
            // No prediction time = binary check (backward compat)
            if (prediction_time <= 0.001f) return 0.f;

            // Calculate how long CC lasts vs spell flight time
            float current_time = (g_sdk && g_sdk->clock_facade) ? g_sdk->clock_facade->get_game_time() : 0.f;
            float max_cc_end = 0.f;

            auto buffs = target->get_buffs();
            for (auto* buff : buffs)
            {
                if (!buff || !buff->is_active()) continue;
                buff_type t = buff->get_type();
                if (t == buff_type::stun || t == buff_type::snare || t == buff_type::charm ||
                    t == buff_type::fear || t == buff_type::taunt || t == buff_type::suppression ||
                    t == buff_type::knockup || t == buff_type::knockback)
                {
                    if (buff->get_end_time() > max_cc_end)
                        max_cc_end = buff->get_end_time();
                }
            }

            float arrival_time_abs = current_time + prediction_time;
            if (max_cc_end >= arrival_time_abs) return 0.f;  // CC lasts until spell lands

            // They wake up before spell lands - scale by free time
            float time_locked = std::max(0.f, max_cc_end - current_time);
            float time_free = prediction_time - time_locked;
            return base_speed * (time_free / prediction_time);
        }

        // 2. ANIMATION LOCK CHECK (Auto-Attack / Cast)
        // FIX: Calculate "Free Time" exactly like CC
        // Do NOT assume speed=0 just because they are attacking NOW
        bool locked_state = (is_auto_attacking(target) || is_casting_spell(target) || is_channeling(target));

        if (locked_state && !target->is_moving())
        {
            // No prediction time = binary check (backward compat)
            if (prediction_time <= 0.001f) return 0.f;

            // Get exact time until animation/windup finishes
            float remaining_lock = get_remaining_lock_time(target);

            // If lock lasts longer than flight time, they are truly stationary
            if (remaining_lock >= prediction_time) return 0.f;

            // Otherwise, they will move for part of the flight
            // Scale speed by the % of time they are free
            float time_free = prediction_time - remaining_lock;
            return base_speed * (time_free / prediction_time);
        }

        return base_speed;
    }

    /**
     * Compute adaptive decay rate based on target mobility
     * Fast-moving targets: faster decay (recent data more important)
     * Slow-moving targets: slower decay (longer history matters)
     */
    inline float get_adaptive_decay_rate(float move_speed)
    {
        // High mobility (>450): 0.90 decay (half-life = 0.34s)
        if (move_speed > 450.f)
            return 0.90f;

        // Normal mobility (350-450): 0.95 decay (half-life = 0.675s)
        if (move_speed > 350.f)
            return 0.95f;

        // Low mobility (<350): 0.97 decay (half-life = 1.15s)
        return 0.97f;
    }

    /**
     * Fuse physics and behavior probabilities using weighted geometric mean
     * When behavior data is sparse, trust physics more. When abundant, blend equally.
     *
     * Formula: P = physics^w × behavior^(1-w) × confidence
     * where w = physics weight based on sample quality and staleness
     *
     * SPECIAL CASE: When physics = 1.0 (physically impossible to dodge),
     * return guaranteed hit regardless of behavior (flash/dash handled by edge cases)
     */
    inline float fuse_probabilities(float physics_prob, float behavior_prob, float confidence, size_t sample_count, float time_since_update = 0.f, float move_speed = 350.f, float distance = 1000.f)
    {
        // CRITICAL: Guaranteed hit override
        // If physics says escape is physically impossible (>= 99%), guarantee the hit
        // Behavior should NOT reduce this (flash/dash handled by edge case confidence penalties)
        constexpr float GUARANTEED_THRESHOLD = 0.99f;
        if (physics_prob >= GUARANTEED_THRESHOLD)
        {
            // Physical escape impossible - return guaranteed hit
            // Still apply confidence for edge cases (flash, spell shield, etc.)
            return std::min(1.0f, confidence);  // Max 1.0, reduced only by edge cases
        }

        // Determine fusion weight based on behavior sample quality
        // PHILOSOPHY: Physics = Constraint (geometry), Behavior = Suggestion (patterns)
        // Physics should dominate slightly to prevent bad behavior reads from vetoing trapped targets
        float physics_weight = 0.60f;  // Default: physics-favored (60/40)

        // If we have very few behavior samples, trust physics heavily
        if (sample_count < MIN_SAMPLES_FOR_BEHAVIOR)
        {
            // Ramp from 0.85 (no samples) down to 0.60 (minimum samples)
            // 0 samples = pure physics, 35 samples = start incorporating behavior
            float factor = static_cast<float>(sample_count) / MIN_SAMPLES_FOR_BEHAVIOR;
            physics_weight = 0.85f - 0.25f * factor;  // 0.85 → 0.60
        }
        else if (sample_count < MIN_SAMPLES_FOR_BEHAVIOR * 2)
        {
            // Ramp from 0.60 down to 0.55 (still physics-favored)
            // 35-70 samples: physics leads, behavior supplements
            float factor = static_cast<float>(sample_count - MIN_SAMPLES_FOR_BEHAVIOR) / MIN_SAMPLES_FOR_BEHAVIOR;
            physics_weight = 0.60f - 0.05f * factor;  // 0.60 → 0.55
        }
        else
        {
            // Abundant data (70+ samples = 3.5+ seconds): physics still leads
            // 55% physics, 45% behavior - physics is constraint, behavior is suggestion
            physics_weight = 0.55f;
        }

        // MOBILITY FACTOR: Fast targets are more reactive and unpredictable
        // Trust physics more for fast targets, behavior more for slow targets
        // Fast targets can execute dodges quickly, making behavior patterns less reliable
        if (move_speed > 380.f)
        {
            // Fast target: increase physics weight
            // 380 → +0.0, 430 → +0.125, 480+ → +0.25
            float speed_factor = std::min((move_speed - 380.f) / 100.f, 1.0f);
            physics_weight = std::min(physics_weight + speed_factor * 0.25f, 0.75f);
        }
        else if (move_speed < 330.f)
        {
            // Slow target: can trust behavior more
            // 330 → -0.0, 280 → -0.05, 230 → -0.1
            float slow_factor = std::min((330.f - move_speed) / 100.f, 1.0f);
            physics_weight = std::max(physics_weight - slow_factor * 0.1f, 0.3f);
        }

        // Staleness detection: If velocity data hasn't updated recently, increase physics weight
        // This handles cases where behavior tracker has stale data (target in fog, networking issues, etc.)
        if (time_since_update > 0.5f)
        {
            // Ramp physics weight up based on staleness
            // 0.5s → +0.1, 1.0s → +0.2, 1.5s+ → +0.3 (capped at 0.8 total)
            float staleness_penalty = std::min(time_since_update - 0.5f, 1.0f) * 0.3f;
            physics_weight = std::min(physics_weight + staleness_penalty, 0.8f);
        }

        // HIGH-PHYSICS BOOST: When physics is very confident, trust it more (but not exclusively)
        // Physics is MATH-based (geometry), Behavior is PATTERN-based (player tendencies)
        // Cap at 80% to respect behavior while still favoring good geometry
        // 80/20 split: behavior still influences (20%) but doesn't override obvious shots
        if (physics_prob > 0.85f)
        {
            // Scale boost: 85% → +0.0, 90% → +0.10, 95%+ → +0.20
            float high_physics_boost = std::min((physics_prob - 0.85f) / 0.10f, 1.0f) * 0.20f;
            physics_weight = std::min(physics_weight + high_physics_boost, 0.80f);  // Capped at 80% (was 95%, then 75%)
        }

        // CLOSE-RANGE BOOST: At point-blank, trust physics heavily (path prediction is precise)
        // Narrow skillshots (Pyke Q: 70u radius) need tight aim - behavior can pull wide
        // <200u: boost physics to 85% | 200-400u: gradual reduction | >400u: no boost
        if (distance < 400.f)
        {
            float close_range_factor = std::max(0.f, (400.f - distance) / 200.f);  // 1.0 at 0u, 0.0 at 400u
            float close_range_boost = close_range_factor * 0.25f;  // Up to +25% physics weight
            physics_weight = std::min(physics_weight + close_range_boost, 0.90f);  // Cap at 90% for point-blank
        }

        // PHYSICS AS HARD CONSTRAINT (VETO LOGIC):
        // If physics says a position is impossible (0.0), the result must be 0.0
        // Physics acts as a constraint (geometry), Behavior refines within valid region
        //
        // OLD BUG: Linear interpolation allowed behavior to "save" impossible predictions
        //   Example: physics=0.0, behavior=0.9 → 0.6*0.0 + 0.4*0.9 = 0.36 (WRONG!)
        //
        // NEW: Physics multiplies the weighted behavior contribution
        //   Formula: fused = physics × (physics_weight + (1-physics_weight) × behavior)
        //   Example: physics=0.0, behavior=0.9 → 0.0 × (0.6 + 0.4*0.9) = 0.0 (CORRECT!)
        //   Example: physics=1.0, behavior=0.5 → 1.0 × (0.6 + 0.4*0.5) = 0.8 (balanced)
        //   Example: physics=0.5, behavior=0.8 → 0.5 × (0.6 + 0.4*0.8) = 0.46 (smooth)
        float fused = physics_prob * (physics_weight + (1.0f - physics_weight) * behavior_prob);

        // Apply confidence as a multiplier
        // Lower safety floor to allow proper penalization from edge cases
        float min_confidence = 0.05f;  // Minimal floor, let edge cases reduce confidence properly
        float effective_confidence = std::max(confidence, min_confidence);

        return fused * effective_confidence;
    }

    // =========================================================================
    // DATA STRUCTURES
    // =========================================================================

    /**
     * Movement snapshot at a specific time
     */
    struct MovementSnapshot
    {
        math::vector3 position;
        math::vector3 velocity;
        float timestamp;
        bool is_auto_attacking;
        bool is_casting;
        bool is_dashing;
        bool is_cced;
        float hp_percent;

        MovementSnapshot() : position{}, velocity{}, timestamp(0.f),
            is_auto_attacking(false), is_casting(false), is_dashing(false),
            is_cced(false), hp_percent(100.f) {
        }
    };

    /**
     * Dodge pattern statistics
     */
    /**
     * Bayesian pattern trust tracking
     * Tracks how often our pattern predictions are correct using Beta-Binomial model
     */
    struct PatternTrust
    {
        float alpha = 5.0f;   // Prior + hits (start with 50% trust)
        float beta = 5.0f;    // Prior + misses

        float get_trust() const {
            return alpha / (alpha + beta);
        }

        void observe_correct() { alpha += 1.0f; }
        void observe_incorrect() { beta += 1.0f; }

        void decay() {
            // Decay toward prior (0.5 trust) each game
            constexpr float prior_a = 5.0f, prior_b = 5.0f;
            constexpr float decay_factor = 0.95f;
            alpha = prior_a + (alpha - prior_a) * decay_factor;
            beta = prior_b + (beta - prior_b) * decay_factor;
        }
    };

    struct DodgePattern
    {
        float left_dodge_frequency;      // [0,1] How often target dodges left
        float right_dodge_frequency;     // [0,1] How often target dodges right
        float forward_frequency;         // [0,1] Forward movement tendency
        float backward_frequency;        // [0,1] Backward movement tendency
        float juke_interval_mean;        // Average time between direction changes
        float juke_interval_variance;    // Variance in juke timing
        float linear_continuation_prob;  // Probability of continuing straight
        float reaction_delay;            // Average reaction time (ms)

        // Pattern repetition detection
        std::vector<int> juke_sequence;          // Last 8 direction changes: -1=left, 0=straight, 1=right
        int last_recorded_move;                  // Last move that was recorded to sequence (for event-based filtering)
        float pattern_confidence;                // [0,1] Confidence in detected pattern
        math::vector3 predicted_next_direction;  // Unit vector of predicted next move
        bool has_pattern;                        // True if repeating pattern detected
        float last_pattern_update_time;          // Timestamp of last pattern update (for expiration)

        // Bayesian pattern trust (NEW)
        PatternTrust pattern_trust;              // Tracks how reliable our pattern predictions are
        int last_predicted_juke;                 // -1=left, 0=none, 1=right - what we predicted
        bool awaiting_juke_result;               // True if we made a prediction and are waiting to see result

        // N-Gram (Markov) pattern recognition
        // Tracks transitions: given previous move, what's the probability of each next move?
        // Key: previous move (-1, 0, 1), Value: map of next move to count
        std::map<int, std::map<int, int>> ngram_transitions;

        // Get N-Gram probability for a predicted move given current state
        float get_ngram_probability(int predicted_move) const
        {
            if (juke_sequence.empty()) return 0.33f;  // No data

            int last_move = juke_sequence.back();
            auto it = ngram_transitions.find(last_move);
            if (it == ngram_transitions.end()) return 0.33f;  // No transitions from this state

            const auto& next_counts = it->second;
            int total = 0;
            int target_count = 0;

            for (const auto& pair : next_counts)
            {
                total += pair.second;
                if (pair.first == predicted_move)
                    target_count = pair.second;
            }

            if (total == 0) return 0.33f;
            return static_cast<float>(target_count) / static_cast<float>(total);
        }

        // Juke magnitude tracking - actual lateral displacement in units
        // This tells us HOW FAR they juke, not just which direction
        std::deque<float> juke_magnitudes;       // Last N juke distances in units
        float average_juke_magnitude;            // Running average for predictions

        // Get observed juke magnitude, or default based on speed
        float get_juke_magnitude(float move_speed) const
        {
            if (juke_magnitudes.size() >= 3)
                return average_juke_magnitude;
            // Default: assume ~0.3s of lateral movement at current speed
            return move_speed * 0.3f;
        }

        DodgePattern() : left_dodge_frequency(0.05f), right_dodge_frequency(0.05f),
            forward_frequency(0.9f), backward_frequency(0.0f),
            juke_interval_mean(0.5f), juke_interval_variance(0.1f),
            linear_continuation_prob(0.95f), reaction_delay(200.f),
            pattern_confidence(0.f), predicted_next_direction{}, has_pattern(false),
            last_pattern_update_time(0.f), pattern_trust{}, last_predicted_juke(0),
            awaiting_juke_result(false), average_juke_magnitude(0.f), last_recorded_move(0) {
        }
    };

    /**
     * CS (Creep Score) behavior patterns
     * NOTE: Currently not implemented - reserved for future enhancement
     * TODO: Track minion proximity and last-hit timing to predict CS movements
     */
     // struct CSPattern { ... }; // Removed - unimplemented feature

     /**
      * Reachable region (physics-based)
      */
    struct ReachableRegion
    {
        math::vector3 center;            // Center of reachable region
        float max_radius;                // Maximum reachable distance
        std::vector<math::vector3> boundary_points; // Discretized boundary
        float area;                      // Total reachable area
        math::vector3 velocity;          // Target velocity (for momentum weighting)
        float pathable_ratio;            // Fraction of region that's walkable (0-1)
                                         // Low ratio = wall-hugging = higher hit chance

        ReachableRegion() : center{}, max_radius(0.f), area(0.f), velocity{}, pathable_ratio(1.0f) {}
    };

    /**
     * Probability density function for behavior prediction
     *
     * NOTE: 32×32 grid with 25u cells provides good balance of performance vs precision.
     * Future enhancement: Consider dynamic grid resolution based on prediction time:
     *   - Short predictions (<0.5s): 48×48 or 64×64 for fine-grained juke detection
     *   - Long predictions (>2s): 32×32 sufficient (broad movement patterns)
     * Trade-off: 64×64 = 4x memory and computation cost (4096 vs 1024 cells)
     */
    struct BehaviorPDF
    {
        // 2D grid-based PDF (discretized for efficiency)
        static constexpr int GRID_SIZE = 32;  // 32×32 = 1024 cells, 800×800u coverage
        float cell_size;  // Dynamic cell size (units per cell) - adjusted per prediction

        float pdf_grid[GRID_SIZE][GRID_SIZE];
        math::vector3 origin;            // Grid origin (center)
        float total_probability;         // Normalization factor

        BehaviorPDF() : cell_size(25.0f), origin{}, total_probability(0.f)
        {
            for (int i = 0; i < GRID_SIZE; ++i)
                for (int j = 0; j < GRID_SIZE; ++j)
                    pdf_grid[i][j] = 0.f;
        }

        // Sample PDF at world position
        float sample(const math::vector3& world_pos) const;

        // Normalize PDF so total probability = 1
        void normalize();

        // Add weighted sample to PDF
        void add_weighted_sample(const math::vector3& pos, float weight);
    };

    /**
     * Opportunistic casting opportunity window
     * Tracks recent predictions to detect peak opportunities
     */
    struct OpportunityWindow
    {
        std::deque<std::pair<float, float>> history;  // (timestamp, hit_chance) last 3s
        float peak_hit_chance;                         // Best hit_chance seen in window
        float peak_timestamp;                          // When peak occurred
        float window_start_time;                       // When tracking began for this spell
        float last_hit_chance;                         // Last hit_chance (for reset detection)

        OpportunityWindow() : peak_hit_chance(0.f), peak_timestamp(0.f), window_start_time(0.f), last_hit_chance(0.f) {}

        void update(float current_time, float hit_chance);
        bool is_peak_opportunity(float current_time, float hit_chance, float adaptive_threshold, float elapsed_time, float patience_window) const;
        float get_adaptive_threshold(float base_threshold, float elapsed_time) const;
    };

    /**
     * Complete prediction result
     */
    struct HybridPredictionResult
    {
        math::vector3 cast_position;     // Optimal aim point (second cast for vector spells)
        math::vector3 first_cast_position; // First cast position for vector spells (Viktor E, Rumble R, Irelia E)
        float hit_chance;                // Combined hit probability [0,1]
        float physics_contribution;      // Physics component [0,1]
        float behavior_contribution;     // Behavior contribution [0,1]
        float confidence_score;          // Confidence modifier [0,1]

        // Opportunistic casting signals (NEW)
        bool is_peak_opportunity;        // True if this is a local maximum and declining
        float opportunity_score;         // [0-1] How good is this moment relative to recent history?
        float adaptive_threshold;        // Threshold adjusted for time waited (decays from base)

        // Debugging/analysis data
        ReachableRegion reachable_region;
        BehaviorPDF behavior_pdf;
        std::string reasoning;           // Mathematical explanation

        // Detailed telemetry debug data (populated during prediction)
        PredictionTelemetry::PredictionEvent telemetry_data;

        bool is_valid;

        HybridPredictionResult() : cast_position{}, first_cast_position{}, hit_chance(0.f),
            physics_contribution(0.f), behavior_contribution(0.f),
            confidence_score(0.f), is_peak_opportunity(false), opportunity_score(0.f),
            adaptive_threshold(0.f), is_valid(false) {
        }
    };

    // =========================================================================
    // BEHAVIOR TRACKER (Per-Target Learning)
    // =========================================================================

    /**
     * Tracks movement patterns for a specific target
     */
    class TargetBehaviorTracker
    {
    private:
        uint32_t network_id_;  // Store ID instead of raw pointer to prevent dangling pointer crashes
        std::deque<MovementSnapshot> movement_history_;
        DodgePattern dodge_pattern_;
        float last_update_time_;
        bool was_visible_last_update_ = true;  // Track fog emergence

        // Direction change tracking
        std::vector<float> direction_change_times_;
        std::vector<float> direction_change_angles_;

        // Average turn angle tracking (lightweight juke detection)
        std::deque<float> recent_turn_angles_;  // Last 6-8 turn angles
        float average_turn_angle_ = 0.f;        // Running average for quick confidence checks
        bool recent_hard_juke_ = false;         // True if sharp turn (>45°) detected in last 5 samples

        // Auto-attack tracking
        float last_aa_time_;
        std::vector<float> post_aa_movement_delays_;

        // PDF caching (avoid rebuilding for multiple spells on same frame)
        mutable BehaviorPDF cached_pdf_;
        mutable float cached_prediction_time_;
        mutable float cached_move_speed_;
        mutable float cached_timestamp_;

        // Opportunistic casting tracking (per spell slot)
        mutable std::unordered_map<int, OpportunityWindow> opportunity_windows_;

        // Smoothed velocity to reduce jitter from spam clicking
        math::vector3 smoothed_velocity_;
        int zero_velocity_frames_ = 0;  // Counter for stop command buffer (prevent 1-frame stop exploits)

        // Event-driven sampling: Track last path endpoint to detect new clicks
        math::vector3 last_path_endpoint_;
        bool has_last_path_endpoint_ = false;

        // Dynamic acceleration measurement (per-target)
        // Start near measured averages (~20-25k), then adapt per-target
        float measured_acceleration_ = 15000.0f;  // Slightly conservative start
        float measured_deceleration_ = 20000.0f;  // Decel measured slightly higher
        int accel_sample_count_ = 0;              // Number of acceleration samples
        int decel_sample_count_ = 0;              // Number of deceleration samples
        float last_measured_speed_ = 0.f;         // Previous frame's speed for delta calculation

    public:
        TargetBehaviorTracker(game_object* target);

        // Update tracking data (call every frame)
        // Takes fresh pointer resolved from object_manager to prevent dangling pointer
        void update(game_object* current_target_ptr);

        // Get learned patterns
        const DodgePattern& get_dodge_pattern() const { return dodge_pattern_; }
        const std::deque<MovementSnapshot>& get_history() const { return movement_history_; }

        // Analyze movement to detect patterns
        void analyze_patterns();

        // Build probability distribution for future position
        BehaviorPDF build_behavior_pdf(float prediction_time, float move_speed) const;

        // Check if target is in animation lock
        bool is_animation_locked() const;

        // Get current velocity
        math::vector3 get_current_velocity() const;

        // Get last update time (for staleness detection)
        float get_last_update_time() const { return last_update_time_; }

        // Get average turn angle (lightweight juke detection)
        // Low angle (< 15°) = running straight, High angle (> 60°) = dancing/juking
        float get_average_turn_angle() const { return average_turn_angle_; }

        // Check if target made a recent sharp turn (for confidence fast-track)
        bool has_recent_hard_juke() const { return recent_hard_juke_; }

        // Opportunistic casting - get or create window for spell slot
        OpportunityWindow& get_opportunity_window(int spell_slot) const;

        // Dynamic acceleration getters (measured from actual enemy behavior)
        float get_measured_acceleration() const { return measured_acceleration_; }
        float get_measured_deceleration() const { return measured_deceleration_; }
        bool has_measured_physics() const { return accel_sample_count_ >= 3 || decel_sample_count_ >= 3; }

    private:
        void update_dodge_pattern();
        void detect_direction_changes();
        math::vector3 compute_velocity(const MovementSnapshot& prev, const MovementSnapshot& curr) const;
    };

    // =========================================================================
    // PHYSICS PREDICTION ENGINE
    // =========================================================================

    // =========================================================================
    // MULTI-TARGET AOE OPTIMIZATION
    // =========================================================================

    /**
     * Circle for Welzl's algorithm (Minimum Enclosing Circle)
     */
    struct Circle
    {
        math::vector3 center;
        float radius;

        Circle() : center{}, radius(0.f) {}
        Circle(const math::vector3& c, float r) : center(c), radius(r) {}

        bool contains(const math::vector3& point) const
        {
            return (point - center).magnitude() <= radius + 0.01f;  // Small epsilon for float precision
        }
    };

    /**
     * Pure kinematic prediction (deterministic)
     */
    class PhysicsPredictor
    {
    public:
        /**
         * Compute reachable region at time t
         *
         * Mathematical model:
         * R(t) = {p : can_reach(p₀, v₀, p, t)}
         *
         * where can_reach considers:
         * - Maximum speed: v_max
         * - Acceleration: a = min(a_max, (v_max - v₀)/t)
         * - Turn rate constraint: Δθ ≤ ω * t
         * - Deceleration: can stop within distance
         */
        static ReachableRegion compute_reachable_region(
            const math::vector3& current_pos,
            const math::vector3& current_velocity,
            float prediction_time,
            float move_speed,
            float turn_rate = DEFAULT_TURN_RATE,
            float acceleration = DEFAULT_ACCELERATION,
            float reaction_time = HUMAN_REACTION_TIME  // Subtract reaction time for realistic dodging
        );

        /**
         * Compute deterministic position prediction (linear extrapolation)
         */
        static math::vector3 predict_linear_position(
            const math::vector3& current_pos,
            const math::vector3& current_velocity,
            float prediction_time
        );

        /**
         * Predict position by following actual game path waypoints
         * More accurate than linear for curved/multi-waypoint paths
         */
        static math::vector3 predict_on_path(
            game_object* target,
            float prediction_time
        );

        /**
         * Compute physics-based hit probability (area method)
         *
         * P_physics = (projectile_area ∩ reachable_area) / reachable_area
         * Used for circular AoE spells where uniform distribution makes sense
         */
        static float compute_physics_hit_probability(
            const math::vector3& cast_position,
            float projectile_radius,
            const ReachableRegion& reachable_region
        );

        /**
         * Compute physics-based hit probability (time-to-dodge method)
         *
         * Superior to area method for linear skillshots
         * P_physics = time_needed_to_escape / time_available_to_dodge
         *
         * Returns 1.0 if target cannot physically escape in time
         */
        static float compute_time_to_dodge_probability(
            const math::vector3& target_position,
            const math::vector3& cast_position,
            float projectile_radius,
            float target_move_speed,
            float arrival_time,
            float reaction_time = HUMAN_REACTION_TIME
        );

        /**
         * Compute time for projectile to reach target
         */
        static float compute_arrival_time(
            const math::vector3& source_pos,
            const math::vector3& target_pos,
            float projectile_speed,
            float cast_delay
        );

        /**
         * Find Minimum Enclosing Circle (MEC) for multi-target AOE
         * Uses Welzl's algorithm - O(n) expected time
         *
         * Returns optimal circle that hits the most targets with smallest radius
         * Perfect for instant AOE spells (Malphite R, Annie R, Orianna R)
         */
        static Circle compute_minimum_enclosing_circle(
            const std::vector<math::vector3>& points
        );

    private:
        // Welzl's algorithm helpers
        static Circle welzl_recursive(
            std::vector<math::vector3>& points,
            std::vector<math::vector3> boundary,
            size_t n
        );

        static Circle make_circle_from_2_points(const math::vector3& p1, const math::vector3& p2);
        static Circle make_circle_from_3_points(const math::vector3& p1, const math::vector3& p2, const math::vector3& p3);
        static float circle_circle_intersection_area(
            const math::vector3& c1, float r1,
            const math::vector3& c2, float r2
        );
    };

    // =========================================================================
    // BEHAVIOR PREDICTION ENGINE
    // =========================================================================

    /**
     * Probabilistic behavior modeling
     */
    class BehaviorPredictor
    {
    public:
        /**
         * Build probability density function from movement history
         *
         * PDF(p, t) = Σᵢ wᵢ * K(p - pᵢ(t))
         *
         * where:
         * - wᵢ = BEHAVIOR_DECAY_RATE^i (recent data weighted more)
         * - K is a kernel function (Gaussian)
         * - pᵢ(t) is predicted position from snapshot i
         */
        static BehaviorPDF build_pdf_from_history(
            const TargetBehaviorTracker& tracker,
            float prediction_time,
            float move_speed
        );

        /**
         * Compute behavior-based hit probability
         *
         * P_behavior = ∫∫ PDF(x,y) dx dy over hit region
         */
        static float compute_behavior_hit_probability(
            const math::vector3& cast_position,
            float projectile_radius,
            const BehaviorPDF& pdf
        );

        /**
         * Predict most likely position from behavior
         */
        static math::vector3 predict_from_behavior(
            const TargetBehaviorTracker& tracker,
            float prediction_time
        );

        /**
         * Apply context-aware adjustments
         * - CS patterns (low HP minions nearby)
         * - HP pressure (low HP = retreat)
         * - Animation locks (AA, spell cast)
         */
        static void apply_contextual_factors(
            BehaviorPDF& pdf,
            const TargetBehaviorTracker& tracker,
            game_object* target
        );
    };

    // =========================================================================
    // HYBRID FUSION ENGINE
    // =========================================================================

    /**
     * Combines physics and behavior using Bayesian fusion
     */
    class HybridFusionEngine
    {
    public:
        /**
         * Compute complete hybrid prediction
         *
         * HitChance(p) = P_physics(p) × P_behavior(p) × Confidence
         *
         * Optimal aim: p* = arg max_{p} HitChance(p)
         */
        static HybridPredictionResult compute_hybrid_prediction(
            game_object* source,
            game_object* target,
            const pred_sdk::spell_data& spell,
            TargetBehaviorTracker& tracker
        );

        /**
         * Compute confidence score
         *
         * Confidence = base_confidence
         *              × distance_factor
         *              × latency_factor
         *              × mobility_factor
         *              × animation_factor
         */
        static float compute_confidence_score(
            game_object* source,
            game_object* target,
            const pred_sdk::spell_data& spell,
            const TargetBehaviorTracker& tracker,
            const EdgeCases::EdgeCaseAnalysis& edge_cases
        );

        /**
         * Find optimal cast position using grid search + gradient ascent
         */
        static math::vector3 find_optimal_cast_position(
            const ReachableRegion& reachable_region,
            const BehaviorPDF& behavior_pdf,
            const math::vector3& source_pos,
            float projectile_radius,
            float confidence
        );

        /**
         * Find optimal AOE position to hit multiple targets
         * Uses Minimum Enclosing Circle (Welzl's algorithm) for fast AOE spells
         * Ideal for instant or fast AOE (< 0.25s delay): Malphite R, Annie R, Orianna R, etc.
         *
         * @param source The casting champion
         * @param targets All potential targets to consider
         * @param spell Spell data (for range/radius)
         * @param max_range Maximum cast range
         * @return Optimal cast position (may be zero vector if no valid position found)
         */
        static math::vector3 find_multi_target_aoe_position(
            game_object* source,
            const std::vector<game_object*>& targets,
            const pred_sdk::spell_data& spell,
            float max_range
        );

    private:
        // Spell-type specific prediction methods
        static HybridPredictionResult compute_circular_prediction(
            game_object* source,
            game_object* target,
            const pred_sdk::spell_data& spell,
            TargetBehaviorTracker& tracker,
            const EdgeCases::EdgeCaseAnalysis& edge_cases
        );

        static HybridPredictionResult compute_linear_prediction(
            game_object* source,
            game_object* target,
            const pred_sdk::spell_data& spell,
            TargetBehaviorTracker& tracker,
            const EdgeCases::EdgeCaseAnalysis& edge_cases
        );

        static HybridPredictionResult compute_targeted_prediction(
            game_object* source,
            game_object* target,
            const pred_sdk::spell_data& spell,
            TargetBehaviorTracker& tracker,
            const EdgeCases::EdgeCaseAnalysis& edge_cases
        );

        static HybridPredictionResult compute_vector_prediction(
            game_object* source,
            game_object* target,
            const pred_sdk::spell_data& spell,
            TargetBehaviorTracker& tracker,
            const EdgeCases::EdgeCaseAnalysis& edge_cases
        );

        static HybridPredictionResult compute_cone_prediction(
            game_object* source,
            game_object* target,
            const pred_sdk::spell_data& spell,
            TargetBehaviorTracker& tracker,
            const EdgeCases::EdgeCaseAnalysis& edge_cases,
            float cone_angle_override = 0.f  // If > 0, use this angle instead of calculating from radius
        );

        // Geometry helpers for spell shapes
        static bool point_in_capsule(
            const math::vector3& point,
            const math::vector3& capsule_start,
            const math::vector3& capsule_end,
            float capsule_radius
        );

        static float compute_capsule_reachability_overlap(
            const math::vector3& capsule_start,
            const math::vector3& capsule_direction,
            float capsule_length,
            float capsule_radius,
            const ReachableRegion& reachable_region
        );

        static float compute_capsule_behavior_probability(
            const math::vector3& capsule_start,
            const math::vector3& capsule_direction,
            float capsule_length,
            float capsule_radius,
            const BehaviorPDF& pdf
        );

        static float evaluate_hit_chance_at_point(
            const math::vector3& point,
            const ReachableRegion& reachable_region,
            const BehaviorPDF& behavior_pdf,
            float projectile_radius,
            float confidence
        );

        // Cone geometry helpers
        static bool point_in_cone(
            const math::vector3& point,
            const math::vector3& cone_origin,
            const math::vector3& cone_direction,
            float cone_angle,
            float cone_range
        );

        static float compute_cone_reachability_overlap(
            const math::vector3& cone_origin,
            const math::vector3& cone_direction,
            float cone_angle,
            float cone_range,
            const ReachableRegion& reachable_region
        );

        static float compute_cone_behavior_probability(
            const math::vector3& cone_origin,
            const math::vector3& cone_direction,
            float cone_angle,
            float cone_range,
            const BehaviorPDF& pdf
        );

        // Vector spell optimization helpers
        struct VectorConfiguration
        {
            math::vector3 first_cast_position{};
            math::vector3 cast_position{};
            float hit_chance = 0.f;
            float physics_prob = 0.f;
            float behavior_prob = 0.f;
        };

        static VectorConfiguration optimize_vector_orientation(
            game_object* source,
            game_object* target,
            const math::vector3& predicted_target_pos,
            const ReachableRegion& reachable_region,
            const BehaviorPDF& behavior_pdf,
            const pred_sdk::spell_data& spell,
            float confidence,
            size_t sample_count,
            float time_since_update = 0.f
        );

        // Opportunistic casting - update result with opportunity signals
        static void update_opportunity_signals(
            HybridPredictionResult& result,
            game_object* source,
            game_object* target,
            const pred_sdk::spell_data& spell,
            TargetBehaviorTracker& tracker
        );
    };

    // =========================================================================
    // GLOBAL TRACKER MANAGER
    // =========================================================================

    /**
     * Manages behavior trackers for all enemy targets
     */
    class PredictionManager
    {
    private:
        static inline std::unordered_map<uint32_t, std::unique_ptr<TargetBehaviorTracker>> trackers_;
        static inline float last_update_time_;

        // Self-measurement for physics testing
        static inline math::vector3 self_last_pos_;
        static inline float self_last_speed_ = 0.f;
        static inline float self_last_time_ = 0.f;

    public:
        /**
         * Update all trackers (call every frame)
         */
        static void update();

        /**
         * Measure local player physics (for testing acceleration/deceleration)
         * Enable debug logging to see output
         */
        static void measure_self_physics();

        /**
         * Get or create tracker for target
         */
        static TargetBehaviorTracker* get_tracker(game_object* target);

        /**
         * Get hybrid prediction for target
         */
        static HybridPredictionResult predict(
            game_object* source,
            game_object* target,
            const pred_sdk::spell_data& spell
        );

        /**
         * Clear all tracking data
         */
        static void clear();
    };

#if ENABLE_PATTERN_TELEMETRY
    // =========================================================================
    // PATTERN TELEMETRY (OPTIONAL)
    // =========================================================================

    /**
     * Tracks pattern detection effectiveness for post-game analysis
     * Prints comprehensive recap when game ends
     */
    class PatternTelemetry
    {
    private:
        struct PatternSample
        {
            std::string champion;
            bool had_pattern;
            int pattern_type;  // 0=alternating, 1=repeating
            float confidence;
            float prediction_error;
        };

        struct OpportunitySample
        {
            bool was_peak;
            float wait_time;
            bool was_taken;
        };

        static inline std::vector<PatternSample> samples_;
        static inline std::vector<OpportunitySample> opportunities_;
        static inline float game_start_time_;
        static inline bool enabled_;

    public:
        static void init()
        {
            samples_.clear();
            opportunities_.clear();
            if (g_sdk && g_sdk->clock_facade)
                game_start_time_ = g_sdk->clock_facade->get_game_time();
            else
                game_start_time_ = 0.f;
            enabled_ = true;
        }

        static void set_enabled(bool enable) { enabled_ = enable; }
        static bool is_enabled() { return enabled_; }

        // Stub implementations - telemetry disabled
        static void log_prediction(
            const std::string& champion_name,
            bool has_pattern,
            int pattern_type,
            float pattern_confidence,
            float prediction_error)
        {
            // Telemetry disabled - no-op
            (void)champion_name; (void)has_pattern; (void)pattern_type;
            (void)pattern_confidence; (void)prediction_error;
        }

        static void log_opportunity(
            bool is_peak,
            float elapsed_time,
            bool cast_taken)
        {
            // Telemetry disabled - no-op
            (void)is_peak; (void)elapsed_time; (void)cast_taken;
        }

        static void print_recap()
        {
            // Telemetry disabled - no-op
        }
    };
#endif

} // namespace HybridPred