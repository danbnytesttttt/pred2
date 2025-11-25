#pragma once

#include "sdk.hpp"
#include "StandalonePredictionSDK.h"  // MUST be included AFTER sdk.hpp for compatibility
#include "EdgeCaseDetection.h"
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
    // Time-based decay half-life (seconds) - how quickly old samples lose relevance
    // At half-life, a sample has 50% weight. At 2x half-life, 25% weight, etc.
    constexpr float BEHAVIOR_HALF_LIFE_BASE = 0.5f;  // 500ms base half-life
    constexpr int MIN_SAMPLES_FOR_BEHAVIOR = 10;    // Minimum data for behavior model

    // Tracker cleanup parameters
    constexpr float TRACKER_TIMEOUT = 30.0f;        // Remove trackers after 30s when target doesn't exist

    // Physics parameters
    constexpr float DEFAULT_TURN_RATE = 2.0f * PI;  // radians/second
    constexpr float DEFAULT_ACCELERATION = 1200.0f; // units/s²
    constexpr float DEFAULT_DECELERATION = 2000.0f; // units/s²

    // Human reaction time parameters (CRITICAL for realistic predictions)
    constexpr float HUMAN_REACTION_TIME = 0.25f;    // Average human reaction time (250ms)
    constexpr float MIN_REACTION_TIME = 0.15f;      // Fast reactions (pros, expecting the spell)
    constexpr float MAX_REACTION_TIME = 0.35f;      // Slow reactions (distracted, teamfight)

    // Confidence parameters
    // TRAVEL-TIME BASED DECAY: Confidence decreases with spell travel time, not raw distance
    // Rationale: A fast spell at 1000 units is accurate. A slow spell at 500 units is not.
    // Travel time = distance/speed + cast_delay, which captures actual reaction window
    constexpr float CONFIDENCE_TRAVEL_TIME_DECAY = 0.8f;   // Per-second travel time penalty
    constexpr float CONFIDENCE_LATENCY_FACTOR = 0.01f;     // Per-ms latency penalty
    constexpr float ANIMATION_LOCK_CONFIDENCE_BOOST = 0.3f; // Boost when immobile

    // =========================================================================
    // HELPER FUNCTIONS
    // =========================================================================

    /**
     * Compute adaptive half-life based on target mobility (TIME-BASED DECAY)
     *
     * Fast-moving targets: shorter half-life (recent data more important)
     * Slow-moving targets: longer half-life (longer history matters)
     *
     * Returns half-life in seconds. Weight formula: exp(-(t_now - t_sample) / half_life)
     * This is frame-rate independent and handles fog/lag gracefully.
     */
    inline float get_adaptive_half_life(float move_speed)
    {
        // High mobility (>450): 0.3s half-life (aggressive decay)
        if (move_speed > 450.f)
            return 0.3f;

        // Normal mobility (350-450): 0.5s half-life (balanced)
        if (move_speed > 350.f)
            return 0.5f;

        // Low mobility (<350): 0.8s half-life (longer memory)
        return 0.8f;
    }

    /**
     * Compute time-based exponential decay weight
     *
     * @param time_delta Time since sample was recorded (seconds)
     * @param half_life Half-life in seconds
     * @return Weight in range (0, 1], where 1 = current time
     */
    inline float compute_time_decay_weight(float time_delta, float half_life)
    {
        if (half_life < EPSILON)
            return 1.0f;  // No decay if half-life is zero

        // w = exp(-t * ln(2) / half_life) = 2^(-t / half_life)
        // Using ln(2) ≈ 0.693 for proper half-life semantics
        constexpr float LN2 = 0.693147f;
        return std::exp(-time_delta * LN2 / half_life);
    }

    /**
     * Get effective move speed accounting for CC status (0 if immobilized)
     * CC'd targets cannot dodge, so their reachable region should be minimal
     */
    inline float get_effective_move_speed(game_object* target)
    {
        if (!target || !target->is_valid()) return 0.f;
        if (target->has_buff_of_type(buff_type::stun) ||
            target->has_buff_of_type(buff_type::snare) ||
            target->has_buff_of_type(buff_type::charm) ||
            target->has_buff_of_type(buff_type::fear) ||
            target->has_buff_of_type(buff_type::taunt) ||
            target->has_buff_of_type(buff_type::suppression) ||
            target->has_buff_of_type(buff_type::knockup) ||
            target->has_buff_of_type(buff_type::knockback))
        {
            return 0.f;
        }
        return target->get_move_speed();
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
    inline float fuse_probabilities(float physics_prob, float behavior_prob, float confidence, size_t sample_count, float time_since_update = 0.f)
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
        float physics_weight = 0.5f;  // Default: equal weight

        // If we have very few behavior samples, trust physics more
        if (sample_count < MIN_SAMPLES_FOR_BEHAVIOR)
        {
            // Ramp from 0.7 (no samples) down to 0.5 (minimum samples)
            float factor = static_cast<float>(sample_count) / MIN_SAMPLES_FOR_BEHAVIOR;
            physics_weight = 0.7f - 0.2f * factor;  // 0.7 → 0.5
        }
        else if (sample_count < MIN_SAMPLES_FOR_BEHAVIOR * 2)
        {
            // Ramp from 0.5 (minimum) down to 0.3 (abundant)
            float factor = static_cast<float>(sample_count - MIN_SAMPLES_FOR_BEHAVIOR) / MIN_SAMPLES_FOR_BEHAVIOR;
            physics_weight = 0.5f - 0.2f * factor;  // 0.5 → 0.3
        }
        else
        {
            // Abundant data: trust behavior more (physics weight = 0.3)
            physics_weight = 0.3f;
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

        // Weighted geometric mean
        float fused = std::pow(physics_prob, physics_weight) * std::pow(behavior_prob, 1.0f - physics_weight);
        return fused * confidence;
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

        // Pattern repetition detection (NEW)
        std::vector<int> juke_sequence;          // Last 8 direction changes: -1=left, 0=straight, 1=right
        float pattern_confidence;                // [0,1] Confidence in detected pattern
        math::vector3 predicted_next_direction;  // Unit vector of predicted next move
        bool has_pattern;                        // True if repeating pattern detected
        float last_pattern_update_time;          // Timestamp of last pattern update (for expiration)

        // N-Gram (Markov) pattern recognition
        // Tracks transitions: given previous move, what's the probability of each next move?
        // Key: previous move (-1, 0, 1), Value: map of next move to count
        std::map<int, std::map<int, int>> ngram_transitions;

        /**
         * Get N-Gram probability for a predicted move given current state
         *
         * Uses LAPLACE SMOOTHING (add-one smoothing) to prevent overconfidence
         * with sparse data. Without smoothing, 1 observation = 100% probability,
         * which is statistically invalid for small sample sizes.
         *
         * Formula: P = (count + 1) / (total + num_categories)
         * - With 0 observations: P = 1/3 = 33% (uniform prior)
         * - With 1 observation out of 1: P = 2/4 = 50% (not 100%)
         * - With 10 observations out of 10: P = 11/13 = 85% (approaches true value)
         */
        float get_ngram_probability(int predicted_move) const
        {
            constexpr int NUM_CATEGORIES = 3;  // left, straight, right
            constexpr float UNIFORM_PRIOR = 1.0f / NUM_CATEGORIES;  // 0.33...

            if (juke_sequence.empty()) return UNIFORM_PRIOR;  // No data - uniform

            int last_move = juke_sequence.back();
            auto it = ngram_transitions.find(last_move);
            if (it == ngram_transitions.end()) return UNIFORM_PRIOR;  // No transitions from this state

            const auto& next_counts = it->second;
            int total = 0;
            int target_count = 0;

            for (const auto& pair : next_counts)
            {
                total += pair.second;
                if (pair.first == predicted_move)
                    target_count = pair.second;
            }

            // LAPLACE SMOOTHING: Add 1 to count, add NUM_CATEGORIES to total
            // This prevents 0% and 100% probabilities from sparse data
            return static_cast<float>(target_count + 1) / static_cast<float>(total + NUM_CATEGORIES);
        }

        DodgePattern() : left_dodge_frequency(0.5f), right_dodge_frequency(0.5f),
            forward_frequency(0.5f), backward_frequency(0.5f),
            juke_interval_mean(0.5f), juke_interval_variance(0.1f),
            linear_continuation_prob(0.6f), reaction_delay(200.f),
            pattern_confidence(0.f), predicted_next_direction{}, has_pattern(false),
            last_pattern_update_time(0.f) {
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

        ReachableRegion() : center{}, max_radius(0.f), area(0.f) {}
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
        float behavior_contribution;     // Behavior component [0,1]
        float confidence_score;          // Confidence modifier [0,1]

        // Opportunistic casting signals (NEW)
        bool is_peak_opportunity;        // True if this is a local maximum and declining
        float opportunity_score;         // [0-1] How good is this moment relative to recent history?
        float adaptive_threshold;        // Threshold adjusted for time waited (decays from base)

        // Debugging/analysis data
        ReachableRegion reachable_region;
        BehaviorPDF behavior_pdf;
        std::string reasoning;           // Mathematical explanation

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

        // Opportunistic casting - get or create window for spell slot
        OpportunityWindow& get_opportunity_window(int spell_slot) const;

    private:
        void update_dodge_pattern();
        void detect_direction_changes();
        math::vector3 compute_velocity(const MovementSnapshot& prev, const MovementSnapshot& curr) const;
    };

    // =========================================================================
    // PHYSICS PREDICTION ENGINE
    // =========================================================================

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

    private:
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
         * - wᵢ = exp(-Δtᵢ * ln(2) / half_life)  [TIME-BASED DECAY]
         *   (Δtᵢ = time since snapshot i, half_life adapts to mobility)
         * - K is a Gaussian kernel function
         * - pᵢ(t) is predicted position from snapshot i
         *
         * Time-based decay is frame-rate independent and handles
         * fog-of-war/lag gracefully (no artificial weight changes).
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
            const EdgeCases::EdgeCaseAnalysis& edge_cases
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

    public:
        /**
         * Update all trackers (call every frame)
         */
        static void update();

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
            game_start_time_ = g_sdk->clock_facade->get_game_time();
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