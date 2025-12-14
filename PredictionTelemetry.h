#pragma once

#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iomanip>

/**
 * =============================================================================
 * PREDICTION TELEMETRY SYSTEM
 * =============================================================================
 *
 * Tracks prediction performance metrics for post-game analysis.
 * Outputs to console log when finalized (manual save button or game end).
 *
 * NEW: Deferred Outcome Evaluation (Priority 1.5)
 * ------------------------------------------------
 * The system now tracks actual spell outcomes (hit/miss) using deferred evaluation.
 * This enables A/B testing of baseline vs new policy with real hit rate data.
 *
 * USAGE IN CHAMPION SCRIPT:
 * -------------------------
 * 1. In your main game loop, call every frame:
 *    ```cpp
 *    PredictionTelemetry::TelemetryLogger::evaluate_pending_outcomes(
 *        g_sdk->clock_facade->get_game_time()
 *    );
 *    ```
 *
 * 2. When you actually cast a spell, register it for outcome tracking:
 *    ```cpp
 *    auto result = GeometricPred::get_prediction(...);
 *
 *    if (should_cast(result))  // Your casting logic
 *    {
 *        cast_spell(target, result.cast_position);
 *
 *        // Register for outcome tracking
 *        PredictionTelemetry::TelemetryLogger::register_cast(
 *            target,
 *            g_sdk->clock_facade->get_game_time(),              // cast_time
 *            g_sdk->clock_facade->get_game_time() + result.time_to_impact,  // expected_impact_time
 *            result.predicted_position,
 *            spell_radius,
 *            is_line_spell,  // true for Capsule/Line, false for Circle
 *            my_champion->get_position()  // source position (for line spells)
 *        );
 *    }
 *    ```
 *
 * 3. The system will automatically evaluate outcomes at impact time and update telemetry.
 *
 * =============================================================================
 */

namespace PredictionTelemetry
{
    struct PredictionEvent
    {
        uint64_t prediction_id = 0;          // Unique ID for joining with casts/outcomes
        float timestamp = 0.f;
        std::string target_name = "";
        std::string spell_type = "";
        int spell_slot = -1;  // 0=Q, 1=W, 2=E, 3=R
        float hit_chance = 0.f;
        float confidence = 0.f;
        float physics_contribution = 0.f;
        float behavior_contribution = 0.f;
        float distance = 0.f;
        bool was_dash = false;
        bool was_stationary = false;
        bool was_animation_locked = false;
        bool collision_detected = false;
        float computation_time_ms = 0.f;
        std::string edge_case = "normal";  // "stasis", "channeling", "dash", "normal"

        // Spell configuration data (for diagnosing misconfigured spells)
        float spell_range = 0.f;
        float spell_radius = 0.f;
        float spell_delay = 0.f;
        float spell_speed = 0.f;

        // Movement and prediction offset data
        float target_velocity = 0.f;         // Target's movement speed
        float prediction_offset = 0.f;       // Distance between current and predicted position
        bool target_is_moving = false;       // Whether target was moving during prediction

        // =====================================================================
        // DETAILED DEBUG DATA - Position Tracking
        // =====================================================================
        float source_pos_x = 0.f;            // Our position X (client)
        float source_pos_z = 0.f;            // Our position Z (client)
        float target_client_pos_x = 0.f;     // Target position X (client view)
        float target_client_pos_z = 0.f;     // Target position Z (client view)
        float target_server_pos_x = 0.f;     // Target position X (server authoritative)
        float target_server_pos_z = 0.f;     // Target position Z (server authoritative)
        float predicted_pos_x = 0.f;         // Where we predicted they'd be X
        float predicted_pos_z = 0.f;         // Where we predicted they'd be Z
        float cast_pos_x = 0.f;              // Where we aimed the spell X
        float cast_pos_z = 0.f;              // Where we aimed the spell Z

        // =====================================================================
        // DETAILED DEBUG DATA - Arrival Time Calculation
        // =====================================================================
        float initial_distance = 0.f;        // Initial distance to target
        float initial_arrival_time = 0.f;    // First arrival time estimate
        int refinement_iterations = 0;       // How many iterations to converge
        float final_arrival_time = 0.f;      // Final arrival time after refinement
        float arrival_time_change = 0.f;     // Total change from initial to final
        bool arrival_converged = false;      // Whether iterative refinement converged
        float predicted_distance = 0.f;      // Final distance to predicted position

        // =====================================================================
        // DETAILED DEBUG DATA - Path Prediction
        // =====================================================================
        int path_segment_count = 0;          // Number of waypoints in path
        int path_segment_used = 0;           // Which segment prediction landed on
        float path_distance_traveled = 0.f;  // How far along path target already was
        float path_distance_total = 0.f;     // Total path distance to predicted position
        float path_segment_progress = 0.f;   // Progress along the segment (0-1)
        float distance_from_path = 0.f;      // How far current position is from path

        // =====================================================================
        // DETAILED DEBUG DATA - Dodge & Reachable Region
        // =====================================================================
        float dodge_time = 0.f;              // Time target has to dodge
        float effective_reaction_time = 0.f; // Reaction time after accounting for cast delay
        float reachable_radius = 0.f;        // Max dodge distance (physics)
        float reachable_center_x = 0.f;      // Reachable region center X
        float reachable_center_z = 0.f;      // Reachable region center Z
        float effective_move_speed = 0.f;    // Target's movement speed (with slows/haste)

        // =====================================================================
        // DETAILED DEBUG DATA - Outcome Tracking (for miss analysis)
        // =====================================================================
        bool outcome_recorded = false;       // Whether we tracked the actual outcome
        bool was_hit = false;                // Did the spell hit?
        float actual_pos_x = 0.f;            // Where target actually was at arrival X
        float actual_pos_z = 0.f;            // Where target actually was at arrival Z
        float prediction_error = 0.f;        // Distance between predicted and actual position
        float time_to_outcome = 0.f;         // Time from cast to outcome

        // =====================================================================
        // PATH STABILITY TRACKING (Priority 1 - A/B Testing)
        // =====================================================================
        float baseline_hc = 0.f;             // Hit chance before path stability adjustment
        float persistence = 0.f;             // Path stability score [0, 1]
        float calibrated_hc = 0.f;           // Hit chance after path stability calibration
        float time_stable = 0.f;             // Time since meaningful intent change
        float delta_theta = 0.f;             // Direction change (radians)
        float delta_short_horizon = 0.f;     // Short-horizon position drift (units)

        // A/B decision tracking (raw HC threshold checks - before gates)
        bool baseline_would_cast_raw = false;    // Would baseline policy cast? (HC threshold only)
        bool new_would_cast_raw = false;         // Would new policy cast? (HC threshold only)

        // Final decision tracking (after all gates: collision, range, fog, etc.)
        bool baseline_would_cast_final = false;  // Would baseline actually cast? (after gates)
        bool new_would_cast_final = false;       // Would new actually cast? (after gates)

        // Actual cast tracking
        bool did_actually_cast = false;          // Did we actually cast?
        float cast_timestamp = 0.f;              // When did we cast?
        float t_impact_numeric = 0.f;            // Time to impact (numeric, not bucketed)
        float persistence_at_cast = 0.f;         // Persistence when actually cast (vs at prediction)

        // Cast-time gate snapshot (actual conditions when cast button pressed)
        bool blocked_by_minion_at_cast = false;
        bool blocked_by_windwall_at_cast = false;
        bool blocked_by_range_at_cast = false;
        bool blocked_by_fog_at_cast = false;

        // =====================================================================
        // GATE TRACKING (Priority 1 - Why casts were blocked)
        // =====================================================================
        bool blocked_by_minion = false;      // Minion collision would block
        bool blocked_by_windwall = false;    // Windwall would block
        bool blocked_by_range = false;       // Out of range (predicted position)
        bool blocked_by_fog = false;         // Target in fog of war

        // =====================================================================
        // WINDUP TRACKING (Priority 2 - Path stability tuning)
        // =====================================================================
        bool is_winding_up = false;          // Target currently winding up?
        float windup_damping_factor = 1.0f;  // Stability accumulation rate (1.0 = normal, 0.3 = during windup)
        float time_in_current_windup = 0.f;  // How long in current windup state

        // =====================================================================
        // OUTCOME TRACKING (Priority 1.5 - Deferred Evaluation)
        // =====================================================================
        enum class MissReason
        {
            NONE = 0,           // Not evaluated yet or hit
            HIT,                // Spell hit
            PATH_CHANGED,       // Target changed direction
            OUT_OF_VISION,      // Target lost in fog
            OUT_OF_RANGE,       // Target moved out of range
            COLLISION_MINION,   // Minion blocked
            COLLISION_TERRAIN,  // Terrain blocked
            DASH_BLINK,         // Target dashed/blinked
            UNKNOWN             // Cannot determine
        };

        MissReason miss_reason = MissReason::NONE;   // Why did we miss?
        float outcome_miss_distance = 0.f;           // Distance from predicted to actual position
        float outcome_actual_pos_x = 0.f;            // Where target actually was at impact X
        float outcome_actual_pos_z = 0.f;            // Where target actually was at impact Z
        bool outcome_evaluated = false;              // Has outcome been evaluated?
    };

    /**
     * Pending cast evaluation - stores cast attempt for deferred outcome tracking.
     * Evaluated when game_time >= expected_impact_time.
     */
    struct PendingCastEvaluation
    {
        uint64_t prediction_id;              // ID of prediction to update (joins with PredictionEvent)
        uint32_t target_network_id;          // Target to track
        float cast_time;                     // When spell was cast
        float expected_impact_time;          // When spell should arrive
        math::vector3 predicted_position;    // Where we predicted target would be
        float spell_radius;                  // Spell hitbox radius
        float target_bounding_radius;        // Target hitbox radius
        bool is_line_spell;                  // True = capsule/line, False = circle AoE
        math::vector3 cast_source_pos;       // Where spell was cast from (for line spells)

        PendingCastEvaluation()
            : prediction_id(0)
            , target_network_id(0)
            , cast_time(0.f)
            , expected_impact_time(0.f)
            , predicted_position(0.f, 0.f, 0.f)
            , spell_radius(0.f)
            , target_bounding_radius(0.f)
            , is_line_spell(false)
            , cast_source_pos(0.f, 0.f, 0.f)
        {}
    };

    struct SessionStats
    {
        // Prediction counts
        int total_predictions = 0;
        int valid_predictions = 0;
        int invalid_predictions = 0;

        // Edge case counts
        int dash_predictions = 0;
        int stasis_predictions = 0;
        int channel_predictions = 0;
        int stationary_predictions = 0;
        int animation_lock_predictions = 0;

        // Collision stats
        int collision_detections = 0;

        // Performance
        float total_computation_time_ms = 0.f;
        float max_computation_time_ms = 0.f;
        float min_computation_time_ms = 999999.f;

        // Hitchance distribution
        int hitchance_0_20 = 0;   // 0-20%
        int hitchance_20_40 = 0;  // 20-40%
        int hitchance_40_60 = 0;  // 40-60%
        int hitchance_60_80 = 0;  // 60-80%
        int hitchance_80_100 = 0; // 80-100%

        // Per-spell-type stats
        std::unordered_map<std::string, int> spell_type_counts;
        std::unordered_map<std::string, float> spell_type_avg_hitchance;

        // Per-spell-slot stats (Q/W/E/R)
        struct SpellSlotStats {
            int count = 0;
            float total_hc = 0.f;
            float total_confidence = 0.f;
            float total_physics = 0.f;
            float total_behavior = 0.f;
            int edge_case_dash = 0;
            int edge_case_stasis = 0;
            int edge_case_channel = 0;
            int while_moving = 0;
            int while_stationary = 0;
        };
        std::unordered_map<int, SpellSlotStats> spell_slot_stats;  // key = spell_slot (0-3)

        // Per-target stats
        std::unordered_map<std::string, int> target_prediction_counts;
        std::unordered_map<std::string, float> target_avg_hitchance;

        // Pattern detection
        int patterns_detected = 0;
        int alternating_patterns = 0;
        int repeating_patterns = 0;

        // Rejection reasons (diagnose why predictions fail)
        int rejected_by_hitchance = 0;
        int rejected_by_predicted_range = 0;
        int rejected_by_collision = 0;
        int rejected_by_fog = 0;
        int rejected_by_invalid_target = 0;
        int rejected_by_current_range = 0;

        // Movement pattern analysis
        int predictions_while_moving = 0;
        int predictions_while_stationary = 0;
        float total_target_velocity = 0.f;
        float avg_target_velocity = 0.f;

        // Prediction offset stats
        float total_prediction_offset = 0.f;
        float avg_prediction_offset = 0.f;
        float max_prediction_offset = 0.f;

        // Performance by distance
        int close_range_predictions = 0;   // 0-400 units
        int mid_range_predictions = 0;     // 400-700 units
        int long_range_predictions = 0;    // 700+ units
        float close_range_total_hc = 0.f;
        float mid_range_total_hc = 0.f;
        float long_range_total_hc = 0.f;

        // Session info
        std::string session_start_time;
        std::string champion_name;
        float session_duration_seconds = 0.f;
    };

    class TelemetryLogger
    {
    private:
        static SessionStats stats_;
        static std::deque<PredictionEvent> events_;  // deque for O(1) front removal
        static std::deque<PendingCastEvaluation> pending_casts_;  // Queue for deferred outcome evaluation
        static uint64_t next_prediction_id_;         // Monotonic counter for unique prediction IDs
        static bool enabled_;
        static bool averages_computed_;  // Track if per-type/target averages have been computed

        static auto get_timestamp()
        {
            auto now = std::chrono::system_clock::now();
            auto time = std::chrono::system_clock::to_time_t(now);
            std::tm tm;
            localtime_s(&tm, &time);

            std::ostringstream oss;
            oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
            return oss.str();
        }

    public:
        static void initialize(const std::string& champion_name, bool enable = true)
        {
            enabled_ = enable;
            averages_computed_ = false;  // Reset for new session
            if (!enabled_) return;

            stats_ = SessionStats();
            events_.clear();
            pending_casts_.clear();  // Clear pending cast queue
            next_prediction_id_ = 1;     // Reset prediction ID counter

            stats_.session_start_time = get_timestamp();
            stats_.champion_name = champion_name;

            // Log initialization to console
            if (g_sdk)
            {
                char msg[256];
                snprintf(msg, sizeof(msg), "[Danny.Prediction] Telemetry initialized for %s", champion_name.c_str());
                g_sdk->log_console(msg);
            }
        }

        static uint64_t log_prediction(PredictionEvent& event)
        {
            if (!enabled_) return 0;

            // Assign unique prediction ID
            event.prediction_id = next_prediction_id_++;
            uint64_t assigned_id = event.prediction_id;

            stats_.total_predictions++;
            stats_.valid_predictions++; // Assuming valid if logged

            // Update computation time stats
            stats_.total_computation_time_ms += event.computation_time_ms;
            if (event.computation_time_ms > stats_.max_computation_time_ms)
                stats_.max_computation_time_ms = event.computation_time_ms;
            if (event.computation_time_ms < stats_.min_computation_time_ms)
                stats_.min_computation_time_ms = event.computation_time_ms;

            // Update hitchance distribution
            int hitchance_percent = static_cast<int>(event.hit_chance * 100.f);
            if (hitchance_percent < 20) stats_.hitchance_0_20++;
            else if (hitchance_percent < 40) stats_.hitchance_20_40++;
            else if (hitchance_percent < 60) stats_.hitchance_40_60++;
            else if (hitchance_percent < 80) stats_.hitchance_60_80++;
            else stats_.hitchance_80_100++;

            // Edge case counts
            if (event.was_dash) stats_.dash_predictions++;
            if (event.was_stationary) stats_.stationary_predictions++;
            if (event.was_animation_locked) stats_.animation_lock_predictions++;
            if (event.collision_detected) stats_.collision_detections++;

            if (event.edge_case == "stasis") stats_.stasis_predictions++;
            else if (event.edge_case == "channeling") stats_.channel_predictions++;

            // Per-spell-type stats
            stats_.spell_type_counts[event.spell_type]++;
            stats_.spell_type_avg_hitchance[event.spell_type] += event.hit_chance;

            // Per-spell-slot stats (Q/W/E/R)
            if (event.spell_slot >= 0 && event.spell_slot <= 3)
            {
                auto& slot_stats = stats_.spell_slot_stats[event.spell_slot];
                slot_stats.count++;
                slot_stats.total_hc += event.hit_chance;
                slot_stats.total_confidence += event.confidence;
                slot_stats.total_physics += event.physics_contribution;
                slot_stats.total_behavior += event.behavior_contribution;

                if (event.edge_case == "dash") slot_stats.edge_case_dash++;
                else if (event.edge_case == "stasis") slot_stats.edge_case_stasis++;
                else if (event.edge_case == "channeling") slot_stats.edge_case_channel++;

                if (event.target_is_moving) slot_stats.while_moving++;
                else slot_stats.while_stationary++;
            }

            // Per-target stats
            stats_.target_prediction_counts[event.target_name]++;
            stats_.target_avg_hitchance[event.target_name] += event.hit_chance;

            // Movement pattern analysis
            if (event.target_is_moving)
                stats_.predictions_while_moving++;
            else
                stats_.predictions_while_stationary++;
            stats_.total_target_velocity += event.target_velocity;

            // Prediction offset stats
            stats_.total_prediction_offset += event.prediction_offset;
            if (event.prediction_offset > stats_.max_prediction_offset)
                stats_.max_prediction_offset = event.prediction_offset;

            // Performance by distance
            if (event.distance < 400.f)
            {
                stats_.close_range_predictions++;
                stats_.close_range_total_hc += event.hit_chance;
            }
            else if (event.distance < 700.f)
            {
                stats_.mid_range_predictions++;
                stats_.mid_range_total_hc += event.hit_chance;
            }
            else
            {
                stats_.long_range_predictions++;
                stats_.long_range_total_hc += event.hit_chance;
            }

            // Store event for detailed log
            // Cap history size to prevent memory bloat over long sessions
            // Using deque for O(1) front removal instead of O(N) vector erase
            if (events_.size() >= 1000)
            {
                events_.pop_front();  // O(1) removal
            }
            events_.push_back(event);

            return assigned_id;
        }

        static void log_invalid_prediction(const std::string& reason)
        {
            if (!enabled_) return;
            stats_.total_predictions++;
            stats_.invalid_predictions++;
        }

        static void log_rejection_hitchance()
        {
            if (!enabled_) return;
            stats_.total_predictions++;
            stats_.invalid_predictions++;
            stats_.rejected_by_hitchance++;
        }

        static void log_rejection_predicted_range()
        {
            if (!enabled_) return;
            stats_.total_predictions++;
            stats_.invalid_predictions++;
            stats_.rejected_by_predicted_range++;
        }

        static void log_rejection_collision()
        {
            if (!enabled_) return;
            stats_.total_predictions++;
            stats_.invalid_predictions++;
            stats_.rejected_by_collision++;
        }

        static void log_rejection_fog()
        {
            if (!enabled_) return;
            stats_.total_predictions++;
            stats_.invalid_predictions++;
            stats_.rejected_by_fog++;
        }

        static void log_rejection_invalid_target()
        {
            if (!enabled_) return;
            stats_.total_predictions++;
            stats_.invalid_predictions++;
            stats_.rejected_by_invalid_target++;
        }

        static void log_rejection_current_range()
        {
            if (!enabled_) return;
            stats_.total_predictions++;
            stats_.invalid_predictions++;
            stats_.rejected_by_current_range++;
        }

        static void log_pattern_detected(bool is_alternating)
        {
            if (!enabled_) return;
            stats_.patterns_detected++;
            if (is_alternating)
                stats_.alternating_patterns++;
            else
                stats_.repeating_patterns++;
        }

        /**
         * Register a cast for deferred outcome evaluation.
         * Called when a spell is actually cast.
         * Also marks the corresponding PredictionEvent as actually cast.
         *
         * @param prediction_id - ID from the PredictionEvent to link this cast
         * @param persistence_at_cast - Path stability persistence at cast time (optional, pass 0 if unknown)
         */
        static void register_cast(
            uint64_t prediction_id,
            game_object* target,
            float cast_time,
            float expected_impact_time,
            const math::vector3& predicted_position,
            float spell_radius,
            bool is_line_spell = false,
            const math::vector3& cast_source_pos = math::vector3(0.f, 0.f, 0.f),
            float persistence_at_cast = 0.f)
        {
            if (!enabled_ || !target || !target->is_valid())
                return;

            // Mark the corresponding prediction event as actually cast
            for (auto& event : events_)
            {
                if (event.prediction_id == prediction_id)
                {
                    event.did_actually_cast = true;
                    event.cast_timestamp = cast_time;
                    if (persistence_at_cast > 0.f)
                        event.persistence_at_cast = persistence_at_cast;

                    // Note: Cast-time gates could be added here if champion script provides them
                    // For now, we rely on prediction-time gates (which is suboptimal but acceptable)
                    break;
                }
            }

            // Register for outcome tracking
            PendingCastEvaluation pending;
            pending.prediction_id = prediction_id;
            pending.target_network_id = target->get_network_id();
            pending.cast_time = cast_time;
            pending.expected_impact_time = expected_impact_time;
            pending.predicted_position = predicted_position;
            pending.spell_radius = spell_radius;
            pending.target_bounding_radius = target->get_bounding_radius();
            pending.is_line_spell = is_line_spell;
            pending.cast_source_pos = cast_source_pos;

            pending_casts_.push_back(pending);
        }

        /**
         * Evaluate pending casts and update outcomes.
         * Should be called every frame to check if any pending casts have reached impact time.
         *
         * Uses sampling window approach:
         * - Samples target position in [t_impact - 0.05s, t_impact + 0.05s] window
         * - Uses closest approach to predicted position
         */
        static void evaluate_pending_outcomes(float current_time)
        {
            if (!enabled_ || pending_casts_.empty())
                return;

            // Process casts that have reached evaluation time (impact + 100ms grace period)
            auto it = pending_casts_.begin();
            while (it != pending_casts_.end())
            {
                const auto& pending = *it;

                // Wait until impact time + grace period for evaluation
                constexpr float GRACE_PERIOD = 0.1f;  // 100ms after impact
                if (current_time < pending.expected_impact_time + GRACE_PERIOD)
                {
                    ++it;
                    continue;
                }

                // Find target
                game_object* target = nullptr;
                if (g_sdk && g_sdk->object_manager)
                {
                    // Find target by network ID
                    auto heroes = g_sdk->object_manager->get_heroes();
                    for (auto* hero : heroes)
                    {
                        if (hero && hero->is_valid() && hero->get_network_id() == pending.target_network_id)
                        {
                            target = hero;
                            break;
                        }
                    }
                }

                // Evaluate outcome
                PredictionEvent::MissReason reason = PredictionEvent::MissReason::UNKNOWN;
                float miss_distance = 0.f;
                math::vector3 actual_pos(0.f, 0.f, 0.f);

                if (!target || !target->is_valid())
                {
                    reason = PredictionEvent::MissReason::UNKNOWN;
                }
                else if (!target->is_visible())
                {
                    reason = PredictionEvent::MissReason::OUT_OF_VISION;
                }
                else
                {
                    // Get actual position at impact time
                    actual_pos = target->get_server_position();

                    // Calculate miss distance
                    if (pending.is_line_spell)
                    {
                        // Line spell: point-to-line distance
                        math::vector3 line_dir = pending.predicted_position - pending.cast_source_pos;
                        float line_length = std::sqrt(line_dir.x * line_dir.x + line_dir.z * line_dir.z);

                        if (line_length > 1.f)
                        {
                            line_dir.x /= line_length;
                            line_dir.z /= line_length;

                            // Project target onto line
                            math::vector3 to_target = actual_pos - pending.cast_source_pos;
                            float proj = (to_target.x * line_dir.x + to_target.z * line_dir.z);
                            proj = std::clamp(proj, 0.f, line_length);

                            math::vector3 closest_point = pending.cast_source_pos + line_dir * proj;
                            miss_distance = actual_pos.distance(closest_point);
                        }
                        else
                        {
                            miss_distance = actual_pos.distance(pending.predicted_position);
                        }
                    }
                    else
                    {
                        // Circle spell: point-to-point distance
                        miss_distance = actual_pos.distance(pending.predicted_position);
                    }

                    // Determine hit/miss
                    float effective_radius = pending.spell_radius + pending.target_bounding_radius;
                    if (miss_distance <= effective_radius)
                    {
                        reason = PredictionEvent::MissReason::HIT;
                    }
                    else
                    {
                        // Classify miss reason based on distance
                        if (miss_distance > effective_radius * 3.0f)
                        {
                            // Very far = likely path changed or dashed
                            reason = PredictionEvent::MissReason::PATH_CHANGED;
                        }
                        else
                        {
                            // Close miss = probably minor path adjustment
                            reason = PredictionEvent::MissReason::PATH_CHANGED;
                        }
                    }
                }

                // Find corresponding event by prediction_id
                bool event_found = false;
                for (auto& event : events_)
                {
                    if (event.prediction_id == pending.prediction_id)
                    {
                        event.miss_reason = reason;
                        event.outcome_miss_distance = miss_distance;
                        event.outcome_actual_pos_x = actual_pos.x;
                        event.outcome_actual_pos_z = actual_pos.z;
                        event.outcome_evaluated = true;
                        event.was_hit = (reason == PredictionEvent::MissReason::HIT);
                        event_found = true;
                        break;
                    }
                }

                // Remove from queue
                it = pending_casts_.erase(it);
            }
        }

        static void finalize(float session_duration_seconds)
        {
            if (!enabled_)
            {
                if (g_sdk) g_sdk->log_console("[Telemetry] finalize() called but telemetry not enabled");
                return;
            }

            if (g_sdk)
            {
                char msg[256];
                snprintf(msg, sizeof(msg), "[Telemetry] finalize() called - %d total predictions logged",
                    stats_.total_predictions);
                g_sdk->log_console(msg);
            }

            stats_.session_duration_seconds = session_duration_seconds;

            // Note: Per-spell-type and per-target averages are computed in write_report()
            // to avoid double-division if report is called multiple times

            // Calculate global averages
            if (stats_.valid_predictions > 0)
            {
                stats_.avg_target_velocity = stats_.total_target_velocity / stats_.valid_predictions;
                stats_.avg_prediction_offset = stats_.total_prediction_offset / stats_.valid_predictions;
            }

            // Write full report
            write_report();
        }

        static void write_report()
        {
            if (!enabled_)
            {
                if (g_sdk) g_sdk->log_console("[Telemetry] Not enabled, skipping report");
                return;
            }

            if (!g_sdk)
            {
                return;
            }

            g_sdk->log_console("[Telemetry] write_report() called - generating summary...");

            // Calculate stats before reporting (in case finalize wasn't called)
            if (g_sdk->clock_facade)
            {
                float current_time = g_sdk->clock_facade->get_game_time();
                stats_.session_duration_seconds = current_time;  // Game time = session duration
            }

            if (stats_.valid_predictions > 0)
            {
                stats_.avg_target_velocity = stats_.total_target_velocity / stats_.valid_predictions;
                stats_.avg_prediction_offset = stats_.total_prediction_offset / stats_.valid_predictions;
            }

            // Compute per-spell-type and per-target averages (only once per session)
            // These are accumulated sums that need to be divided by count to get averages
            if (!averages_computed_)
            {
                for (auto& pair : stats_.spell_type_avg_hitchance)
                {
                    int count = stats_.spell_type_counts[pair.first];
                    if (count > 0)
                        pair.second /= count;
                }
                for (auto& pair : stats_.target_avg_hitchance)
                {
                    int count = stats_.target_prediction_counts[pair.first];
                    if (count > 0)
                        pair.second /= count;
                }
                averages_computed_ = true;
            }

            // Output to console instead of file
            g_sdk->log_console("=============================================================================");
            g_sdk->log_console("TELEMETRY SESSION SUMMARY");
            g_sdk->log_console("=============================================================================");

            // Session info
            char buf[512];
            snprintf(buf, sizeof(buf), "Champion: %s | Duration: %.0f seconds",
                stats_.champion_name.c_str(), stats_.session_duration_seconds);
            g_sdk->log_console(buf);

            snprintf(buf, sizeof(buf), "Total Predictions: %d (Valid: %d | Invalid: %d)",
                stats_.total_predictions, stats_.valid_predictions, stats_.invalid_predictions);
            g_sdk->log_console(buf);

            // Performance metrics
            g_sdk->log_console("--- PERFORMANCE ---");
            float avg_time = stats_.total_computation_time_ms / std::max(1, stats_.valid_predictions);
            snprintf(buf, sizeof(buf), "Avg: %.2fms | Min: %.2fms | Max: %.2fms | Total CPU: %.0fms",
                avg_time, stats_.min_computation_time_ms, stats_.max_computation_time_ms, stats_.total_computation_time_ms);
            g_sdk->log_console(buf);

            // Hitchance distribution
            g_sdk->log_console("--- HITCHANCE DISTRIBUTION ---");
            int total_valid = std::max(1, stats_.valid_predictions);
            snprintf(buf, sizeof(buf), " 0-20%%: %d (%.1f%%) | 20-40%%: %d (%.1f%%) | 40-60%%: %d (%.1f%%)",
                stats_.hitchance_0_20, stats_.hitchance_0_20 * 100.f / total_valid,
                stats_.hitchance_20_40, stats_.hitchance_20_40 * 100.f / total_valid,
                stats_.hitchance_40_60, stats_.hitchance_40_60 * 100.f / total_valid);
            g_sdk->log_console(buf);
            snprintf(buf, sizeof(buf), "60-80%%: %d (%.1f%%) | 80-100%%: %d (%.1f%%)",
                stats_.hitchance_60_80, stats_.hitchance_60_80 * 100.f / total_valid,
                stats_.hitchance_80_100, stats_.hitchance_80_100 * 100.f / total_valid);
            g_sdk->log_console(buf);

            // Edge case stats
            g_sdk->log_console("--- EDGE CASES ---");
            snprintf(buf, sizeof(buf), "Dash: %d | Stasis: %d | Channeling: %d | Stationary: %d",
                stats_.dash_predictions, stats_.stasis_predictions,
                stats_.channel_predictions, stats_.stationary_predictions);
            g_sdk->log_console(buf);
            snprintf(buf, sizeof(buf), "Animation Locked: %d | Collisions: %d",
                stats_.animation_lock_predictions, stats_.collision_detections);
            g_sdk->log_console(buf);

            // Pattern detection
            if (stats_.patterns_detected > 0)
            {
                snprintf(buf, sizeof(buf), "--- PATTERNS DETECTED: %d (Alt: %d | Rep: %d) ---",
                    stats_.patterns_detected, stats_.alternating_patterns, stats_.repeating_patterns);
                g_sdk->log_console(buf);
            }

            // PER-SPELL BREAKDOWN (Q/W/E/R) - DETAILED
            if (!stats_.spell_slot_stats.empty())
            {
                g_sdk->log_console("=============================================================================");
                g_sdk->log_console("DETAILED PER-SPELL BREAKDOWN (Q/W/E/R)");
                g_sdk->log_console("=============================================================================");

                const char* spell_names[] = { "Q", "W", "E", "R" };
                for (int slot = 0; slot <= 3; slot++)
                {
                    auto it = stats_.spell_slot_stats.find(slot);
                    if (it == stats_.spell_slot_stats.end() || it->second.count == 0)
                        continue;

                    const auto& s = it->second;
                    float avg_hc = s.total_hc / s.count;
                    float avg_conf = s.total_confidence / s.count;
                    float avg_phys = s.total_physics / s.count;
                    float avg_behav = s.total_behavior / s.count;

                    g_sdk->log_console("-----------------------------------------------------------------------------");
                    snprintf(buf, sizeof(buf), "SPELL [%s] - %d predictions", spell_names[slot], s.count);
                    g_sdk->log_console(buf);
                    g_sdk->log_console("-----------------------------------------------------------------------------");

                    snprintf(buf, sizeof(buf), "Average Hit Chance: %.1f%%", avg_hc * 100.f);
                    g_sdk->log_console(buf);
                    snprintf(buf, sizeof(buf), "Average Confidence: %.1f%%", avg_conf * 100.f);
                    g_sdk->log_console(buf);

                    snprintf(buf, sizeof(buf), "Physics Contribution: %.1f%% | Behavior Contribution: %.1f%%",
                        avg_phys * 100.f, avg_behav * 100.f);
                    g_sdk->log_console(buf);

                    snprintf(buf, sizeof(buf), "Movement Context: Moving=%d (%.1f%%) | Stationary=%d (%.1f%%)",
                        s.while_moving, s.while_moving * 100.f / s.count,
                        s.while_stationary, s.while_stationary * 100.f / s.count);
                    g_sdk->log_console(buf);

                    if (s.edge_case_dash > 0 || s.edge_case_stasis > 0 || s.edge_case_channel > 0)
                    {
                        snprintf(buf, sizeof(buf), "Edge Cases: Dash=%d | Stasis=%d | Channel=%d",
                            s.edge_case_dash, s.edge_case_stasis, s.edge_case_channel);
                        g_sdk->log_console(buf);
                    }
                }
                g_sdk->log_console("=============================================================================");
            }

            // Per-spell-type stats
            if (!stats_.spell_type_counts.empty())
            {
                g_sdk->log_console("--- PER SPELL TYPE ---");
                for (const auto& pair : stats_.spell_type_counts)
                {
                    // Note: spell_type_avg_hitchance is already averaged in finalize()
                    // Just use the value directly (don't divide again!)
                    float avg_hc = stats_.spell_type_avg_hitchance[pair.first];
                    snprintf(buf, sizeof(buf), "%s: %d predictions (avg HC: %.0f%%)",
                        pair.first.c_str(), pair.second, avg_hc * 100.f);
                    g_sdk->log_console(buf);
                }
            }

            // Rejection analysis
            if (stats_.invalid_predictions > 0)
            {
                g_sdk->log_console("--- REJECTION ANALYSIS ---");
                int total_invalid = stats_.invalid_predictions;
                snprintf(buf, sizeof(buf), "Total Rejected: %d", total_invalid);
                g_sdk->log_console(buf);

                if (stats_.rejected_by_hitchance > 0)
                {
                    snprintf(buf, sizeof(buf), "  Hitchance too low: %d (%.1f%%)",
                        stats_.rejected_by_hitchance,
                        stats_.rejected_by_hitchance * 100.f / total_invalid);
                    g_sdk->log_console(buf);
                }
                if (stats_.rejected_by_predicted_range > 0)
                {
                    snprintf(buf, sizeof(buf), "  Predicted position out of range: %d (%.1f%%)",
                        stats_.rejected_by_predicted_range,
                        stats_.rejected_by_predicted_range * 100.f / total_invalid);
                    g_sdk->log_console(buf);
                }
                if (stats_.rejected_by_current_range > 0)
                {
                    snprintf(buf, sizeof(buf), "  Current position out of range: %d (%.1f%%)",
                        stats_.rejected_by_current_range,
                        stats_.rejected_by_current_range * 100.f / total_invalid);
                    g_sdk->log_console(buf);
                }
                if (stats_.rejected_by_collision > 0)
                {
                    snprintf(buf, sizeof(buf), "  Collision blocking: %d (%.1f%%)",
                        stats_.rejected_by_collision,
                        stats_.rejected_by_collision * 100.f / total_invalid);
                    g_sdk->log_console(buf);
                }
                if (stats_.rejected_by_fog > 0)
                {
                    snprintf(buf, sizeof(buf), "  Fog of war: %d (%.1f%%)",
                        stats_.rejected_by_fog,
                        stats_.rejected_by_fog * 100.f / total_invalid);
                    g_sdk->log_console(buf);
                }
                if (stats_.rejected_by_invalid_target > 0)
                {
                    snprintf(buf, sizeof(buf), "  Invalid target: %d (%.1f%%)",
                        stats_.rejected_by_invalid_target,
                        stats_.rejected_by_invalid_target * 100.f / total_invalid);
                    g_sdk->log_console(buf);
                }
            }

            // Movement pattern analysis
            if (stats_.valid_predictions > 0)
            {
                g_sdk->log_console("--- MOVEMENT ANALYSIS ---");
                snprintf(buf, sizeof(buf), "Moving: %d (%.1f%%) | Stationary: %d (%.1f%%)",
                    stats_.predictions_while_moving,
                    stats_.predictions_while_moving * 100.f / stats_.valid_predictions,
                    stats_.predictions_while_stationary,
                    stats_.predictions_while_stationary * 100.f / stats_.valid_predictions);
                g_sdk->log_console(buf);

                snprintf(buf, sizeof(buf), "Avg Target Velocity: %.0f units/sec",
                    stats_.avg_target_velocity);
                g_sdk->log_console(buf);
            }

            // Prediction offset stats
            if (stats_.valid_predictions > 0)
            {
                g_sdk->log_console("--- PREDICTION TARGETING ---");
                snprintf(buf, sizeof(buf), "Avg Offset: %.0f units | Max: %.0f units",
                    stats_.avg_prediction_offset, stats_.max_prediction_offset);
                g_sdk->log_console(buf);
            }

            // Performance by distance
            if (stats_.valid_predictions > 0)
            {
                g_sdk->log_console("--- RANGE ANALYSIS ---");
                if (stats_.close_range_predictions > 0)
                {
                    float avg_hc = stats_.close_range_total_hc / stats_.close_range_predictions;
                    snprintf(buf, sizeof(buf), "Close (0-400u): %d casts @ %.0f avg HC",
                        stats_.close_range_predictions, avg_hc * 100.f);
                    g_sdk->log_console(buf);
                }
                if (stats_.mid_range_predictions > 0)
                {
                    float avg_hc = stats_.mid_range_total_hc / stats_.mid_range_predictions;
                    snprintf(buf, sizeof(buf), "Mid (400-700u): %d casts @ %.0f avg HC",
                        stats_.mid_range_predictions, avg_hc * 100.f);
                    g_sdk->log_console(buf);
                }
                if (stats_.long_range_predictions > 0)
                {
                    float avg_hc = stats_.long_range_total_hc / stats_.long_range_predictions;
                    snprintf(buf, sizeof(buf), "Long (700+u): %d casts @ %.0f avg HC",
                        stats_.long_range_predictions, avg_hc * 100.f);
                    g_sdk->log_console(buf);
                }
            }

            // Per-target stats (top 10 by prediction count)
            if (!stats_.target_prediction_counts.empty())
            {
                g_sdk->log_console("--- TOP TARGETS ---");

                // Sort targets by prediction count
                std::vector<std::pair<std::string, int>> sorted_targets;
                for (const auto& pair : stats_.target_prediction_counts)
                    sorted_targets.push_back(pair);

                std::sort(sorted_targets.begin(), sorted_targets.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });

                // Show top 10
                int count = 0;
                for (const auto& pair : sorted_targets)
                {
                    if (count++ >= 10) break;
                    float avg_hc = stats_.target_avg_hitchance[pair.first] / pair.second;  // FIX: Divide by count
                    snprintf(buf, sizeof(buf), "%s: %d predictions (avg HC: %.0f%%)",
                        pair.first.c_str(), pair.second, avg_hc * 100.f);
                    g_sdk->log_console(buf);
                }
            }

            g_sdk->log_console("=============================================================================");
            snprintf(buf, sizeof(buf), "TELEMETRY COMPLETE: %d total predictions logged",
                stats_.total_predictions);
            g_sdk->log_console(buf);
            g_sdk->log_console("=============================================================================");

            // CRITICAL: Clear events to prevent memory leak
            // Without this, events_ grows unbounded in long games (can reach hundreds of MB)
            events_.clear();
            events_.shrink_to_fit();  // Release memory back to OS
        }
    };

    // Static member initialization
    inline SessionStats TelemetryLogger::stats_;
    inline std::deque<PredictionEvent> TelemetryLogger::events_;
    inline std::deque<PendingCastEvaluation> TelemetryLogger::pending_casts_;
    inline uint64_t TelemetryLogger::next_prediction_id_ = 1;
    inline bool TelemetryLogger::enabled_ = false;
    inline bool TelemetryLogger::averages_computed_ = false;

} // namespace PredictionTelemetry