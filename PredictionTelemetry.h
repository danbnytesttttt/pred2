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
 * =============================================================================
 */

namespace PredictionTelemetry
{
    struct PredictionEvent
    {
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
        static bool enabled_;

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
            if (!enabled_) return;

            stats_ = SessionStats();
            events_.clear();

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

        static void log_prediction(const PredictionEvent& event)
        {
            if (!enabled_) return;

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

            // Calculate averages for per-spell-type
            for (auto& pair : stats_.spell_type_avg_hitchance)
            {
                int count = stats_.spell_type_counts[pair.first];
                if (count > 0)
                    pair.second /= count;
            }

            // Calculate averages for per-target
            for (auto& pair : stats_.target_avg_hitchance)
            {
                int count = stats_.target_prediction_counts[pair.first];
                if (count > 0)
                    pair.second /= count;
            }

            // Calculate new averages
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
                    snprintf(buf, sizeof(buf), "Close (0-400u): %d casts @ %.0f%% avg HC",
                        stats_.close_range_predictions, avg_hc * 100.f);
                    g_sdk->log_console(buf);
                }
                if (stats_.mid_range_predictions > 0)
                {
                    float avg_hc = stats_.mid_range_total_hc / stats_.mid_range_predictions;
                    snprintf(buf, sizeof(buf), "Mid (400-700u): %d casts @ %.0f%% avg HC",
                        stats_.mid_range_predictions, avg_hc * 100.f);
                    g_sdk->log_console(buf);
                }
                if (stats_.long_range_predictions > 0)
                {
                    float avg_hc = stats_.long_range_total_hc / stats_.long_range_predictions;
                    snprintf(buf, sizeof(buf), "Long (700+u): %d casts @ %.0f%% avg HC",
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
                    float avg_hc = stats_.target_avg_hitchance[pair.first];
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
    inline bool TelemetryLogger::enabled_ = false;

} // namespace PredictionTelemetry