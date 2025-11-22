#include "CustomPredictionSDK.h"
#include "EdgeCaseDetection.h"
#include "PredictionSettings.h"
#include "PredictionTelemetry.h"
#include "PredictionVisuals.h"
#include "FogOfWarTracker.h"
#include "StandalonePredictionSDK.h"  // For math::is_zero
#include <algorithm>
#include <limits>
#include <sstream>
#include <chrono>

// =============================================================================
// TARGETED SPELL PREDICTION
// =============================================================================

pred_sdk::pred_data CustomPredictionSDK::targetted(pred_sdk::spell_data spell_data)
{
    pred_sdk::pred_data result{};

    try
    {
        PRED_DEBUG_LOG("[Danny.Prediction] targetted() called (point-and-click spell)");

        // Targeted spells don't need prediction - just return target position
        if (!spell_data.source || !spell_data.source->is_valid())
        {
            result.hitchance = pred_sdk::hitchance::any;
            return result;
        }

        // Use target selector to find best target
        // CRITICAL: Check target_selector is available before calling
        if (!sdk::target_selector)
        {
            result.hitchance = pred_sdk::hitchance::any;
            return result;
        }

        auto* target = sdk::target_selector->get_hero_target();

        if (!target || !target->is_valid())
        {
            result.hitchance = pred_sdk::hitchance::any;
            return result;
        }

        // For targeted spells, prediction is trivial
        result.cast_position = target->get_position();
        result.predicted_position = target->get_position();
        result.hitchance = pred_sdk::hitchance::very_high;
        result.target = target;
        result.is_valid = true;
    }
    catch (...)
    {
        result.hitchance = pred_sdk::hitchance::any;
        result.is_valid = false;
    }

    return result;
}

// =============================================================================
// SKILLSHOT PREDICTION (AUTO-TARGET)
// =============================================================================

pred_sdk::pred_data CustomPredictionSDK::predict(pred_sdk::spell_data spell_data)
{
    pred_sdk::pred_data result{};

    try
    {
        // Safety: Validate SDK is initialized
        if (!g_sdk || !g_sdk->object_manager)
        {
            result.hitchance = pred_sdk::hitchance::any;
            result.is_valid = false;
            return result;
        }

        if (PredictionSettings::get().enable_debug_logging)
        {
            char debug_msg[512];
            snprintf(debug_msg, sizeof(debug_msg),
                "[Danny.Prediction] Auto-target predict() - source=0x%p range=%.0f type=%d",
                (void*)spell_data.source, spell_data.range, static_cast<int>(spell_data.spell_type));
            g_sdk->log_console(debug_msg);
        }

        // If source is null, use local player as fallback
        if (!spell_data.source || !spell_data.source->is_valid())
        {
            spell_data.source = g_sdk->object_manager->get_local_player();
            if (PredictionSettings::get().enable_debug_logging)
                g_sdk->log_console("[Danny.Prediction] Auto-target: Using local player as source");
        }

        // Auto-select best target
        game_object* best_target = get_best_target(spell_data);

        if (!best_target)
        {
            if (PredictionSettings::get().enable_debug_logging)
                g_sdk->log_console("[Danny.Prediction] Auto-target: No valid target found");
            result.hitchance = pred_sdk::hitchance::any;
            return result;
        }

        // FIXED: Use safe string formatting to prevent buffer overflow
        if (PredictionSettings::get().enable_debug_logging)
        {
            std::string debug_msg = "[Danny.Prediction] Auto-target selected: " + best_target->get_char_name();
            g_sdk->log_console(debug_msg.c_str());
        }

        return predict(best_target, spell_data);
    }
    catch (...)
    {
        result.hitchance = pred_sdk::hitchance::any;
        result.is_valid = false;
        return result;
    }
}

// =============================================================================
// SKILLSHOT PREDICTION (SPECIFIC TARGET)
// =============================================================================

pred_sdk::pred_data CustomPredictionSDK::predict(game_object* obj, pred_sdk::spell_data spell_data)
{
    pred_sdk::pred_data result{};

    // MASTER TRY-CATCH: Prevent ANY crash from this function
    try
    {

        // Safety: Validate SDK is initialized
        if (!g_sdk || !g_sdk->object_manager || !g_sdk->clock_facade)
        {
            result.hitchance = pred_sdk::hitchance::any;
            result.is_valid = false;
            return result;
        }

        // FIXED: Use safe debug logging
        if (PredictionSettings::get().enable_debug_logging)
        {
            char debug_msg[256];
            snprintf(debug_msg, sizeof(debug_msg), "[Danny.Prediction] predict(target) called - obj=0x%p source=0x%p",
                (void*)obj, (void*)spell_data.source);
            g_sdk->log_console(debug_msg);
        }

        // Validation: obj must exist and be valid
        if (!obj || !obj->is_valid())
        {
            PRED_DEBUG_LOG("[Danny.Prediction] EARLY EXIT: Invalid target obj!");
            PredictionTelemetry::TelemetryLogger::log_rejection_invalid_target();
            result.hitchance = pred_sdk::hitchance::any;
            return result;
        }

        // If source is null/invalid, use local player as default (expected behavior for spell scripts)
        // This is normal - spell scripts typically don't set source, expecting prediction SDK to auto-fill
        if (!spell_data.source || !spell_data.source->is_valid())
        {
            spell_data.source = g_sdk->object_manager->get_local_player();

            // Validate local player is valid and has valid position
            if (!spell_data.source || !spell_data.source->is_valid())
            {
                // Should never happen - local player invalid
                g_sdk->log_console("[Danny.Prediction] CRITICAL ERROR: Local player is null or invalid!");
                result.hitchance = pred_sdk::hitchance::any;
                result.is_valid = false;
                return result;
            }

            // Validate position is not zero/invalid (sanity check)
            math::vector3 source_pos = spell_data.source->get_position();
            if (math::is_zero(source_pos))
            {
                // Local player has invalid position (zero vector)
                char err_msg[256];
                snprintf(err_msg, sizeof(err_msg),
                    "[Danny.Prediction] ERROR: Local player position is zero! Source may be invalid.");
                g_sdk->log_console(err_msg);
                result.hitchance = pred_sdk::hitchance::any;
                result.is_valid = false;
                return result;
            }
        }

        // FIXED: Safe debug logging for spell details
        if (PredictionSettings::get().enable_debug_logging)
        {
            char debug_msg[256];
            snprintf(debug_msg, sizeof(debug_msg), "[Danny.Prediction] Spell: Range=%.0f Radius=%.0f Delay=%.2f Speed=%.0f Type=%d",
                spell_data.range, spell_data.radius, spell_data.delay, spell_data.projectile_speed,
                static_cast<int>(spell_data.spell_type));
            g_sdk->log_console(debug_msg);
        }

        // CRITICAL: Check range BEFORE prediction to avoid wasting computation
        // Cache positions to prevent inconsistency from flash/dash
        math::vector3 source_pos = spell_data.source->get_position();
        math::vector3 target_pos = obj->get_position();
        float distance_to_target = target_pos.distance(source_pos);

        // FIX: Use target bounding radius dynamically (Cho'Gath = 100+, Malphite = 80, etc.)
        float target_radius = obj->get_bounding_radius();
        float effective_max_range = spell_data.range + target_radius + 25.f;  // 25 for buffer

        // For vector spells, max hit range = cast_range + range (e.g., Viktor E = 525 + 1100 = 1625)
        if (spell_data.spell_type == pred_sdk::spell_type::vector)
        {
            float first_cast_range = spell_data.cast_range;
            if (first_cast_range < 1.f) first_cast_range = spell_data.range; // Fallback if not set
            effective_max_range = first_cast_range + spell_data.range + target_radius + 25.f;
        }

        if (distance_to_target > effective_max_range)
        {
            if (PredictionSettings::get().enable_debug_logging)
            {
                char range_msg[256];
                snprintf(range_msg, sizeof(range_msg),
                    "[Danny.Prediction] Target out of range: %.0f > %.0f (range + radius %.0f)",
                    distance_to_target, effective_max_range, target_radius);
                g_sdk->log_console(range_msg);
            }
            PredictionTelemetry::TelemetryLogger::log_rejection_current_range();
            result.hitchance = pred_sdk::hitchance::any;
            result.is_valid = false;
            return result;
        }

        // CRITICAL: Check fog of war status
        float current_time = g_sdk->clock_facade->get_game_time();
        FogOfWarTracker::update_visibility(obj, current_time);

        auto [should_predict, fog_confidence_multiplier] = FogOfWarTracker::should_predict_target(obj, current_time);

        if (!should_predict)
        {
            // Target is in fog for too long - don't cast at stale position
            PredictionTelemetry::TelemetryLogger::log_rejection_fog();
            result.hitchance = pred_sdk::hitchance::any;
            result.is_valid = false;
            return result;
        }

        // Start telemetry timing
        auto telemetry_start = std::chrono::high_resolution_clock::now();

        // Use hybrid prediction system - wrapped in try-catch for safety
        HybridPred::HybridPredictionResult hybrid_result;
        try
        {
            hybrid_result = HybridPred::PredictionManager::predict(spell_data.source, obj, spell_data);
        }
        catch (...)
        {
            // Prediction system crashed - return invalid result
            result.hitchance = pred_sdk::hitchance::any;
            result.is_valid = false;
            return result;
        }

        // End telemetry timing
        auto telemetry_end = std::chrono::high_resolution_clock::now();
        float computation_time_ms = std::chrono::duration<float, std::milli>(telemetry_end - telemetry_start).count();

        // FIXED: Safe debug logging for prediction details
        if (PredictionSettings::get().enable_debug_logging)
        {
            try
            {
                std::stringstream ss;
                ss << "[Danny.Prediction] Target: " << obj->get_char_name()
                    << " | Valid: " << (hybrid_result.is_valid ? "YES" : "NO")
                    << " | HitChance: " << (hybrid_result.hit_chance * 100.f) << "% (" << hybrid_result.hit_chance << " raw)";
                g_sdk->log_console(ss.str().c_str());
            }
            catch (...) { /* Ignore logging errors */ }
        }

        if (!hybrid_result.is_valid)
        {
            if (!hybrid_result.reasoning.empty() && PredictionSettings::get().enable_debug_logging)
            {
                std::string debug_msg = "[Danny.Prediction] Reason invalid: " + hybrid_result.reasoning;
                g_sdk->log_console(debug_msg.c_str());
            }

            // Log invalid prediction to telemetry
            if (PredictionSettings::get().enable_telemetry)
            {
                PredictionTelemetry::TelemetryLogger::log_invalid_prediction(hybrid_result.reasoning);
            }

            result.hitchance = pred_sdk::hitchance::any;
            return result;
        }

        // Convert hybrid result to pred_data
        result = convert_to_pred_data(hybrid_result, obj, spell_data);

        // Apply fog of war confidence penalty
        if (fog_confidence_multiplier < 1.0f)
        {
            // Reduce hit chance for targets in fog
            float original_hc = hybrid_result.hit_chance;
            hybrid_result.hit_chance *= fog_confidence_multiplier;

            // Re-convert to enum with reduced hit chance
            result.hitchance = convert_hit_chance_to_enum(hybrid_result.hit_chance);

            if (PredictionSettings::get().enable_debug_logging)
            {
                char fog_msg[256];
                snprintf(fog_msg, sizeof(fog_msg),
                    "[Danny.Prediction] FOG PENALTY: HC %.0f%% -> %.0f%% (multiplier: %.2f)",
                    original_hc * 100.f, hybrid_result.hit_chance * 100.f, fog_confidence_multiplier);
                g_sdk->log_console(fog_msg);
            }
        }

        // DEFENSIVE PROGRAMMING: Enforce hitchance threshold at SDK level
        // This protects against buggy champion scripts that don't check hitchance properly
        bool should_cast = (result.hitchance >= spell_data.expected_hitchance);

        if (!should_cast)
        {
            if (PredictionSettings::get().enable_debug_logging)
            {
                char reject_msg[256];
                snprintf(reject_msg, sizeof(reject_msg),
                    "[REJECT] Hitchance %d below threshold %d - invalidating prediction",
                    result.hitchance, spell_data.expected_hitchance);
                g_sdk->log_console(reject_msg);
            }
            PredictionTelemetry::TelemetryLogger::log_rejection_hitchance();
            result.is_valid = false;
            result.hitchance = pred_sdk::hitchance::any;
            return result;
        }

        // CRITICAL: Validate predicted position is within spell range
        // This prevents casting at targets that will walk out of range
        float predicted_distance;

        // For vector spells (Viktor E, Rumble R), check vector length, not distance from player
        // spell_data.range is the vector length (first_cast to cast_position), not player to end
        // Note: first_cast_position range is already validated in optimize_vector_orientation
        if (spell_data.spell_type == pred_sdk::spell_type::vector)
        {
            predicted_distance = result.cast_position.distance(result.first_cast_position);
        }
        else
        {
            predicted_distance = result.cast_position.distance(source_pos);
        }

        // For linear skillshots, the hitbox extends beyond the center point by the radius
        float range_buffer = (spell_data.spell_type == pred_sdk::spell_type::linear) ? spell_data.radius : 0.f;

        float effective_range = spell_data.range + range_buffer;

        // FIX: Account for movement during cast animation
        // Prediction calculates where they'll be when spell ARRIVES
        // But they also move during our cast animation (~0.25s) BEFORE spell starts traveling
        // This is especially important for max-range casts
        float range_usage = predicted_distance / std::max(spell_data.range, 1.f);

        if (range_usage > 0.85f)  // Near max range - be more careful
        {
            // Check if target is moving away from the predicted position
            auto* tracker = HybridPred::PredictionManager::get_tracker(obj);
            if (tracker)
            {
                math::vector3 target_velocity = tracker->get_current_velocity();

                // For vector spells, use direction along the vector, not from player
                math::vector3 to_predicted;
                if (spell_data.spell_type == pred_sdk::spell_type::vector)
                {
                    to_predicted = result.cast_position - result.first_cast_position;
                }
                else
                {
                    to_predicted = result.cast_position - source_pos;
                }

                float to_pred_mag = to_predicted.magnitude();

                if (to_pred_mag > 0.001f)
                {
                    math::vector3 to_pred_dir = to_predicted / to_pred_mag;
                    // Positive = moving away from us (in same direction as predicted pos)
                    float away_speed = target_velocity.dot(to_pred_dir);

                    if (away_speed > 100.f)  // Moving away at significant speed
                    {
                        // They'll move further during cast animation (~0.25s)
                        float cast_animation_movement = away_speed * 0.25f;
                        effective_range -= cast_animation_movement;
                    }
                }
            }
        }

        // CHECK: Is the PREDICTED position in range?
        if (predicted_distance > effective_range)
        {
            if (PredictionSettings::get().enable_debug_logging)
            {
                char range_msg[256];
                snprintf(range_msg, sizeof(range_msg),
                    "[REJECT] Predicted pos out of range: %.0f > %.0f (Usage: %.0f%%)",
                    predicted_distance, effective_range, range_usage * 100.f);
                g_sdk->log_console(range_msg);
            }
            PredictionTelemetry::TelemetryLogger::log_rejection_predicted_range();
            result.is_valid = false;
            result.hitchance = pred_sdk::hitchance::any;
            return result;
        }

        // Check collision if required
        if (!spell_data.forbidden_collisions.empty())
        {
            try
            {
                pred_sdk::collision_ret collision = collides(result.cast_position, spell_data, obj);
                if (collision.collided)
                {
                    // CRITICAL: For non-piercing skillshots, ANY collision invalidates the prediction
                    PredictionTelemetry::TelemetryLogger::log_rejection_collision();
                    result.is_valid = false;
                    result.hitchance = pred_sdk::hitchance::any;
                    return result;
                }
            }
            catch (...) { /* Ignore collision check errors */ }
        }

        // Log successful prediction to telemetry (wrapped in try-catch for safety)
        if (PredictionSettings::get().enable_telemetry)
        {
            try
            {
                PredictionTelemetry::PredictionEvent event;
                event.timestamp = g_sdk->clock_facade->get_game_time();
                event.target_name = obj->get_char_name();

                // Map spell type enum to string
                switch (spell_data.spell_type)
                {
                case pred_sdk::spell_type::linear: event.spell_type = "linear"; break;
                case pred_sdk::spell_type::circular: event.spell_type = "circular"; break;
                case pred_sdk::spell_type::targetted: event.spell_type = "targeted"; break;
                case pred_sdk::spell_type::vector: event.spell_type = "vector"; break;
                default: event.spell_type = "unknown"; break;
                }

                event.hit_chance = hybrid_result.hit_chance;
                event.confidence = hybrid_result.confidence_score;
                event.distance = spell_data.source->get_position().distance(obj->get_position());
                event.computation_time_ms = computation_time_ms;

                // Spell configuration data (for diagnosing misconfigured spells)
                event.spell_range = spell_data.range;
                event.spell_radius = spell_data.radius;
                event.spell_delay = spell_data.delay;
                event.spell_speed = spell_data.projectile_speed;

                // Movement and prediction offset data
                math::vector3 current_pos = obj->get_position();
                math::vector3 predicted_pos = result.cast_position;
                event.prediction_offset = predicted_pos.distance(current_pos);
                event.target_velocity = obj->get_move_speed();

                // Check if target is moving by examining path
                auto path = obj->get_path();
                event.target_is_moving = (path.size() > 1);

                // Extract edge case info from reasoning
                if (hybrid_result.reasoning.find("STASIS") != std::string::npos)
                    event.edge_case = "stasis";
                else if (hybrid_result.reasoning.find("CHANNEL") != std::string::npos ||
                    hybrid_result.reasoning.find("RECALL") != std::string::npos)
                    event.edge_case = "channeling";
                else if (hybrid_result.reasoning.find("DASH") != std::string::npos)
                {
                    event.edge_case = "dash";
                    event.was_dash = true;
                }
                else
                    event.edge_case = "normal";

                // Check for stationary/animation lock
                event.was_stationary = hybrid_result.reasoning.find("STATIONARY") != std::string::npos;
                event.was_animation_locked = hybrid_result.reasoning.find("animation") != std::string::npos ||
                    hybrid_result.reasoning.find("LOCKED") != std::string::npos;
                event.collision_detected = false;  // Will be updated if collision check fails

                PredictionTelemetry::TelemetryLogger::log_prediction(event);
            }
            catch (...) { /* Ignore telemetry errors */ }
        }

        return result;

    } // End master try
    catch (...)
    {
        // CRITICAL: Catch ANY exception to prevent crash
        result.hitchance = pred_sdk::hitchance::any;
        result.is_valid = false;
        return result;
    }
}

// =============================================================================
// PATH PREDICTION (LINEAR)
// =============================================================================

math::vector3 CustomPredictionSDK::predict_on_path(game_object* obj, float time, bool use_server_pos)
{
    try
    {
        if (!obj || !obj->is_valid())
            return math::vector3{};

        // FIX: Use the smart segment-walking predictor
        // This matches what the actual casting logic uses
        // Prevents "Visual Lie" where drawn line goes through walls but cast doesn't
        return HybridPred::PhysicsPredictor::predict_on_path(obj, time);
    }
    catch (...)
    {
        return math::vector3{};
    }
}

// =============================================================================
// COLLISION DETECTION
// =============================================================================

pred_sdk::collision_ret CustomPredictionSDK::collides(
    const math::vector3& end_point,
    pred_sdk::spell_data spell_data,
    const game_object* target_obj)
{
    pred_sdk::collision_ret result{};
    result.collided = false;

    try
    {
        if (spell_data.forbidden_collisions.empty())
            return result;

        if (!spell_data.source || !spell_data.source->is_valid())
            return result;

        math::vector3 start = spell_data.source->get_position();

        // Simple collision check
        if (check_collision_simple(start, end_point, spell_data, target_obj))
        {
            result.collided = true;
            result.collided_units.clear(); // Simplified - would need actual collision units
        }

        return result;
    }
    catch (...)
    {
        return result;  // Return no collision on error
    }
}

// =============================================================================
// UTILITY FUNCTIONS IMPLEMENTATION
// =============================================================================

float CustomPredictionSDK::CustomPredictionUtils::get_spell_range(
    pred_sdk::spell_data& data,
    game_object* target,
    game_object* source)
{
    if (!source)
        source = data.source;

    if (!source || !source->is_valid())
        return data.range;

    float base_range = data.range;

    // Adjust for targeting type
    if (target && target->is_valid())
    {
        if (data.targetting_type == pred_sdk::targetting_type::center_to_edge)
        {
            // Add target bounding radius
            base_range += target->get_bounding_radius();
        }
        else if (data.targetting_type == pred_sdk::targetting_type::edge_to_edge)
        {
            // Add both source and target bounding radius
            base_range += source->get_bounding_radius();
            base_range += target->get_bounding_radius();
        }
    }

    return base_range;
}

bool CustomPredictionSDK::CustomPredictionUtils::is_in_range(
    pred_sdk::spell_data& data,
    math::vector3 cast_position,
    game_object* target)
{
    if (!data.source || !data.source->is_valid())
        return false;

    math::vector3 source_pos = data.source->get_position();
    float distance = source_pos.distance(cast_position);

    float effective_range = get_spell_range(data, target, data.source);

    // FIX: Allow small buffer for edge-of-range hits
    // For linear spells, the hitbox extends beyond the center by spell radius
    // Example: 1000 range spell with 60 radius can hit at 1060 (edge hit)
    constexpr float EDGE_HIT_BUFFER = 50.f;
    return distance <= effective_range + EDGE_HIT_BUFFER;
}

float CustomPredictionSDK::CustomPredictionUtils::get_spell_hit_time(
    pred_sdk::spell_data& data,
    math::vector3 pos,
    game_object* target)
{
    if (!data.source || !data.source->is_valid())
        return 0.f;

    return HybridPred::PhysicsPredictor::compute_arrival_time(
        data.source->get_position(),
        pos,
        data.projectile_speed,
        data.delay
    );
}

float CustomPredictionSDK::CustomPredictionUtils::get_spell_escape_time(
    pred_sdk::spell_data& data,
    game_object* target)
{
    if (!target || !target->is_valid() || !data.source || !data.source->is_valid())
        return 0.f;

    float current_distance = target->get_position().distance(data.source->get_position());
    float spell_range = get_spell_range(data, target, data.source);

    if (current_distance >= spell_range)
        return 0.f; // Already out of range

    float distance_to_escape = spell_range - current_distance;
    float move_speed = target->get_move_speed();

    // FIXED: Guard against zero/very low move speed (CC'd, dead, etc.)
    if (move_speed < 1.f)
        return std::numeric_limits<float>::max(); // Effectively can't escape

    return distance_to_escape / move_speed;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

pred_sdk::pred_data CustomPredictionSDK::convert_to_pred_data(
    const HybridPred::HybridPredictionResult& hybrid_result,
    game_object* target,
    const pred_sdk::spell_data& spell_data)
{
    pred_sdk::pred_data result{};

    // Copy positions
    result.cast_position = hybrid_result.cast_position;
    result.first_cast_position = hybrid_result.first_cast_position;  // For vector spells (Viktor E, Rumble R, Irelia E)
    result.predicted_position = hybrid_result.cast_position;
    result.target = target;
    result.is_valid = hybrid_result.is_valid;

    // Convert hit chance to enum
    result.hitchance = convert_hit_chance_to_enum(hybrid_result.hit_chance);

    // Calculate hit time
    if (spell_data.source && spell_data.source->is_valid())
    {
        result.intersection_time = HybridPred::PhysicsPredictor::compute_arrival_time(
            spell_data.source->get_position(),
            hybrid_result.cast_position,
            spell_data.projectile_speed,
            spell_data.delay
        );
    }

    return result;
}

pred_sdk::hitchance CustomPredictionSDK::convert_hit_chance_to_enum(float hit_chance)
{
    // Map [0,1] to hitchance enum
    // Adjusted thresholds for more aggressive casting (less "holding")
    if (hit_chance >= 0.95f)
        return pred_sdk::hitchance::guaranteed_hit;
    else if (hit_chance >= 0.80f)  // very_high: kept at 80%
        return pred_sdk::hitchance::very_high;
    else if (hit_chance >= 0.65f)  // high: kept at 65%
        return pred_sdk::hitchance::high;
    else if (hit_chance >= 0.50f)  // medium: 50%
        return pred_sdk::hitchance::medium;
    else if (hit_chance >= 0.30f)  // low: 30%
        return pred_sdk::hitchance::low;
    else
        return pred_sdk::hitchance::any;
}

game_object* CustomPredictionSDK::get_best_target(const pred_sdk::spell_data& spell_data)
{
    // CRITICAL: Validate SDK before any operations
    if (!g_sdk || !g_sdk->object_manager)
        return nullptr;

    if (!spell_data.source || !spell_data.source->is_valid())
    {
        if (PredictionSettings::get().enable_debug_logging && g_sdk)
            g_sdk->log_console("[Danny.Prediction] get_best_target: Invalid source");
        return nullptr;
    }

    // TARGET STICKINESS: Prevent rapid switching between similar-scored targets
    // This reduces jitter and allows behavior tracking to build up
    static uint32_t last_target_id = 0;
    static float last_target_score = 0.f;
    constexpr float STICKINESS_THRESHOLD = 0.15f;  // Need 15% better score to switch

    // Calculate search range: Tactical vs Global spells
    float search_range = spell_data.range;
    if (spell_data.range < 2500.f)
    {
        float buffer = std::min(spell_data.range * 0.5f, 300.f);
        search_range = spell_data.range + buffer;
    }

    // IMPROVED: Iterate ALL enemies and compare scores
    // Don't blindly accept SDK's target - it optimizes for auto-attacks, not skillshots
    game_object* best_target = nullptr;
    float best_score = -1.f;
    game_object* current_target = nullptr;
    float current_target_score = 0.f;

    // Get SDK's preferred target for soft priority boost
    game_object* sdk_preferred = nullptr;
    if (sdk::target_selector)
    {
        sdk_preferred = sdk::target_selector->get_hero_target();
    }

    auto all_heroes = g_sdk->object_manager->get_heroes();

    for (auto* hero : all_heroes)
    {
        // Basic validity checks
        if (!hero || !hero->is_valid() || hero->is_dead())
            continue;
        if (hero->get_team_id() == spell_data.source->get_team_id())
            continue;

        // CRITICAL: Explicit visibility check - don't waste time on fog targets
        if (!hero->is_visible())
            continue;

        // Check range
        float distance = hero->get_position().distance(spell_data.source->get_position());
        if (distance > search_range)
            continue;

        // Calculate score based on hit probability
        float score = calculate_target_score(hero, spell_data);

        // SDK Priority Boost (Soft Priority)
        // If SDK's target selector likes this target, give small bonus
        // but do NOT blindly accept - compare against all targets
        if (sdk_preferred && sdk_preferred->get_network_id() == hero->get_network_id())
        {
            score *= 1.2f;  // 20% preference for SDK selected target
        }

        // Small bonus for targets currently in range (tiebreaker)
        if (distance <= spell_data.range)
        {
            score *= 1.1f;
        }

        // Track if this is our current sticky target
        if (hero->get_network_id() == last_target_id)
        {
            current_target = hero;
            current_target_score = score;
        }

        if (score > best_score)
        {
            best_score = score;
            best_target = hero;
        }
    }

    // STICKINESS: Prefer current target unless new target is significantly better
    // This prevents jitter between similar-scored targets
    if (current_target && current_target_score > 0.f && best_target != current_target)
    {
        // Only switch if new target is STICKINESS_THRESHOLD better
        if (best_score < current_target_score * (1.f + STICKINESS_THRESHOLD))
        {
            // Keep current target - not worth switching
            best_target = current_target;
            best_score = current_target_score;
        }
    }

    // Update sticky target tracking
    if (best_target)
    {
        last_target_id = best_target->get_network_id();
        last_target_score = best_score;
    }
    else
    {
        last_target_id = 0;
        last_target_score = 0.f;
    }

    if (PredictionSettings::get().enable_debug_logging && best_target)
    {
        float distance = best_target->get_position().distance(spell_data.source->get_position());
        char debug[256];
        snprintf(debug, sizeof(debug), "[Danny.Prediction] Selected target: %s at %.0f units (score: %.2f)",
            best_target->get_char_name().c_str(), distance, best_score);
        g_sdk->log_console(debug);
    }

    return best_target;
}

float CustomPredictionSDK::calculate_target_score(
    game_object* target,
    const pred_sdk::spell_data& spell_data)
{
    if (!target || !target->is_valid())
        return 0.f;

    // Analyze edge cases for this target
    EdgeCases::EdgeCaseAnalysis edge_cases = EdgeCases::analyze_target(target, spell_data.source);

    // Filter out invalid targets
    if (edge_cases.is_clone)
        return 0.f;  // Don't target clones

    if (edge_cases.blocked_by_windwall)
        return 0.f;  // Can't hit through windwall

    // Get hybrid prediction for this target
    HybridPred::HybridPredictionResult pred_result =
        HybridPred::PredictionManager::predict(spell_data.source, target, spell_data);

    if (!pred_result.is_valid)
        return 0.f;

    float score = pred_result.hit_chance;

    // Apply edge case priority multipliers (HUGE impact)
    score *= edge_cases.priority_multiplier;

    // Distance factor for scoring
    float distance = target->get_position().distance(spell_data.source->get_position());
    float distance_factor = 0.f;
    if (spell_data.range > 0.f)
    {
        distance_factor = 1.f - std::min(distance / spell_data.range, 1.f);
    }

    // IMPROVED: Reduced distance penalty for better sniping and kiting
    // Old: 0.3 + 0.7*factor = 70% penalty at max range (too harsh)
    // New: 0.7 + 0.3*factor = 30% penalty at max range
    // Example: Target at max range gets 0.7x multiplier (30% penalty)
    //          Target at half range gets 0.85x multiplier (15% penalty)
    //          Target at point blank gets 1.0x multiplier (no penalty)

    // For GLOBAL spells (Lux R, Ezreal R, etc.), remove distance penalty entirely
    if (spell_data.range > 2500.f)
    {
        // Global spell - distance doesn't matter, only hit chance
        // No distance penalty applied
    }
    else
    {
        score *= (0.7f + distance_factor * 0.3f);
    }

    return score;
}

// =============================================================================
// AOE PREDICTION (MULTI-TARGET CLUSTER)
// =============================================================================

CustomPredictionSDK::aoe_pred_result CustomPredictionSDK::predict_aoe_cluster(
    pred_sdk::spell_data spell_data,
    int min_hits,
    float min_single_hc,
    bool priority_weighted)
{
    CustomPredictionSDK::aoe_pred_result result;

    try
    {
        if (!g_sdk || !g_sdk->object_manager)
            return result;

        // Ensure source is valid
        if (!spell_data.source || !spell_data.source->is_valid())
        {
            spell_data.source = g_sdk->object_manager->get_local_player();
            if (!spell_data.source)
                return result;
        }

        math::vector3 source_pos = spell_data.source->get_position();

        // Step 1: Get targets using target selector's sorted list for priority ordering
        struct TargetPrediction
        {
            game_object* target;
            math::vector3 predicted_pos;
            float hit_chance;
            float priority;  // Based on index in sorted list
        };
        std::vector<TargetPrediction> candidates;

        // Use target selector's sorted heroes for proper priority ordering
        std::vector<game_object*> enemy_list;
        if (sdk::target_selector)
        {
            auto sorted = sdk::target_selector->get_sorted_heroes();
            for (auto* hero : sorted)
            {
                if (hero && hero->is_valid() && !hero->is_dead() &&
                    hero->get_team_id() != spell_data.source->get_team_id())
                {
                    enemy_list.push_back(hero);
                }
            }
        }

        // Fallback if target selector unavailable
        if (enemy_list.empty())
        {
            auto heroes = g_sdk->object_manager->get_heroes();
            for (auto* hero : heroes)
            {
                if (hero && hero->is_valid() && !hero->is_dead() &&
                    hero->get_team_id() != spell_data.source->get_team_id())
                {
                    enemy_list.push_back(hero);
                }
            }
        }

        // Process each enemy with index-based priority
        for (size_t i = 0; i < enemy_list.size(); ++i)
        {
            game_object* hero = enemy_list[i];

            if (!hero->is_visible())
                continue;

            float dist = hero->get_position().distance(source_pos);
            if (dist > spell_data.range + 200.f)
                continue;

            // Get individual prediction
            pred_sdk::pred_data pred = predict(hero, spell_data);
            if (!pred.is_valid)
                continue;

            // Convert hitchance enum to float
            float hc = 0.f;
            switch (pred.hitchance)
            {
            case pred_sdk::hitchance::very_high: hc = 0.9f; break;
            case pred_sdk::hitchance::high: hc = 0.7f; break;
            case pred_sdk::hitchance::medium: hc = 0.5f; break;
            case pred_sdk::hitchance::low: hc = 0.3f; break;
            default: hc = 0.1f; break;
            }

            if (hc < min_single_hc)
                continue;

            // Priority based on index: top target = 1.5x, -0.1 per position
            float priority = 1.0f;
            if (priority_weighted)
            {
                priority = 1.0f + std::max(0.f, 0.5f - (i * 0.1f));
            }

            candidates.push_back({ hero, pred.cast_position, hc, priority });
        }

        result.targets_in_range = static_cast<int>(candidates.size());

        if (candidates.size() < static_cast<size_t>(min_hits))
            return result;

        // Step 2: Build test points - target positions + midpoints between targets
        std::vector<math::vector3> test_points;

        // Add each target's predicted position
        for (const auto& c : candidates)
        {
            test_points.push_back(c.predicted_pos);
        }

        // Add midpoints between all pairs of targets
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            for (size_t j = i + 1; j < candidates.size(); ++j)
            {
                math::vector3 midpoint;
                midpoint.x = (candidates[i].predicted_pos.x + candidates[j].predicted_pos.x) * 0.5f;
                midpoint.y = (candidates[i].predicted_pos.y + candidates[j].predicted_pos.y) * 0.5f;
                midpoint.z = (candidates[i].predicted_pos.z + candidates[j].predicted_pos.z) * 0.5f;
                test_points.push_back(midpoint);
            }
        }

        // Step 3: Find best test point
        math::vector3 best_pos;
        float best_score = -1.f;
        std::vector<game_object*> best_hit_targets;
        std::vector<float> best_hit_chances;

        for (const auto& test_pos : test_points)
        {
            // Check if position is in range
            if (test_pos.distance(source_pos) > spell_data.range)
                continue;

            // Calculate score for this position
            float score = 0.f;
            std::vector<game_object*> hits;
            std::vector<float> hcs;

            for (const auto& c : candidates)
            {
                float dist_to_pos = c.predicted_pos.distance(test_pos);
                if (dist_to_pos <= spell_data.radius)
                {
                    // Target would be hit - weight by priority and hit chance
                    float contrib = c.hit_chance * c.priority;
                    score += contrib;
                    hits.push_back(c.target);
                    hcs.push_back(c.hit_chance);
                }
            }

            if (score > best_score && hits.size() >= static_cast<size_t>(min_hits))
            {
                best_score = score;
                best_pos = test_pos;
                best_hit_targets = hits;
                best_hit_chances = hcs;
            }
        }

        // Step 4: Populate result
        if (best_hit_targets.size() >= static_cast<size_t>(min_hits))
        {
            result.cast_position = best_pos;
            result.hit_targets = best_hit_targets;
            result.hit_chances = best_hit_chances;
            result.expected_hits = 0.f;

            float min_hc = 1.f;
            float sum_hc = 0.f;

            for (size_t i = 0; i < best_hit_chances.size(); ++i)
            {
                result.expected_hits += best_hit_chances[i];
                min_hc = std::min(min_hc, best_hit_chances[i]);
                sum_hc += best_hit_chances[i];
            }

            result.min_hit_chance = min_hc;
            result.avg_hit_chance = sum_hc / best_hit_chances.size();
            result.is_valid = true;
        }
    }
    catch (...)
    {
        // AOE solver crashed - return invalid result
        result.is_valid = false;
    }

    return result;
}

// =============================================================================
// LINEAR AOE PREDICTION (MULTI-TARGET LINE)
// =============================================================================

CustomPredictionSDK::aoe_pred_result CustomPredictionSDK::predict_linear_aoe(
    pred_sdk::spell_data spell_data,
    int min_hits,
    float min_single_hc,
    bool priority_weighted)
{
    CustomPredictionSDK::aoe_pred_result result;

    try
    {
        if (!g_sdk || !g_sdk->object_manager)
            return result;

        if (!spell_data.source || !spell_data.source->is_valid())
        {
            spell_data.source = g_sdk->object_manager->get_local_player();
            if (!spell_data.source)
                return result;
        }

        math::vector3 source_pos = spell_data.source->get_position();

        // Step 1: Get targets and their predictions
        struct TargetPrediction
        {
            game_object* target;
            math::vector3 predicted_pos;
            float hit_chance;
            float priority;
        };
        std::vector<TargetPrediction> candidates;

        std::vector<game_object*> enemy_list;
        if (sdk::target_selector)
        {
            auto sorted = sdk::target_selector->get_sorted_heroes();
            for (auto* hero : sorted)
            {
                if (hero && hero->is_valid() && !hero->is_dead() &&
                    hero->get_team_id() != spell_data.source->get_team_id())
                {
                    enemy_list.push_back(hero);
                }
            }
        }

        if (enemy_list.empty())
        {
            auto heroes = g_sdk->object_manager->get_heroes();
            for (auto* hero : heroes)
            {
                if (hero && hero->is_valid() && !hero->is_dead() &&
                    hero->get_team_id() != spell_data.source->get_team_id())
                {
                    enemy_list.push_back(hero);
                }
            }
        }

        for (size_t i = 0; i < enemy_list.size(); ++i)
        {
            game_object* hero = enemy_list[i];

            if (!hero->is_visible())
                continue;

            float dist = hero->get_position().distance(source_pos);
            if (dist > spell_data.range + 200.f)
                continue;

            pred_sdk::pred_data pred = predict(hero, spell_data);
            if (!pred.is_valid)
                continue;

            float hc = 0.f;
            switch (pred.hitchance)
            {
            case pred_sdk::hitchance::very_high: hc = 0.9f; break;
            case pred_sdk::hitchance::high: hc = 0.7f; break;
            case pred_sdk::hitchance::medium: hc = 0.5f; break;
            case pred_sdk::hitchance::low: hc = 0.3f; break;
            default: hc = 0.1f; break;
            }

            if (hc < min_single_hc)
                continue;

            float priority = 1.0f;
            if (priority_weighted)
            {
                priority = 1.0f + std::max(0.f, 0.5f - (i * 0.1f));
            }

            candidates.push_back({ hero, pred.cast_position, hc, priority });
        }

        result.targets_in_range = static_cast<int>(candidates.size());

        if (candidates.size() < static_cast<size_t>(min_hits))
            return result;

        // Step 2: Test each target position as potential cast direction
        // For linear spells, cast_position is the END of the line
        math::vector3 best_pos;
        float best_score = -1.f;
        std::vector<game_object*> best_hit_targets;
        std::vector<float> best_hit_chances;

        for (const auto& primary : candidates)
        {
            // Direction from source to this target
            math::vector3 dir = primary.predicted_pos - source_pos;
            float dist = dir.magnitude();
            if (dist < 0.001f)
                continue;
            dir = dir / dist;

            // Cast position is at max range in this direction
            math::vector3 cast_pos = source_pos + dir * spell_data.range;

            // Check which targets this line would hit
            float score = 0.f;
            std::vector<game_object*> hits;
            std::vector<float> hcs;

            for (const auto& c : candidates)
            {
                // Point-to-line distance check
                math::vector3 to_target = c.predicted_pos - source_pos;
                float projection = to_target.dot(dir);

                // Must be in front (positive projection) and within range
                if (projection < 0.f || projection > spell_data.range)
                    continue;

                // Calculate perpendicular distance to line
                math::vector3 closest_on_line = source_pos + dir * projection;
                float perp_dist = c.predicted_pos.distance(closest_on_line);

                // Hit if within spell radius + target hitbox
                float target_radius = c.target->get_bounding_radius();
                if (perp_dist <= spell_data.radius + target_radius)
                {
                    float contrib = c.hit_chance * c.priority;
                    score += contrib;
                    hits.push_back(c.target);
                    hcs.push_back(c.hit_chance);
                }
            }

            if (score > best_score && hits.size() >= static_cast<size_t>(min_hits))
            {
                best_score = score;
                best_pos = cast_pos;
                best_hit_targets = hits;
                best_hit_chances = hcs;
            }
        }

        // Step 3: Populate result
        if (best_hit_targets.size() >= static_cast<size_t>(min_hits))
        {
            result.cast_position = best_pos;
            result.hit_targets = best_hit_targets;
            result.hit_chances = best_hit_chances;
            result.expected_hits = 0.f;

            float min_hc = 1.f;
            float sum_hc = 0.f;

            for (size_t i = 0; i < best_hit_chances.size(); ++i)
            {
                result.expected_hits += best_hit_chances[i];
                min_hc = std::min(min_hc, best_hit_chances[i]);
                sum_hc += best_hit_chances[i];
            }

            result.min_hit_chance = min_hc;
            result.avg_hit_chance = sum_hc / best_hit_chances.size();
            result.is_valid = true;
        }
    }
    catch (...)
    {
        result.is_valid = false;
    }

    return result;
}

// =============================================================================
// CONE AOE PREDICTION (MULTI-TARGET WEDGE)
// =============================================================================

CustomPredictionSDK::aoe_pred_result CustomPredictionSDK::predict_cone_aoe(
    pred_sdk::spell_data spell_data,
    int min_hits,
    float min_single_hc,
    bool priority_weighted)
{
    CustomPredictionSDK::aoe_pred_result result;

    try
    {
        if (!g_sdk || !g_sdk->object_manager)
            return result;

        // Ensure source is valid
        if (!spell_data.source || !spell_data.source->is_valid())
        {
            spell_data.source = g_sdk->object_manager->get_local_player();
            if (!spell_data.source)
                return result;
        }

        math::vector3 source_pos = spell_data.source->get_position();

        // Get cone parameters
        float cone_range = spell_data.range;
        float cone_half_angle = std::atan2(spell_data.radius, spell_data.range); // Fallback

        // Try to get actual cone angle from spell data
        if (spell_data.spell_slot >= 0)
        {
            spell_entry* spell_entry_ptr = spell_data.source->get_spell(spell_data.spell_slot);
            if (spell_entry_ptr)
            {
                auto spell_info = spell_entry_ptr->get_data();
                if (spell_info)
                {
                    auto static_data = spell_info->get_static_data();
                    if (static_data)
                    {
                        float angle = static_data->get_cast_cone_angle();
                        if (angle > 0.f)
                        {
                            cone_half_angle = (angle * 0.5f) * (3.14159265f / 180.f);
                        }
                        float dist = static_data->get_cast_cone_distance();
                        if (dist > 0.f)
                        {
                            cone_range = dist;
                        }
                    }
                }
            }
        }

        // Step 1: Get targets with predictions
        struct TargetPrediction
        {
            game_object* target;
            math::vector3 predicted_pos;
            float hit_chance;
            float priority;
        };
        std::vector<TargetPrediction> candidates;

        // Get enemy list
        std::vector<game_object*> enemy_list;
        if (sdk::target_selector)
        {
            auto sorted = sdk::target_selector->get_sorted_heroes();
            for (auto* hero : sorted)
            {
                if (hero && hero->is_valid() && !hero->is_dead() &&
                    hero->get_team_id() != spell_data.source->get_team_id())
                {
                    enemy_list.push_back(hero);
                }
            }
        }

        if (enemy_list.empty())
        {
            auto heroes = g_sdk->object_manager->get_heroes();
            for (auto* hero : heroes)
            {
                if (hero && hero->is_valid() && !hero->is_dead() &&
                    hero->get_team_id() != spell_data.source->get_team_id())
                {
                    enemy_list.push_back(hero);
                }
            }
        }

        // Process each enemy
        for (size_t i = 0; i < enemy_list.size(); ++i)
        {
            game_object* hero = enemy_list[i];

            if (!hero->is_visible())
                continue;

            float dist = hero->get_position().distance(source_pos);
            if (dist > cone_range + 200.f)
                continue;

            // Get individual prediction
            pred_sdk::pred_data pred = predict(hero, spell_data);
            if (!pred.is_valid)
                continue;

            // Convert hitchance enum to float
            float hc = 0.f;
            switch (pred.hitchance)
            {
            case pred_sdk::hitchance::very_high: hc = 0.9f; break;
            case pred_sdk::hitchance::high: hc = 0.7f; break;
            case pred_sdk::hitchance::medium: hc = 0.5f; break;
            case pred_sdk::hitchance::low: hc = 0.3f; break;
            default: hc = 0.1f; break;
            }

            if (hc < min_single_hc)
                continue;

            float priority = 1.0f;
            if (priority_weighted)
            {
                priority = 1.0f + std::max(0.f, 0.5f - (i * 0.1f));
            }

            candidates.push_back({ hero, pred.cast_position, hc, priority });
        }

        result.targets_in_range = static_cast<int>(candidates.size());

        if (candidates.size() < static_cast<size_t>(min_hits))
            return result;

        // Step 2: Test multiple directions to find optimal cone orientation
        constexpr int NUM_DIRECTIONS = 36; // Test every 10 degrees
        math::vector3 best_dir;
        float best_score = -1.f;
        std::vector<game_object*> best_hit_targets;
        std::vector<float> best_hit_chances;

        for (int i = 0; i < NUM_DIRECTIONS; ++i)
        {
            float angle = (2.f * 3.14159265f * i) / NUM_DIRECTIONS;
            math::vector3 direction(std::cos(angle), 0.f, std::sin(angle));

            // Check which targets fall within this cone
            float score = 0.f;
            std::vector<game_object*> hits;
            std::vector<float> hcs;

            for (const auto& c : candidates)
            {
                math::vector3 to_target = c.predicted_pos - source_pos;
                float dist_to_target = to_target.magnitude();

                // Check range
                if (dist_to_target > cone_range)
                    continue;

                // Check angle
                if (dist_to_target > 0.001f)
                {
                    math::vector3 target_dir = to_target / dist_to_target;
                    float dot = direction.x * target_dir.x + direction.z * target_dir.z;
                    float angle_to_target = std::acos(std::clamp(dot, -1.f, 1.f));

                    if (angle_to_target <= cone_half_angle)
                    {
                        // Target is within cone
                        float contrib = c.hit_chance * c.priority;
                        score += contrib;
                        hits.push_back(c.target);
                        hcs.push_back(c.hit_chance);
                    }
                }
            }

            if (score > best_score && hits.size() >= static_cast<size_t>(min_hits))
            {
                best_score = score;
                best_dir = direction;
                best_hit_targets = hits;
                best_hit_chances = hcs;
            }
        }

        // Step 3: Populate result
        if (best_hit_targets.size() >= static_cast<size_t>(min_hits))
        {
            // Cast position is at max range in best direction
            result.cast_position = source_pos + best_dir * cone_range;
            result.hit_targets = best_hit_targets;
            result.hit_chances = best_hit_chances;
            result.expected_hits = 0.f;

            float min_hc = 1.f;
            float sum_hc = 0.f;

            for (size_t i = 0; i < best_hit_chances.size(); ++i)
            {
                result.expected_hits += best_hit_chances[i];
                min_hc = std::min(min_hc, best_hit_chances[i]);
                sum_hc += best_hit_chances[i];
            }

            result.min_hit_chance = min_hc;
            result.avg_hit_chance = sum_hc / best_hit_chances.size();
            result.is_valid = true;
        }
    }
    catch (...)
    {
        result.is_valid = false;
    }

    return result;
}

// =============================================================================
// AUTO-ROUTING AOE PREDICTION
// =============================================================================

CustomPredictionSDK::aoe_pred_result CustomPredictionSDK::predict_aoe(
    pred_sdk::spell_data spell_data,
    int min_hits,
    float min_single_hc,
    bool priority_weighted)
{
    // Check for cone spell first (auto-detection)
    if (spell_data.spell_slot >= 0 && spell_data.source)
    {
        spell_entry* spell_entry_ptr = spell_data.source->get_spell(spell_data.spell_slot);
        if (spell_entry_ptr)
        {
            auto spell_info = spell_entry_ptr->get_data();
            if (spell_info)
            {
                auto static_data = spell_info->get_static_data();
                if (static_data && static_data->get_cast_cone_angle() > 0.f)
                {
                    return predict_cone_aoe(spell_data, min_hits, min_single_hc, priority_weighted);
                }
            }
        }
    }

    // Route to appropriate solver based on spell type
    switch (spell_data.spell_type)
    {
    case pred_sdk::spell_type::circular:
        return predict_aoe_cluster(spell_data, min_hits, min_single_hc, priority_weighted);

    case pred_sdk::spell_type::linear:
    case pred_sdk::spell_type::vector:
        return predict_linear_aoe(spell_data, min_hits, min_single_hc, priority_weighted);

    default:
        // Default to circular for unknown types
        return predict_aoe_cluster(spell_data, min_hits, min_single_hc, priority_weighted);
    }
}

bool CustomPredictionSDK::check_collision_simple(
    const math::vector3& start,
    const math::vector3& end,
    const pred_sdk::spell_data& spell_data,
    const game_object* target_obj)
{
    // CRITICAL: Validate SDK before checking collisions
    if (!g_sdk || !g_sdk->object_manager)
        return false;  // No collision if can't check

    // Check each collision type
    for (auto collision_type : spell_data.forbidden_collisions)
    {
        if (collision_type == pred_sdk::collision_type::unit)
        {
            auto minions = g_sdk->object_manager->get_minions();

            for (auto* minion : minions)
            {
                if (!minion || minion == target_obj)
                    continue;

                if (!is_collision_object(minion, spell_data))
                    continue;

                // Only ENEMY minions block skillshots
                if (minion->get_team_id() == spell_data.source->get_team_id())
                    continue;

                // Skip wards
                std::string name = minion->get_char_name();
                if (name.find("Ward") != std::string::npos ||
                    name.find("Trinket") != std::string::npos ||
                    name.find("YellowTrinket") != std::string::npos)
                    continue;

                // Point-to-line distance check
                math::vector3 minion_pos = minion->get_position();
                math::vector3 line_diff = end - start;
                float line_length = line_diff.magnitude();
                if (line_length < 0.001f)
                    continue;  // Start and end are same point
                math::vector3 line_dir = line_diff / line_length;
                math::vector3 to_minion = minion_pos - start;

                float projection = to_minion.dot(line_dir);
                float minion_radius = minion->get_bounding_radius();
                float collision_radius = spell_data.radius + minion_radius;

                // FIX: Account for bounding radius when checking projection bounds
                // Minion can block if its hitbox overlaps the line, even if center is behind/beyond
                if (projection < -minion_radius || projection > line_length + minion_radius)
                    continue;

                // Clamp projection to line segment for distance calculation
                float clamped_proj = std::max(0.f, std::min(projection, line_length));
                math::vector3 closest_point = start + line_dir * clamped_proj;
                float distance = minion_pos.distance(closest_point);

                if (distance <= collision_radius)
                {
                    return true;  // Collision detected
                }
            }
        }
        else if (collision_type == pred_sdk::collision_type::hero)
        {
            auto heroes = g_sdk->object_manager->get_heroes();

            for (auto* hero : heroes)
            {
                if (!hero || hero == target_obj || hero == spell_data.source)
                    continue;

                if (!is_collision_object(hero, spell_data))
                    continue;

                // Only ENEMY heroes block skillshots
                if (hero->get_team_id() == spell_data.source->get_team_id())
                    continue;

                math::vector3 hero_pos = hero->get_position();
                math::vector3 line_diff = end - start;
                float line_length = line_diff.magnitude();
                if (line_length < 0.001f)
                    continue;  // Start and end are same point
                math::vector3 line_dir = line_diff / line_length;
                math::vector3 to_hero = hero_pos - start;

                float projection = to_hero.dot(line_dir);
                float hero_radius = hero->get_bounding_radius();
                float collision_radius = spell_data.radius + hero_radius;

                // FIX: Account for bounding radius when checking projection bounds
                if (projection < -hero_radius || projection > line_length + hero_radius)
                    continue;

                // Clamp projection to line segment for distance calculation
                float clamped_proj = std::max(0.f, std::min(projection, line_length));
                math::vector3 closest_point = start + line_dir * clamped_proj;
                float distance = hero_pos.distance(closest_point);

                if (distance <= collision_radius)
                {
                    return true;  // Collision detected
                }
            }
        }
        // Terrain collision skipped - would need navmesh API
    }

    return false;
}

bool CustomPredictionSDK::is_collision_object(
    game_object* obj,
    const pred_sdk::spell_data& spell_data)
{
    if (!obj || !obj->is_valid() || obj->is_dead())
        return false;

    // Object must be targetable
    if (!obj->is_targetable())
        return false;

    // Check visibility
    if (!obj->is_visible())
        return false;

    return true;
}