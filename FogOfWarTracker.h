#pragma once

#include "sdk.hpp"
#include <unordered_map>
#include <string>

/**
 * =============================================================================
 * FOG OF WAR TRACKER
 * =============================================================================
 *
 * Tracks when enemies enter/exit fog of war to prevent casting at stale
 * last-known positions.
 *
 * - Tracks time since enemy was last visible
 * - Only allows predictions for a brief window after losing vision
 * - Requires higher confidence for fog predictions
 *
 * =============================================================================
 */

namespace FogOfWarTracker
{
    struct VisibilityData
    {
        bool is_currently_visible = true;
        float time_entered_fog = 0.f;      // Game time when target entered fog
        float time_last_visible = 0.f;      // Game time when target was last seen
        math::vector3 last_known_position;  // Position when we last saw them
        math::vector3 last_known_velocity;  // Velocity when we last saw them
    };

    inline std::unordered_map<uint32_t, VisibilityData> g_visibility_data;  // Key by network_id (unique)

    // Settings
    struct FogSettings
    {
        bool enable_fog_predictions = false;    // Allow predictions into fog
        float max_fog_prediction_time = 0.5f;   // Max time to predict after losing vision (seconds)
        float fog_confidence_multiplier = 0.5f; // Multiply hit chance by this for fog predictions

        static FogSettings& get()
        {
            static FogSettings instance;
            return instance;
        }
    };

    /**
     * Update visibility tracking for a target
     */
    inline void update_visibility(game_object* target, float current_time)
    {
        if (!target || !target->is_valid())
            return;

        uint32_t key = target->get_network_id();  // Use network_id (unique per entity)
        auto& data = g_visibility_data[key];

        bool currently_visible = target->is_visible();

        // Track transition from visible -> fog
        if (data.is_currently_visible && !currently_visible)
        {
            data.time_entered_fog = current_time;
            data.last_known_position = target->get_position();

            // CRITICAL: Check if g_sdk and clock_facade are available before using
            if (g_sdk)
            {
                char msg[256];
                snprintf(msg, sizeof(msg), "[FogTracker] Target (ID:%u) entered fog at time %.1fs",
                    key, current_time);
                g_sdk->log_console(msg);
            }
        }
        // Track transition from fog -> visible
        else if (!data.is_currently_visible && currently_visible)
        {
            if (g_sdk)
            {
                char msg[256];
                snprintf(msg, sizeof(msg), "[FogTracker] Target (ID:%u) became visible again at time %.1fs",
                    key, current_time);
                g_sdk->log_console(msg);
            }
        }

        data.is_currently_visible = currently_visible;

        if (currently_visible)
        {
            data.time_last_visible = current_time;
            data.last_known_position = target->get_position();
        }
    }

    /**
     * Check if we should allow prediction for this target
     * Returns: {should_allow, confidence_multiplier}
     */
    inline std::pair<bool, float> should_predict_target(game_object* target, float current_time)
    {
        if (!target || !target->is_valid())
            return { false, 0.f };

        uint32_t key = target->get_network_id();
        auto it = g_visibility_data.find(key);

        // No tracking data yet - assume visible
        if (it == g_visibility_data.end())
            return { true, 1.0f };

        const auto& data = it->second;

        // Target is visible - full prediction allowed
        if (data.is_currently_visible)
            return { true, 1.0f };

        // Target is in fog
        const auto& settings = FogSettings::get();

        // Fog predictions disabled entirely
        if (!settings.enable_fog_predictions)
        {
            if (g_sdk)
            {
                char msg[256];
                snprintf(msg, sizeof(msg), "[FogTracker] Blocking prediction for target (ID:%u) - in fog and fog predictions disabled",
                    key);
                g_sdk->log_console(msg);
            }
            return { false, 0.f };
        }

        // Check how long they've been in fog
        float time_in_fog = current_time - data.time_entered_fog;

        // Too long in fog - don't predict
        if (time_in_fog > settings.max_fog_prediction_time)
        {
            if (g_sdk)
            {
                char msg[256];
                snprintf(msg, sizeof(msg), "[FogTracker] Blocking prediction for target (ID:%u) - in fog for %.2fs (max: %.2fs)",
                    key, time_in_fog, settings.max_fog_prediction_time);
                g_sdk->log_console(msg);
            }
            return { false, 0.f };
        }

        // Allow prediction but with reduced confidence
        if (g_sdk)
        {
            char msg[256];
            snprintf(msg, sizeof(msg), "[FogTracker] Allowing fog prediction for target (ID:%u) - in fog for %.2fs (confidence multiplier: %.2f)",
                key, time_in_fog, settings.fog_confidence_multiplier);
            g_sdk->log_console(msg);
        }

        return { true, settings.fog_confidence_multiplier };
    }

    /**
     * Get time since target was last visible
     */
    inline float get_time_since_visible(game_object* target, float current_time)
    {
        if (!target || !target->is_valid())
            return 999.f;

        uint32_t key = target->get_network_id();
        auto it = g_visibility_data.find(key);

        if (it == g_visibility_data.end())
            return 0.f;

        return current_time - it->second.time_last_visible;
    }

    /**
     * Clear all tracking data
     */
    inline void clear()
    {
        g_visibility_data.clear();
    }

} // namespace FogOfWarTracker