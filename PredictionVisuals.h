#pragma once

#include "sdk.hpp"

/**
 * =============================================================================
 * PREDICTION VISUALIZATION SYSTEM
 * =============================================================================
 *
 * Draws real-time prediction indicators:
 * - Line from player to predicted position
 * - Circle at predicted cast position with spell radius
 *
 * =============================================================================
 */

namespace PredictionVisuals
{
    // Settings
    struct VisualsSettings
    {
        bool enabled = true;
        bool draw_current_position = true;       // Draw circle at enemy's current position
        bool draw_predicted_position = true;     // Draw circle at predicted position
        bool draw_movement_line = false;         // Draw line current → predicted (disabled by default)
        float prediction_time = 0.75f;           // How far ahead to predict (seconds)

        uint32_t main_color = 0xFFE19D9D;  // Main color for all visuals (salmon/pink default)

        float current_circle_radius = 65.0f;
        float predicted_circle_radius = 80.0f;
        float line_thickness = 2.0f;
        float circle_thickness = 2.5f;

        static VisualsSettings& get()
        {
            static VisualsSettings instance;
            return instance;
        }

        // Helper: Create a lighter version of a color for current position indicator
        static uint32_t make_lighter(uint32_t color)
        {
            uint8_t a = (color >> 24) & 0xFF;
            uint8_t r = (color >> 16) & 0xFF;
            uint8_t g = (color >> 8) & 0xFF;
            uint8_t b = color & 0xFF;

            // Blend with white (255,255,255) at 70% to make it much lighter
            r = static_cast<uint8_t>(r + (255 - r) * 0.7f);
            g = static_cast<uint8_t>(g + (255 - g) * 0.7f);
            b = static_cast<uint8_t>(b + (255 - b) * 0.7f);

            return (a << 24) | (r << 16) | (g << 8) | b;
        }
    };

    /**
     * Update and draw continuous prediction for current target
     */
    inline void draw_continuous_prediction(float current_time)
    {
        // Safety checks
        if (!VisualsSettings::get().enabled)
            return;

        if (!g_sdk || !g_sdk->renderer)
            return;

        if (!sdk::target_selector)
            return;

        // Get current target from target selector - only draw for YOUR CURRENT TARGET
        auto* target = sdk::target_selector->get_hero_target();
        if (!target)
        {
            // No target selected - don't draw anything
            return;
        }

        // CRITICAL: Additional validity checks to prevent crashes
        try
        {
            if (!target->is_valid())
                return;

            if (target->is_dead())
                return;

            if (!target->is_visible())
                return;

            // Additional safety for dummies/special units
            if (!target->is_targetable())
                return;

            // FIX: Only draw for enemies within 1500 range
            if (g_sdk && g_sdk->object_manager)
            {
                auto* local_player = g_sdk->object_manager->get_local_player();
                if (local_player && local_player->is_valid())
                {
                    float distance = target->get_position().distance(local_player->get_position());
                    if (distance > 1500.f)
                        return;
                }
            }
        }
        catch (...)
        {
            // Target became invalid - skip drawing
            return;
        }

        const auto& settings = VisualsSettings::get();

        // Get current position and calculate proper velocity
        math::vector3 current_pos;
        math::vector3 velocity;

        try
        {
            current_pos = target->get_position();

            // Calculate velocity from path (same as prediction SDK does)
            auto path = target->get_path();
            if (path.size() > 1)
            {
                // Get direction to next waypoint
                math::vector3 next_waypoint = path[1];
                math::vector3 diff = next_waypoint - current_pos;

                // CRASH PROTECTION: Check for zero vector before normalizing
                float diff_magnitude = diff.magnitude();
                if (diff_magnitude < 0.001f)
                {
                    velocity = math::vector3(0, 0, 0);
                }
                else
                {
                    math::vector3 direction = diff / diff_magnitude;
                    float move_speed = target->get_move_speed();
                    velocity = direction * move_speed;
                }
            }
            else
            {
                // Not moving or no path - use zero velocity
                velocity = math::vector3(0, 0, 0);
            }
        }
        catch (...)
        {
            // Target became invalid mid-frame
            return;
        }

        // Only draw if target is actually moving
        float velocity_magnitude = velocity.magnitude();
        if (velocity_magnitude < 1.0f)
            return;  // Target is stationary - prediction not useful

        // Simple linear prediction: position + velocity * time
        math::vector3 predicted_pos = current_pos + velocity * settings.prediction_time;

        // Draw current position (very light version of main color)
        if (settings.draw_current_position)
        {
            try
            {
                uint32_t light_color = VisualsSettings::make_lighter(settings.main_color);
                g_sdk->renderer->add_circle_3d(
                    current_pos,
                    settings.current_circle_radius,
                    settings.circle_thickness,
                    light_color
                );
            }
            catch (...) { /* Ignore render errors */ }
        }

        // Draw predicted position (main color circle)
        if (settings.draw_predicted_position)
        {
            try
            {
                g_sdk->renderer->add_circle_3d(
                    predicted_pos,
                    settings.predicted_circle_radius,
                    settings.circle_thickness,
                    settings.main_color
                );
            }
            catch (...) { /* Ignore render errors */ }
        }

        // Draw skillshot line connected to circle edge (main color)
        if (settings.draw_movement_line)
        {
            try
            {
                // CRITICAL: Validate SDK and object_manager before accessing
                if (!g_sdk || !g_sdk->object_manager)
                    return;

                auto* local_player = g_sdk->object_manager->get_local_player();
                if (local_player && local_player->is_valid())
                {
                    math::vector3 player_pos = local_player->get_position();

                    // Calculate line endpoint at circle edge (not center)
                    // This makes line and circle appear as one connected shape
                    math::vector3 diff = predicted_pos - player_pos;
                    float diff_magnitude = diff.magnitude();

                    // CRASH PROTECTION: Skip if player and prediction are at same spot
                    if (diff_magnitude < 0.001f)
                        return;

                    math::vector3 direction = diff / diff_magnitude;
                    math::vector3 circle_edge = predicted_pos - direction * settings.predicted_circle_radius;

                    math::vector2 screen_player = g_sdk->renderer->world_to_screen(player_pos);
                    math::vector2 screen_edge = g_sdk->renderer->world_to_screen(circle_edge);

                    bool player_valid = (screen_player.x != 0.f || screen_player.y != 0.f);
                    bool edge_valid = (screen_edge.x != 0.f || screen_edge.y != 0.f);

                    if (player_valid && edge_valid)
                    {
                        g_sdk->renderer->add_line_2d(
                            screen_player,
                            screen_edge,
                            settings.line_thickness,
                            settings.main_color
                        );
                    }
                }
            }
            catch (...) { /* Ignore render errors */ }
        }
    }

    /**
     * Clear all resources (call when game ends or plugin unloads)
     */
    inline void clear()
    {
        // Nothing to clear for continuous prediction system
    }

} // namespace PredictionVisuals