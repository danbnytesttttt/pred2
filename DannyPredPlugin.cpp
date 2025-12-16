#include "sdk.hpp"
#include "CustomPredictionSDK.h"
#include "PredictionSettings.h"
#include "PredictionVisuals.h"
#include "PredictionTelemetry.h"
#include "FogOfWarTracker.h"

CustomPredictionSDK customPrediction;

// Global variable for champion name
std::string MyHeroNamePredCore;

// Menu category pointer
menu_category* g_menu = nullptr;

// Global map to store hit chance per target (for display)
std::unordered_map<uint32_t, float> g_hit_chance_map;

// Helper function to update hit chance for display (called from prediction code)
void update_hit_chance_display(uint32_t target_id, float hit_chance)
{
    g_hit_chance_map[target_id] = hit_chance;
}

// Update callback function
void __fastcall on_update()
{
    try
    {
        CustomPredictionSDK::update_trackers();
    }
    catch (...) { /* Prevent any crash from update callback */ }
}

// Render callback function for visual indicators
void __fastcall on_draw()
{
    try
    {
        if (!g_sdk || !g_sdk->clock_facade || !g_sdk->renderer)
            return;

        float current_time = g_sdk->clock_facade->get_game_time();

        // Only draw if visuals are enabled
        if (PredictionSettings::get().enable_visuals)
        {
            PredictionVisuals::draw_continuous_prediction(current_time);
        }

        // Draw debug hitchance text on left side of screen
        if (PredictionVisuals::VisualsSettings::get().draw_debug_text && g_sdk->object_manager)
        {
            auto* local_player = g_sdk->object_manager->get_local_player();
            if (local_player)
            {
                int local_team = local_player->get_team_id();
                auto heroes = g_sdk->object_manager->get_heroes();

                // Collect visible enemies with hitchances
                game_object* visible_enemies[5] = {nullptr};
                float hitchances[5] = {0.0f};
                const char* names[5] = {nullptr};
                int count = 0;

                for (auto* enemy : heroes)
                {
                    if (!enemy || !enemy->is_valid() || enemy->is_dead())
                        continue;

                    // Skip allies - only show enemies
                    if (enemy->get_team_id() == local_team)
                        continue;

                    // Check if we have a hit chance for this enemy
                    uint32_t enemy_id = enemy->get_network_id();
                    auto it = g_hit_chance_map.find(enemy_id);
                    if (it != g_hit_chance_map.end() && count < 5)
                    {
                        visible_enemies[count] = enemy;
                        hitchances[count] = it->second * 100.0f;  // Convert to percentage
                        names[count] = enemy->get_char_name();
                        count++;
                    }
                }

                // Draw all collected enemies
                if (count > 0)
                {
                    PredictionVisuals::draw_debug_hitchance_all(visible_enemies, hitchances, names, count);
                }
            }
        }

        // Draw hit chance % below enemy feet
        if (PredictionSettings::get().enable_hit_chance_display && g_sdk->object_manager)
        {
            auto* local_player = g_sdk->object_manager->get_local_player();
            if (!local_player) return;

            int local_team = local_player->get_team_id();
            auto heroes = g_sdk->object_manager->get_heroes();
            for (auto* enemy : heroes)
            {
                if (!enemy || !enemy->is_valid() || enemy->is_dead())
                    continue;

                // Skip allies - only show hit chance for enemies
                if (enemy->get_team_id() == local_team)
                    continue;

                // Check if we have a hit chance for this enemy
                uint32_t enemy_id = enemy->get_network_id();
                auto it = g_hit_chance_map.find(enemy_id);
                if (it != g_hit_chance_map.end())
                {
                    float hit_chance = it->second;

                    // Get enemy position (below feet)
                    math::vector3 enemy_pos = enemy->get_position();
                    enemy_pos.y -= 50.f;  // Slightly below feet

                    // Convert to screen space
                    math::vector2 screen_pos = g_sdk->renderer->world_to_screen(enemy_pos);

                    // Format text
                    char text_buffer[32];
                    snprintf(text_buffer, sizeof(text_buffer), "%.0f%%", hit_chance * 100.f);

                    // Choose color based on hit chance
                    uint32_t color;
                    if (hit_chance >= 0.75f)
                        color = 0xFF00FF00;  // Green (high)
                    else if (hit_chance >= 0.5f)
                        color = 0xFFFFFF00;  // Yellow (medium)
                    else if (hit_chance >= 0.25f)
                        color = 0xFFFF8800;  // Orange (low)
                    else
                        color = 0xFFFF0000;  // Red (very low)

                    // Draw text centered
                    g_sdk->renderer->add_text(text_buffer, 14.f, screen_pos, 0x01, color);  // 0x01 = centered
                }
            }
        }
    }
    catch (...) { /* Prevent any crash from draw callback */ }
}

namespace Prediction
{
    void LoadPrediction()
    {
        // CRITICAL: Validate SDK before registering callbacks
        if (!g_sdk || !g_sdk->event_manager)
            return;

        // Register update callback for tracker updates
        g_sdk->event_manager->register_callback(event_manager::event::game_update, reinterpret_cast<void*>(on_update));

        // Register draw callback for visual indicators
        g_sdk->event_manager->register_callback(event_manager::event::draw_world, reinterpret_cast<void*>(on_draw));

        // Create menu category for settings
        g_menu = g_sdk->menu_manager->add_category("danny_prediction", "Danny.Prediction");

        if (g_menu)
        {
            g_menu->add_label("Debug & Logging");

            g_menu->add_checkbox("debug_logs", "Enable Debug Logs", false, [](bool value) {
                PredictionSettings::get().enable_debug_logging = value;
                });

            g_menu->add_checkbox("telemetry", "Enable Telemetry", true, [](bool value) {
                PredictionSettings::get().enable_telemetry = value;

                // Reinitialize telemetry with the new setting
                if (g_sdk && g_sdk->object_manager)
                {
                    game_object* local_player = g_sdk->object_manager->get_local_player();
                    if (local_player)
                    {
                        std::string champion_name = local_player->get_char_name();
                        PredictionTelemetry::TelemetryLogger::initialize(champion_name, value);
                    }
                }
                });

            g_menu->add_checkbox("visuals", "Draw Predictions", false, [](bool value) {
                PredictionSettings::get().enable_visuals = value;
                });

            g_menu->add_checkbox("hit_chance_display", "Show Hit Chance %", false, [](bool value) {
                PredictionSettings::get().enable_hit_chance_display = value;
                });

            g_menu->add_checkbox("debug_hitchance_text", "Debug Hitchance (Left Side)", false, [](bool value) {
                PredictionVisuals::VisualsSettings::get().draw_debug_text = value;
                });

            g_menu->add_hotkey("output_telemetry", "Output Telemetry Report", 0, false, false, [](std::string*, bool pressed) {
                if (pressed && g_sdk)
                {
                    g_sdk->log_console("[Danny.Prediction] ===== TELEMETRY REPORT =====");
                    PredictionTelemetry::TelemetryLogger::write_report();
                }
                });

            g_menu->add_label("Prediction Features");

            g_menu->add_checkbox("dash_pred", "Dash Endpoint Prediction", true, [](bool value) {
                PredictionSettings::get().enable_dash_prediction = value;
                });
        }

        // Initialize telemetry system with local player's champion name
        if (g_sdk && g_sdk->object_manager)
        {
            game_object* local_player = g_sdk->object_manager->get_local_player();
            if (local_player)
            {
                std::string champion_name = local_player->get_char_name();
                PredictionTelemetry::TelemetryLogger::initialize(
                    champion_name,
                    PredictionSettings::get().enable_telemetry
                );
            }
        }

        if (PredictionSettings::get().enable_debug_logging)
            g_sdk->log_console("[Danny.Prediction] Loaded - visuals and trackers initialized");
    }

    void UnloadPrediction()
    {
        // Write telemetry report before unloading
        if (g_sdk && PredictionSettings::get().enable_telemetry)
        {
            g_sdk->log_console("[Danny.Prediction] ===== SESSION TELEMETRY REPORT =====");
            PredictionTelemetry::TelemetryLogger::write_report();
        }

        // Unregister callbacks (safe even if SDK is null)
        if (g_sdk && g_sdk->event_manager)
        {
            g_sdk->event_manager->unregister_callback(event_manager::event::game_update, reinterpret_cast<void*>(on_update));
            g_sdk->event_manager->unregister_callback(event_manager::event::draw_world, reinterpret_cast<void*>(on_draw));
        }

        // Clean up all subsystems (always safe to call)
        // GeometricPred is stateless - no cleanup needed
        FogOfWarTracker::clear();
        PredictionVisuals::clear();

        if (g_sdk && PredictionSettings::get().enable_debug_logging)
            g_sdk->log_console("[Danny.Prediction] Unloaded - all subsystems cleared");
    }
}

// Name export - makes it appear in Prediction dropdown instead of as toggle
// Following naming convention: AuthorName.Prediction
extern "C" __declspec(dllexport) const char* Name = "Danny.Prediction";

extern "C" __declspec(dllexport) int SDKVersion = SDK_VERSION;
extern "C" __declspec(dllexport) module_type Type = module_type::pred;

extern "C" __declspec(dllexport) bool PluginLoad(core_sdk* sdk, void** custom_sdk)
{
    g_sdk = sdk;

    // Validate all prerequisites BEFORE setting any pointers
    if (!sdk_init::target_selector())
    {
        return false;
    }

    if (!g_sdk->object_manager || !g_sdk->object_manager->get_local_player())
    {
        return false;
    }

    // All validations passed - now set pointers
    *custom_sdk = &customPrediction;
    sdk::prediction = &customPrediction;

    MyHeroNamePredCore = g_sdk->object_manager->get_local_player()->get_char_name();

    Prediction::LoadPrediction();

    return true;
}

extern "C" __declspec(dllexport) void PluginUnload()
{
    Prediction::UnloadPrediction();
}