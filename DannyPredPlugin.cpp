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
                });

            g_menu->add_checkbox("visuals", "Draw Predictions", false, [](bool value) {
                PredictionSettings::get().enable_visuals = value;
                });

            g_menu->add_checkbox("physics_measure", "Measure Physics (Calibration)", false, [](bool value) {
                PredictionSettings::get().enable_physics_measurement = value;
                });

            g_menu->add_checkbox("output_telemetry", "Output Telemetry Report", false, [](bool value) {
                if (value && g_sdk)
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
        HybridPred::PredictionManager::clear();
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