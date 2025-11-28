#pragma once

/**
 * =============================================================================
 * PREDICTION SETTINGS & CONFIGURATION
 * =============================================================================
 *
 * Global settings for Danny Prediction SDK
 * Accessed via PredictionSettings::get()
 *
 * =============================================================================
 */

namespace PredictionSettings
{
    /**
     * Stores last spell data for dump/debug feature
     */
    struct LastSpellData
    {
        float range = 0.f;
        float radius = 0.f;
        float delay = 0.f;
        float speed = 0.f;
        int spell_type = 0;
        int hitchance = 0;
        bool valid = false;
    };

    /**
     * Configuration settings with menu integration
     */
    struct Settings
    {
        // Last spell data for debug dump
        LastSpellData last_spell_data;
        // Debug settings
        bool enable_debug_logging = false;  // Verbose console logging
        bool enable_telemetry = true;       // Log predictions to file for analysis
        bool enable_visuals = false;        // Draw prediction indicators (disabled by default)
        bool enable_physics_measurement = false;  // Measure acceleration/deceleration (for calibration)

        // Edge case toggles
        bool enable_dash_prediction = true;  // Predict at dash endpoints

        // Performance settings
        int grid_search_resolution = 8;  // Grid search size (8 = balanced, 16 = high quality)

        // Advanced settings
        bool enable_cs_prediction = false;  // Predict toward low HP minions (experimental)
        bool enable_hp_pressure = true;     // Low HP retreat bias

        Settings() {}
    };

    /**
     * Get global configuration instance
     */
    inline Settings& get()
    {
        static Settings instance;
        return instance;
    }

} // namespace PredictionSettings

// Helper macro for debug logging
#define PRED_DEBUG_LOG(msg) if (PredictionSettings::get().enable_debug_logging) g_sdk->log_console(msg)
#define PRED_DEBUG_LOG_FMT(buffer, fmt, ...) \
    if (PredictionSettings::get().enable_debug_logging) { \
        snprintf(buffer, sizeof(buffer), fmt, __VA_ARGS__); \
        g_sdk->log_console(buffer); \
    }