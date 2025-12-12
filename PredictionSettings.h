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
        bool enable_hit_chance_display = false;  // Show hit chance % below enemy feet

        // Edge case toggles
        bool enable_dash_prediction = true;  // Predict at dash endpoints

        // Geometric prediction settings
        bool prefer_safe_shots = false;      // Only cast on VeryHigh/Undodgeable (more conservative)

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