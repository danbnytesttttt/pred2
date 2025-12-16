#pragma once

#include "sdk.hpp"
#include "GeometricPrediction.h"
#include <string>

/**
 * =============================================================================
 * CUSTOM PREDICTION SDK IMPLEMENTATION
 * =============================================================================
 *
 * This class implements the pred_sdk interface and integrates the hybrid
 * prediction system as a custom prediction plugin.
 *
 * Usage:
 * ------
 * 1. Declare instance: CustomPredictionSDK customPrediction;
 * 2. Export as module_type::pred
 * 3. Return pointer in PluginLoad via custom_sdk parameter
 * 4. Update prediction manager every frame
 *
 * Example:
 * --------
 * CustomPredictionSDK customPrediction;
 *
 * extern "C" __declspec(dllexport) int SDKVersion = SDK_VERSION;
 * extern "C" __declspec(dllexport) module_type Type = module_type::pred;
 *
 * extern "C" __declspec(dllexport) bool PluginLoad(core_sdk* sdk, void** custom_sdk) {
 *     g_sdk = sdk;
 *
 *     if (!sdk_init::target_selector()) {
 *         return false;
 *     }
 *
 *     *custom_sdk = &customPrediction;
 *
 *     return true;
 * }
 *
 * extern "C" __declspec(dllexport) void PluginUnload() {
 *     // GeometricPred is stateless - no cleanup needed
 * }
 *
 * =============================================================================
 */

class CustomPredictionSDK : public pred_sdk
{
public:
    CustomPredictionSDK();
    ~CustomPredictionSDK();

    // =========================================================================
    // PRED_SDK INTERFACE IMPLEMENTATION
    // =========================================================================

    /**
     * Get utility functions
     */
    pred_sdk::utils* util() override;

    /**
     * Get prediction for targeted spells (point-and-click)
     */
    pred_sdk::pred_data targetted(pred_sdk::spell_data spell_data) override;

    /**
     * Get prediction for skillshot (auto-target selection)
     */
    pred_sdk::pred_data predict(pred_sdk::spell_data spell_data) override;

    /**
     * Get prediction for skillshot on specific target
     */
    pred_sdk::pred_data predict(game_object* obj, pred_sdk::spell_data spell_data) override;

    /**
     * Predict position on path (simple linear prediction)
     */
    math::vector3 predict_on_path(game_object* obj, float time, bool use_server_pos = false) override;

    /**
     * Check collision with environment/units
     */
    collision_ret collides(const math::vector3& end_point, pred_sdk::spell_data spell_data, const game_object* target) override;

    // =========================================================================
    // AOE PREDICTION (MULTI-TARGET)
    // =========================================================================

    /**
     * AOE prediction result with multi-target info
     */
    struct aoe_pred_result
    {
        math::vector3 cast_position;          // Optimal cast position for cluster
        float expected_hits;                   // Expected number of targets hit (weighted sum)
        int targets_in_range;                  // Number of valid targets considered
        std::vector<game_object*> hit_targets; // Targets expected to be hit
        std::vector<float> hit_chances;        // Hit chance for each target
        float min_hit_chance;                  // Lowest individual hit chance
        float avg_hit_chance;                  // Average hit chance across targets
        bool is_valid;                         // True if meets minimum requirements

        aoe_pred_result() : expected_hits(0.f), targets_in_range(0),
            min_hit_chance(0.f), avg_hit_chance(0.f), is_valid(false) {
        }
    };

    /**
     * Predict optimal cast position for AOE spell to hit multiple targets
     *
     * @param spell_data Spell configuration
     * @param min_hits Minimum targets required (default 2)
     * @param min_single_hc Minimum hit chance per target (default 0.25)
     * @param priority_weighted Weight by target priority (carries > tanks)
     * @return AOE prediction result with optimal position and expected hits
     */
     /**
      * Predict optimal cast position for circular AOE spell
      */
    aoe_pred_result predict_aoe_cluster(
        pred_sdk::spell_data spell_data,
        int min_hits = 2,
        float min_single_hc = 0.25f,
        bool priority_weighted = false
    );

    /**
     * Predict optimal cast direction for linear AOE spell
     * Returns cast_position at end of line (max range in best direction)
     */
    aoe_pred_result predict_linear_aoe(
        pred_sdk::spell_data spell_data,
        int min_hits = 2,
        float min_single_hc = 0.25f,
        bool priority_weighted = false
    );

    /**
     * Predict optimal cast direction for cone AOE spell
     * Tests multiple directions and returns the one hitting most targets within cone angle
     */
    aoe_pred_result predict_cone_aoe(
        pred_sdk::spell_data spell_data,
        int min_hits = 2,
        float min_single_hc = 0.25f,
        bool priority_weighted = false
    );

    /**
     * Auto-routing AOE prediction based on spell_type
     * - circular -> predict_aoe_cluster
     * - linear/vector -> predict_linear_aoe
     * - cone (auto-detected) -> predict_cone_aoe
     */
    aoe_pred_result predict_aoe(
        pred_sdk::spell_data spell_data,
        int min_hits = 2,
        float min_single_hc = 0.25f,
        bool priority_weighted = false
    );

    // =========================================================================
    // UTILITY CLASS IMPLEMENTATION
    // =========================================================================

    class CustomPredictionUtils : public pred_sdk::utils
    {
    public:
        /**
         * Get effective spell range considering target hitbox
         */
        float get_spell_range(pred_sdk::spell_data& data, game_object* target, game_object* source) override;

        /**
         * Check if position is within spell range
         */
        bool is_in_range(pred_sdk::spell_data& data, math::vector3 cast_position, game_object* target) override;

        /**
         * Get time for spell to hit target at position
         */
        float get_spell_hit_time(pred_sdk::spell_data& data, math::vector3 pos, game_object* target = nullptr) override;

        /**
         * Get time for target to escape spell range
         */
        float get_spell_escape_time(pred_sdk::spell_data& data, game_object* target) override;
    };

    // =========================================================================
    // UPDATE FUNCTION (CALL EVERY FRAME)
    // =========================================================================

    /**
     * Update all behavior trackers
     * IMPORTANT: Call this every frame in your main loop or on_update callback
     */
    static void update_trackers();

private:
    CustomPredictionUtils utils_;

    // =========================================================================
    // HELPER FUNCTIONS
    // =========================================================================

    /**
     * Convert geometric prediction result to pred_sdk::pred_data
     */
    pred_sdk::pred_data convert_geometric_to_pred_data(
        const GeometricPred::PredictionResult& geo_result,
        game_object* target,
        const pred_sdk::spell_data& spell_data
    );

    /**
     * Convert hit chance float [0,1] to hitchance enum
     */
    pred_sdk::hitchance convert_hit_chance_to_enum(float hit_chance);

    /**
     * Get best target for spell using hybrid prediction
     */
    game_object* get_best_target(const pred_sdk::spell_data& spell_data);

    /**
     * Calculate hit chance score for target prioritization
     */
    float calculate_target_score(game_object* target, const pred_sdk::spell_data& spell_data);

    /**
     * Simple collision check (simplified implementation)
     */
    bool check_collision_simple(
        const math::vector3& start,
        const math::vector3& end,
        const pred_sdk::spell_data& spell_data,
        const game_object* target
    );

    /**
     * Check if object blocks skillshot
     */
    bool is_collision_object(game_object* obj, const pred_sdk::spell_data& spell_data);
};

// =============================================================================
// INLINE IMPLEMENTATIONS
// =============================================================================

inline CustomPredictionSDK::CustomPredictionSDK()
{
}

inline CustomPredictionSDK::~CustomPredictionSDK()
{
    // GeometricPred is stateless - no cleanup needed
}

inline pred_sdk::utils* CustomPredictionSDK::util()
{
    return &utils_;
}

inline void CustomPredictionSDK::update_trackers()
{
    // GeometricPred is stateless - no per-frame updates needed
    // FogOfWarTracker updates are handled per-prediction, not globally
}