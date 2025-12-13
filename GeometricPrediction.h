#pragma once

/**
 * ============================================================================
 * GEOMETRIC PREDICTION SYSTEM
 * ============================================================================
 *
 * A lightweight, deterministic prediction engine based on pure geometry.
 *
 * Philosophy:
 * - Linear path following (no acceleration modeling)
 * - Geometric constraints (can they physically escape?)
 * - Environmental validation (minions, walls, windwall)
 * - Time-To-Exit (TTE) grading for confidence levels
 *
 * Replaces the 4000-line hybrid system with ~700 lines of focused logic.
 *
 * Core Components:
 * 1. predict_linear_path()  - The Engine (where will they be?)
 * 2. calculate_hitchance()  - The Brain  (should we cast?)
 * 3. get_prediction()       - The Driver (main entry point)
 *
 * Integration:
 * - EdgeCaseDetection.h: Stasis, dash, windwall, minion, clone detection
 * - PredictionSettings.h: User configuration
 * - PredictionTelemetry.h: Performance tracking
 * ============================================================================
 */

#include "sdk.hpp"
#include "EdgeCaseDetection.h"
#include "PredictionSettings.h"
#include "PredictionTelemetry.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>

namespace GeometricPred
{
    // =========================================================================
    // CONSTANTS
    // =========================================================================

    constexpr float PI = 3.14159265358979323846f;
    constexpr float EPSILON = 1e-6f;

    // Reaction time thresholds (for grading confidence)
    constexpr float REACTION_UNDODGEABLE = 0.0f;   // No time to react
    constexpr float REACTION_VERY_HIGH = 0.1f;     // 100ms window
    constexpr float REACTION_HIGH = 0.25f;         // 250ms window
    constexpr float REACTION_MEDIUM = 0.4f;        // 400ms window
    // Anything > 0.4s = Low confidence

    // Path prediction - use SDK data, no arbitrary heuristics

    // Minion collision constants
    constexpr float MINION_SEARCH_RADIUS = 150.f;  // Search radius around spell path (algorithm parameter)
    constexpr float MINION_RELEVANCE_RANGE = 2000.f;  // Only check minions within this range (performance optimization)
    constexpr float LANE_BASE_DPS = 50.f;          // Fallback DPS estimate when health prediction SDK unavailable

    // AOE prediction - calculate movement buffer based on spell parameters
    constexpr float AOE_MAX_MOVE_SPEED = 450.f;  // Assume max 450 MS for range filtering (high but not unrealistic)

    // =========================================================================
    // ENUMS
    // =========================================================================

    /**
     * Spell collision shapes
     * Circle: AoE explosions (Annie R, Lux E, Ziggs Q/W/E/R, Orianna R)
     * Capsule: Linear skillshots (Morgana Q, Blitz Hook, Xerath E, Lux Q)
     *          Also covers "rectangles" (Xerath Q = capsule with length)
     * Line: Same as Capsule (alias for compatibility)
     * Cone: Wedge-shaped AoE (Annie W, Cassiopeia Q, Rumble E)
     */
    enum class SpellShape
    {
        Circle,   // Point-target AoE
        Capsule,  // Line + width (missiles, most skillshots)
        Line,     // Alias for Capsule
        Cone,     // Wedge-shaped directional AoE
        Vector    // Two-position cast (Viktor E, Rumble R, Taliyah W)
    };

    /**
     * Graded hit chance levels
     * Maps reaction_window to confidence levels + edge case states
     */
    enum class HitChance
    {
        Impossible,     // Target is dead, invulnerable, out of range, or spell blocked
        Clone,          // Target is a clone (Shaco, Wukong, LeBlanc, Neeko)
        SpellShielded,  // Target has spell shield (Sivir E, Banshee's, Malz passive)
        Windwalled,     // Windwall blocks the spell (Yasuo W, Braum E, Samira W)
        MinionBlocked,  // Minion collision detected
        Low,            // >400ms reaction window - easily dodgeable
        Medium,         // 250-400ms window - requires attention
        High,           // 100-250ms window - difficult to dodge
        VeryHigh,       // <100ms window - requires quick reaction
        Undodgeable,    // Physically impossible to escape or no time to react
        Immobile,       // CC'd and cannot move
        Stasis,         // In stasis (Zhonya's, Bard R, etc.) - special timing needed
        Channeling,     // Channeling or recalling - high priority stationary target
        Dashing         // Dashing - predict to endpoint
    };

    // =========================================================================
    // DATA STRUCTURES
    // =========================================================================

    /**
     * Input parameters for prediction
     */
    struct PredictionInput
    {
        game_object* source;        // Casting champion
        game_object* target;        // Target to predict
        SpellShape shape;           // Spell collision shape
        float spell_width;          // Spell radius (circle) or width (capsule)
        float spell_range;          // Spell max range (for capsules/vectors - vector length)
        float missile_speed;        // Projectile speed (0 = instant)
        float cast_delay;           // Windup/cast time before spell launches
        float proc_delay;           // Additional delay before damage (e.g., Syndra Q = 0.6s)

        // Vector spell parameters (Viktor E, Rumble R, Taliyah W)
        math::vector3 first_cast_position;  // Start position for vector spells (if zero, auto-calculated)
        float cast_range;                   // Max range for first cast position (default = spell_range)

        PredictionInput()
            : source(nullptr), target(nullptr), shape(SpellShape::Capsule),
              spell_width(70.f), spell_range(1000.f), missile_speed(1500.f),
              cast_delay(0.25f), proc_delay(0.f),
              first_cast_position(math::vector3{}), cast_range(0.f)
        {}
    };

    /**
     * Prediction result with cast position and confidence
     */
    struct PredictionResult
    {
        math::vector3 cast_position;        // Where to aim (end position for vectors)
        math::vector3 first_cast_position;  // Start position for vector spells (Viktor E, Rumble R)
        HitChance hit_chance;               // Confidence level
        bool should_cast;                   // Simple yes/no recommendation

        // Core metrics
        float hit_chance_float;       // Hit chance as 0-1 float (for telemetry)
        float reaction_window;        // Time enemy has to dodge (seconds)
        float time_to_impact;         // Spell arrival time (seconds)
        float distance_to_exit;       // Distance enemy must travel to escape

        // Edge case tracking (for telemetry)
        bool is_stasis;               // Target in stasis
        bool is_dash;                 // Target dashing
        bool is_channeling;           // Target channeling/recalling
        bool is_immobile;             // Target CC'd
        bool is_animation_locked;     // Target in AA/cast animation
        bool is_slowed;               // Target slowed
        bool is_csing;                // Target is likely last-hitting (attention divided)
        float dash_confidence;        // Dash-specific confidence multiplier

        // Collision tracking
        bool minion_collision;        // Minion blocks spell
        bool windwall_detected;       // Windwall blocks spell
        bool spell_shield_detected;   // Spell shield active
        bool is_clone_target;         // Target is clone

        // Position data (for telemetry)
        math::vector3 target_current_pos;   // Target's current position
        math::vector3 predicted_position;   // Where we predicted they'll be
        float prediction_offset;            // Distance between current and predicted
        float distance_to_target;           // Distance from source to target
        bool target_is_moving;              // Whether target is moving
        float target_velocity;              // Target's movement speed

        // Timing data
        float stasis_wait_time;       // How long to wait before casting (stasis)
        float computation_time_ms;    // How long prediction took

        // Debug
        const char* block_reason;     // Why we can't cast (if Impossible)
        const char* edge_case_type;   // "normal", "stasis", "dash", "channeling"

        PredictionResult()
            : cast_position{}, first_cast_position{}, hit_chance(HitChance::Impossible), should_cast(false),
              hit_chance_float(0.f), reaction_window(0.f), time_to_impact(0.f), distance_to_exit(0.f),
              is_stasis(false), is_dash(false), is_channeling(false), is_immobile(false),
              is_animation_locked(false), is_slowed(false), is_csing(false), dash_confidence(1.0f),
              minion_collision(false), windwall_detected(false), spell_shield_detected(false),
              is_clone_target(false), target_current_pos{}, predicted_position{},
              prediction_offset(0.f), distance_to_target(0.f), target_is_moving(false),
              target_velocity(0.f), stasis_wait_time(0.f), computation_time_ms(0.f),
              block_reason(nullptr), edge_case_type("normal")
        {}
    };

    // =========================================================================
    // UTILITY FUNCTIONS (Extracted & Enhanced from Hybrid)
    // =========================================================================

    namespace Utils
    {
        /**
         * 2D DISTANCE HELPERS
         * League logic is 2D (X/Z plane). Height (Y) should not affect distance.
         * Example: River (Y=-50) to Mid Lane (Y=50) - height doesn't inflate range.
         */

        inline float distance_2d(const math::vector3& a, const math::vector3& b)
        {
            float dx = a.x - b.x;
            float dz = a.z - b.z;
            return std::sqrt(dx * dx + dz * dz);
        }

        inline float sqr_distance_2d(const math::vector3& a, const math::vector3& b)
        {
            float dx = a.x - b.x;
            float dz = a.z - b.z;
            return dx * dx + dz * dz;
        }

        inline float magnitude_2d(const math::vector3& v)
        {
            return std::sqrt(v.x * v.x + v.z * v.z);
        }

        inline math::vector3 flatten_2d(const math::vector3& v)
        {
            return math::vector3(v.x, 0.f, v.z);
        }

        /**
         * TENACITY ESTIMATION
         * Tenacity reduces CC duration (Mercury Treads, runes, champion passives)
         * Returns estimated tenacity as 0-1 multiplier (0.3 = 30% tenacity)
         */
        inline float estimate_tenacity(game_object* target)
        {
            if (!target) return 0.f;

            // Use SDK to get actual tenacity value
            // Includes: Mercury Treads, Legend Tenacity, champion passives, etc.
            return target->get_percent_cc_reduction();
        }

        /**
         * INVULNERABILITY DETECTION
         * Targets that literally cannot be hit (Zhonya's, Fizz E, Vlad W, etc.)
         */
        inline bool is_invulnerable(game_object* target)
        {
            if (!target) return false;

            // Check invulnerability buff
            if (target->has_buff_of_type(buff_type::invulnerable))
                return true;

            // Check untargetable state (Fizz E, Vlad W, Yi Q, Elise E, Xayah R, Kayn R)
            if (!target->is_targetable())
                return true;

            // Check stasis (Zhonya's, Bard R)
            // Note: Stasis might not be a separate buff type in all SDKs
            // May need to check specific buff names instead

            return false;
        }

        /**
         * SPELL SHIELD DETECTION
         * Shields that block the first incoming spell (Sivir E, Banshee's, etc.)
         */
        inline bool has_spell_shield(game_object* target)
        {
            if (!target) return false;

            // Check for spell shield buff type
            if (target->has_buff_of_type(buff_type::spell_shield))
                return true;

            // Fallback: Check common spell shield buff names
            auto buffs = target->get_buffs();
            for (auto* buff : buffs)
            {
                if (!buff || !buff->is_active()) continue;

                const char* name = buff->get_name();
                if (name && (
                    std::strstr(name, "SivirE") ||
                    std::strstr(name, "SivirShield") ||
                    std::strstr(name, "NocturneW") ||
                    std::strstr(name, "NocturneShroudofDarkness") ||
                    std::strstr(name, "BansheesVeil") ||
                    std::strstr(name, "MalzaharPassive") ||
                    std::strstr(name, "MorganaE") ||
                    std::strstr(name, "BlackShield")))
                {
                    return true;
                }
            }

            return false;
        }

        /**
         * ANIMATION LOCK TIME
         * How long target is locked in place (AA windup, spell cast, channel)
         * Filters out moving-cast spells (Syndra Q, Orianna Q, etc.)
         */
        inline float get_animation_lock_time(game_object* target)
        {
            if (!target) return 0.f;

            // Check if target is moving (filters moving-cast spells)
            if (target->is_moving())
            {
                // If they have a movement path while casting, it's a moving-cast (not locked)
                auto path = target->get_path();
                if (!path.empty() && path.back().distance(target->get_position()) > 10.f)
                {
                    return 0.f;  // Moving-cast, not locked
                }
            }

            auto active_cast = target->get_active_spell_cast();
            if (!active_cast) return 0.f;

            auto spell_cast = active_cast->get_spell_cast();
            if (!spell_cast) return 0.f;

            if (!g_sdk || !g_sdk->clock_facade) return 0.f;

            float current_time = g_sdk->clock_facade->get_game_time();
            float cast_start = active_cast->get_cast_start_time();

            // Get windup time (accounts for attack speed if AA)
            float windup = 0.f;
            if (spell_cast->is_basic_attack())
            {
                windup = target->get_attack_cast_delay();
            }
            else
            {
                windup = spell_cast->get_cast_delay();
            }

            float end_time = cast_start + windup;
            float remaining = std::max(0.f, end_time - current_time);

            // Also check for stationary channels (Malz R, Recall, Kat R, MF R)
            float channel_end = active_cast->get_cast_channeling_end_time();
            if (channel_end > 0.f && !target->is_moving())
            {
                // Stationary channel
                float channel_remaining = std::max(0.f, channel_end - current_time);
                remaining = std::max(remaining, channel_remaining);
            }

            return remaining;
        }

        /**
         * COMPUTE ARRIVAL TIME
         * When will the spell hit the target? (accounts for ping, cast delay, travel time)
         * Extracted from HybridPrediction.cpp:2334-2379
         */
        inline float compute_arrival_time(
            const math::vector3& source_pos,
            const math::vector3& target_pos,
            float projectile_speed,
            float cast_delay,
            float proc_delay = 0.f)
        {
            float distance = distance_2d(source_pos, target_pos);

            // Instant spell (no projectile travel)
            if (projectile_speed < EPSILON || projectile_speed >= 1e10f)
            {
                return cast_delay + proc_delay;
            }

            // PING COMPENSATION: Account for one-way network delay
            // Server doesn't receive our cast command until ping/2 seconds from now
            float ping_delay = 0.f;
            if (g_sdk && g_sdk->net_client)
            {
                float ping_ms = static_cast<float>(g_sdk->net_client->get_ping());
                ping_delay = ping_ms / 2000.f;  // One-way, convert to seconds
                ping_delay = std::clamp(ping_delay, 0.005f, 0.15f);  // 5-150ms range
            }

            return ping_delay + cast_delay + proc_delay + (distance / projectile_speed);
        }

        /**
         * EFFECTIVE REACTION TIME (FOG OF WAR ADJUSTMENT)
         * If enemy can see our cast animation, they react during our windup
         * If hidden (fog/brush), they only react when spell appears
         * Extracted from HybridPrediction.h:266-294
         */
        inline float get_effective_reaction_time(
            game_object* source,
            game_object* target,
            float cast_delay)
        {
            constexpr float HUMAN_REACTION_TIME = 0.20f;  // Average reaction time

            // Default: assume visible cast
            bool is_hidden_from_enemy = false;

            if (g_sdk && g_sdk->nav_mesh && source && target)
            {
                int enemy_team = target->get_team_id();
                math::vector3 source_pos = source->get_position();

                // Check if we're in fog of war for enemy team
                is_hidden_from_enemy = g_sdk->nav_mesh->is_in_fow_for_team(source_pos, enemy_team);
            }

            if (is_hidden_from_enemy)
            {
                // Hidden cast: They only react when spell appears (after cast_delay)
                return HUMAN_REACTION_TIME;
            }
            else
            {
                // Visible cast: They start reacting during our windup
                // Their reaction time is consumed while we cast
                return std::max(0.05f, HUMAN_REACTION_TIME - cast_delay);
            }
        }

        /**
         * MINION COLLISION CHECK
         * Does a minion block the spell path?
         * For linear skillshots only (Morgana Q, Blitz Hook, Thresh Q)
         */
        inline bool has_minion_collision(
            const math::vector3& spell_start,
            const math::vector3& spell_end,
            float spell_width)
        {
            if (!g_sdk || !g_sdk->object_manager) return false;

            // Get all minions
            auto minions = g_sdk->object_manager->get_minions();

            // Calculate spell path direction
            math::vector3 spell_dir = spell_end - spell_start;
            float spell_length = magnitude_2d(spell_dir);
            if (spell_length < EPSILON) return false;

            spell_dir = spell_dir / spell_length;  // Normalize

            // Check each minion for collision
            for (auto* minion : minions)
            {
                if (!minion || !minion->is_valid() || minion->is_dead())
                    continue;

                // Only check enemy/neutral minions
                // (Don't block on allied minions - game doesn't do that for most spells)
                if (minion->get_team_id() == g_sdk->local_player->get_team_id())
                    continue;

                math::vector3 minion_pos = minion->get_position();

                // Project minion position onto spell path
                math::vector3 to_minion = minion_pos - spell_start;
                float proj_length = to_minion.dot(spell_dir);

                // Check if minion is along the spell path (not behind or beyond)
                if (proj_length < 0.f || proj_length > spell_length)
                    continue;

                // Find closest point on spell path to minion
                math::vector3 closest_point = spell_start + spell_dir * proj_length;
                float lateral_dist = distance_2d(minion_pos, closest_point);

                // Check collision: spell radius + actual minion bounding radius from SDK
                float minion_radius = minion->get_bounding_radius();
                float collision_radius = spell_width * 0.5f + minion_radius;
                if (lateral_dist < collision_radius)
                {
                    return true;  // Minion blocks the spell
                }
            }

            return false;  // No collision
        }

        /**
         * TERRAIN BLOCKING (CHOKE DETECTION)
         * Is target trapped against walls? Can they escape left/right?
         * Extracted from HybridPrediction.cpp:2269-2312
         *
         * Returns:
         * - 2.0f if trapped on both sides (guaranteed hit)
         * - 1.5f if one side blocked (predictable dodge)
         * - 1.0f if both sides open (normal)
         */
        inline float get_terrain_blocking_multiplier(
            const math::vector3& target_pos,
            const math::vector3& spell_center,
            float distance_to_exit)
        {
            if (!g_sdk || !g_sdk->nav_mesh) return 1.0f;

            // Calculate aim direction (spell to target)
            math::vector3 aim_dir = target_pos - spell_center;
            float aim_magnitude = magnitude_2d(aim_dir);
            if (aim_magnitude < EPSILON) return 1.0f;

            aim_dir = aim_dir / aim_magnitude;  // Normalize

            // Calculate perpendicular escape directions (left and right)
            math::vector3 escape_dir_left(-aim_dir.z, 0.f, aim_dir.x);   // 90° left
            math::vector3 escape_dir_right(aim_dir.z, 0.f, -aim_dir.x);  // 90° right

            // Calculate escape points (distance_to_exit + safety margin)
            constexpr float SAFETY_MARGIN = 20.f;
            float escape_distance = distance_to_exit + SAFETY_MARGIN;

            math::vector3 escape_point_left = target_pos + escape_dir_left * escape_distance;
            math::vector3 escape_point_right = target_pos + escape_dir_right * escape_distance;

            // Check if escape paths are walkable
            bool can_dodge_left = g_sdk->nav_mesh->is_pathable(escape_point_left);
            bool can_dodge_right = g_sdk->nav_mesh->is_pathable(escape_point_right);

            // Trapped on both sides = guaranteed hit
            if (!can_dodge_left && !can_dodge_right)
                return 2.0f;  // TRAPPED!

            // One side blocked = predictable dodge direction
            if (!can_dodge_left || !can_dodge_right)
                return 1.5f;  // 50% boost

            // Both sides open = normal
            return 1.0f;
        }

        /**
         * WINDWALL DETECTION
         * Is there a windwall blocking the spell path?
         * Checks for Yasuo W, Braum E, Samira W
         */
        inline bool has_windwall_blocking(
            const math::vector3& spell_start,
            const math::vector3& spell_end)
        {
            if (!g_sdk || !g_sdk->object_manager) return false;

            // Get enemy champions
            auto enemies = g_sdk->object_manager->get_enemy_heroes();

            for (auto* enemy : enemies)
            {
                if (!enemy || !enemy->is_valid() || enemy->is_dead())
                    continue;

                // Check for Yasuo W (Wind Wall)
                if (std::strcmp(enemy->get_champion_name(), "Yasuo") == 0)
                {
                    // Check if Yasuo has active Wind Wall
                    // This would require checking for the wall object in object_manager
                    // or checking for specific buff/object type
                    // Implementation depends on SDK capabilities
                    // TODO: Implement if SDK provides wall object detection
                }

                // Check for Braum E (Unbreakable)
                if (std::strcmp(enemy->get_champion_name(), "Braum") == 0)
                {
                    // Braum E blocks projectiles in a direction
                    // Would need to check if he's facing our spell direction
                    // TODO: Implement if critical
                }

                // Check for Samira W (Blade Whirl)
                if (std::strcmp(enemy->get_champion_name(), "Samira") == 0)
                {
                    // Samira W destroys projectiles around her
                    // Check if she has the buff active
                    // TODO: Implement if critical
                }
            }

            // For now, return false (windwall detection requires SDK-specific implementation)
            // Most SDKs don't expose wall objects easily
            return false;
        }

        /**
         * ESCAPE DISTANCE CALCULATION (CIRCLE)
         * How far must target run to escape a circular AoE?
         */
        inline float calculate_escape_distance_circle(
            const math::vector3& target_pos,
            const math::vector3& spell_center,
            float spell_radius,
            float target_radius)
        {
            // Distance from target center to spell edge
            float distance_to_center = distance_2d(target_pos, spell_center);
            float distance_to_exit = spell_radius + target_radius - distance_to_center;

            return std::max(0.f, distance_to_exit);
        }

        /**
         * POINT IN CAPSULE (Extracted from HybridPrediction.cpp:5074-5093)
         * Capsule = line segment + radius
         * Point is inside if distance to line segment ≤ radius
         */
        inline bool point_in_capsule(
            const math::vector3& point,
            const math::vector3& capsule_start,
            const math::vector3& capsule_end,
            float capsule_radius)
        {
            math::vector3 segment = capsule_end - capsule_start;
            math::vector3 to_point = point - capsule_start;

            float segment_length_sq = segment.x * segment.x + segment.z * segment.z;

            if (segment_length_sq < EPSILON)
            {
                // Degenerate case: start == end (treat as sphere)
                float dist_sq = to_point.x * to_point.x + to_point.z * to_point.z;
                return dist_sq <= capsule_radius * capsule_radius;
            }

            // Project point onto segment: t = dot(to_point, segment) / |segment|²
            float t = (to_point.x * segment.x + to_point.z * segment.z) / segment_length_sq;

            // Clamp t to [0,1] to stay on segment (not extend beyond endpoints)
            t = std::clamp(t, 0.f, 1.f);

            // Find closest point on segment
            math::vector3 closest_point = capsule_start;
            closest_point.x += segment.x * t;
            closest_point.z += segment.z * t;

            // Check if point is within radius of closest point on segment
            float dist_sq = sqr_distance_2d(point, closest_point);
            return dist_sq <= capsule_radius * capsule_radius;
        }

        /**
         * POINT IN CONE (Extracted from HybridPrediction.cpp:5214-5244)
         * Cone = circular sector in 3D
         * Point is inside if: distance ≤ range AND angle ≤ half_angle
         */
        inline bool point_in_cone(
            const math::vector3& point,
            const math::vector3& cone_origin,
            const math::vector3& cone_direction,  // Normalized
            float cone_half_angle,                // In radians
            float cone_range)
        {
            math::vector3 to_point = point - cone_origin;
            float distance_sq = to_point.x * to_point.x + to_point.z * to_point.z;

            // Check range first (cheaper than angle check)
            if (distance_sq > cone_range * cone_range)
                return false;

            float distance = std::sqrt(distance_sq);

            // CRASH PROTECTION: Safety margin for division
            constexpr float MIN_SAFE_DISTANCE = 0.01f;
            if (distance < MIN_SAFE_DISTANCE)
                return true;  // At origin or too close - always inside

            // Check angle: cos(angle) = dot(to_point, cone_direction) / |to_point|
            float dot_product = to_point.x * cone_direction.x + to_point.z * cone_direction.z;
            float cos_angle = dot_product / distance;

            // cos decreases as angle increases, so:
            // If cos(actual_angle) >= cos(half_angle), then actual_angle <= half_angle
            float cos_half_angle = std::cos(cone_half_angle);
            return cos_angle >= cos_half_angle;
        }

        /**
         * ESCAPE DISTANCE CALCULATION (CAPSULE)
         * How far must target run to escape a line skillshot?
         * Uses proven point_in_capsule() geometry
         */
        inline float calculate_escape_distance_capsule(
            const math::vector3& target_pos,
            const math::vector3& spell_start,
            const math::vector3& spell_direction,  // Normalized
            float spell_width,
            float spell_range,
            float target_radius)
        {
            // Calculate spell capsule endpoints
            math::vector3 spell_end = spell_start + spell_direction * spell_range;

            // Effective spell radius (spell width + target hitbox)
            float effective_radius = (spell_width * 0.5f) + target_radius;

            // Check if target is currently inside the spell
            if (point_in_capsule(target_pos, spell_start, spell_end, effective_radius))
            {
                // Target is inside - calculate minimum distance to exit

                // Project target onto spell line
                math::vector3 to_target = target_pos - spell_start;
                float proj_length = to_target.dot(spell_direction);
                proj_length = std::clamp(proj_length, 0.f, spell_range);

                // Find closest point on spell line
                math::vector3 closest_point = spell_start + spell_direction * proj_length;

                // Lateral distance to line
                float lateral_dist = distance_2d(target_pos, closest_point);

                // Lateral escape distance (perpendicular)
                float lateral_escape = effective_radius - lateral_dist;

                // Backward escape distance (to start of capsule)
                float backward_escape = proj_length + target_radius;

                // CRITICAL FIX: Forward escape distance (to end of capsule)
                // Target near the front can escape by running forward off the end
                float forward_escape = (spell_range - proj_length) + target_radius;

                // Return minimum of ALL THREE escape paths (lateral, backward, forward)
                return std::max(0.f, std::min({lateral_escape, backward_escape, forward_escape}));
            }
            else
            {
                // Target is outside spell - already safe
                return 0.f;
            }
        }

        /**
         * ESCAPE DISTANCE CALCULATION (CONE)
         * How far must target run to escape a cone AoE?
         *
         * Cone = circular sector from origin
         * Target can escape by:
         * 1. Exiting through cone edge (perpendicular to cone radius)
         * 2. Exiting backward (past cone origin)
         * 3. Exiting forward (beyond cone range)
         */
        inline float calculate_escape_distance_cone(
            const math::vector3& target_pos,
            const math::vector3& cone_origin,
            const math::vector3& cone_direction,  // Normalized
            float cone_half_angle,                // In radians
            float cone_range,
            float target_radius)
        {
            // Check if target is inside cone using existing geometry function
            if (!point_in_cone(target_pos, cone_origin, cone_direction, cone_half_angle, cone_range))
                return 0.f;  // Already outside, no escape needed

            math::vector3 to_target = target_pos - cone_origin;
            float distance = magnitude_2d(to_target);

            // SAFETY: If at origin, already escaped (impossible to be outside a cone at its origin)
            constexpr float MIN_SAFE_DISTANCE = 0.01f;
            if (distance < MIN_SAFE_DISTANCE)
                return target_radius;  // Minimum escape distance

            // Normalize to_target for angle calculations
            math::vector3 to_target_norm = to_target / distance;

            // Calculate actual angle from cone center
            float dot_product = to_target_norm.x * cone_direction.x + to_target_norm.z * cone_direction.z;
            float cos_angle = std::clamp(dot_product, -1.f, 1.f);
            float actual_angle = std::acos(cos_angle);

            // ESCAPE OPTION 1: Exit through side of cone (perpendicular escape)
            // Distance = distance_from_center * sin(angle_difference)
            // angle_difference = actual_angle - cone_half_angle
            float angle_to_edge = std::max(0.f, actual_angle - cone_half_angle);
            float side_escape = distance * std::sin(angle_to_edge) + target_radius;

            // ESCAPE OPTION 2: Exit backward (past cone origin)
            float backward_escape = distance + target_radius;

            // ESCAPE OPTION 3: Exit forward (beyond cone range)
            float forward_escape = cone_range - distance + target_radius;

            // Return shortest escape path
            return std::max(0.f, std::min({side_escape, backward_escape, forward_escape}));
        }

    } // namespace Utils

    // =========================================================================
    // CORE PREDICTION ENGINE
    // =========================================================================

    /**
     * THE ENGINE: Predict target position via linear path following
     *
     * No acceleration modeling - just constant velocity along path waypoints
     * Includes intelligent heuristics:
     * - Start-of-path dampening (acceleration phase)
     * - End-of-path clamping (don't overshoot destination)
     * - Path staleness (long animation locks invalidate paths)
     *
     * Extracted from HybridPrediction.cpp:1957-2097 (predict_on_path)
     */
    inline math::vector3 predict_linear_path(
        game_object* target,
        float prediction_time,
        float animation_lock_time = 0.f)
    {
        if (!target || !target->is_valid() || target->is_dead())
            return math::vector3{};

        // Use server position (authoritative, avoids 30-100ms client lag)
        math::vector3 position = target->get_server_position();

        // CC CHECK: Check if target is immobilized (hard CC)
        // Includes knockback (prevents following normal path during knockback)
        bool has_hard_cc = target->has_buff_of_type(buff_type::stun) ||
            target->has_buff_of_type(buff_type::snare) ||
            target->has_buff_of_type(buff_type::charm) ||
            target->has_buff_of_type(buff_type::fear) ||
            target->has_buff_of_type(buff_type::taunt) ||
            target->has_buff_of_type(buff_type::suppression) ||
            target->has_buff_of_type(buff_type::knockup) ||
            target->has_buff_of_type(buff_type::knockback) ||
            target->has_buff_of_type(buff_type::asleep);

        // CC DURATION CHECK: If CC'd, check if they wake up before spell arrives
        // CRITICAL FIX: Don't treat partial CC as full immobile
        // Example: Target stunned for 0.2s, spell arrives in 1.0s → they move for 0.8s
        // TENACITY FIX: Apply tenacity reduction to CC duration
        float cc_remaining = 0.f;
        if (has_hard_cc && g_sdk && g_sdk->clock_facade)
        {
            float current_time = g_sdk->clock_facade->get_game_time();
            float max_cc_end = 0.f;
            float max_cc_start = 0.f;

            auto buffs = target->get_buffs();
            for (auto* buff : buffs)
            {
                if (!buff || !buff->is_active()) continue;
                buff_type t = buff->get_type();
                if (t == buff_type::stun || t == buff_type::snare || t == buff_type::charm ||
                    t == buff_type::fear || t == buff_type::taunt || t == buff_type::suppression ||
                    t == buff_type::knockup || t == buff_type::knockback || t == buff_type::asleep)
                {
                    float end_time = buff->get_end_time();
                    if (end_time > max_cc_end)
                    {
                        max_cc_end = end_time;
                        max_cc_start = buff->get_start_time();
                    }
                }
            }

            // Calculate base CC duration from buff
            float base_duration = max_cc_end - max_cc_start;

            // Apply tenacity reduction (knockups are NOT reduced by tenacity!)
            // Check if the CC is knockup/knockback (immune to tenacity)
            bool is_airborne = false;
            for (auto* buff : buffs)
            {
                if (!buff || !buff->is_active()) continue;
                buff_type t = buff->get_type();
                if (t == buff_type::knockup || t == buff_type::knockback)
                {
                    is_airborne = true;
                    break;
                }
            }

            float tenacity = 0.f;
            if (!is_airborne)
            {
                tenacity = Utils::estimate_tenacity(target);
            }

            // Reduce duration by tenacity: actual_duration = base * (1 - tenacity)
            float actual_duration = base_duration * (1.0f - tenacity);
            float adjusted_cc_end = max_cc_start + actual_duration;

            cc_remaining = std::max(0.f, adjusted_cc_end - current_time);

            // If CC lasts longer than prediction time, target is immobile
            if (cc_remaining >= prediction_time)
            {
                return position;  // CC'd for entire duration
            }

            // Otherwise, they'll move for part of the time (handled below with animation_lock_time)
            // Treat CC as an animation lock
            animation_lock_time = std::max(animation_lock_time, cc_remaining);
        }

        auto path = target->get_path();

        // No path or stationary
        if (path.size() <= 1)
            return position;

        float move_speed = target->get_move_speed();
        if (move_speed < 1.f)
            return position;

        // ANIMATION LOCK: Stop-then-go model
        // Target stays still during lock, then moves at full speed
        float effective_movement_time = prediction_time;
        if (animation_lock_time > 0.f)
        {
            effective_movement_time = std::max(0.f, prediction_time - animation_lock_time);
            if (effective_movement_time <= 0.f)
                return position;  // Lock lasts longer than prediction time
        }

        // Use SDK velocity for accurate distance prediction
        // Accounts for actual movement state (accelerating, decelerating, turning)
        math::vector3 velocity = target->get_velocity();
        float actual_speed = velocity.magnitude();

        // Fallback to move_speed if velocity is zero (just started moving)
        if (actual_speed < 1.f)
            actual_speed = move_speed;

        float distance_to_travel = actual_speed * effective_movement_time;

        // End-of-Path Clamping (don't overshoot destination)
        float remaining_path = 0.f;
        for (size_t i = 1; i < path.size(); ++i)
        {
            math::vector3 seg_start = (i == 1) ? position : path[i - 1];
            math::vector3 seg_end = path[i];
            remaining_path += (seg_end - seg_start).magnitude();
        }

        if (distance_to_travel > remaining_path)
            distance_to_travel = remaining_path;

        // SIMPLE LINEAR PATH ITERATION
        float distance_traveled = 0.f;

        for (size_t i = 1; i < path.size(); ++i)
        {
            math::vector3 segment_start = (i == 1) ? position : path[i - 1];
            math::vector3 segment_end = path[i];
            math::vector3 segment_vec = segment_end - segment_start;
            float segment_length = segment_vec.magnitude();

            if (segment_length < 0.001f)
                continue;  // Skip zero-length segments

            // Check if prediction endpoint is on this segment
            if (distance_traveled + segment_length >= distance_to_travel)
            {
                // Found the segment - linear interpolate
                float distance_into_segment = distance_to_travel - distance_traveled;
                math::vector3 direction = segment_vec / segment_length;
                math::vector3 predicted_pos = segment_start + direction * distance_into_segment;

                // WALL COLLISION CHECK: Ensure predicted position is pathable
                if (g_sdk && g_sdk->nav_mesh && !g_sdk->nav_mesh->is_pathable(predicted_pos))
                {
                    return segment_start;  // Clamp to safe position
                }

                return predicted_pos;
            }

            distance_traveled += segment_length;
        }

        // Traveled entire path - return final waypoint
        return path.back();
    }

    // =========================================================================
    // MAIN PREDICTION FUNCTIONS (INTEGRATED WITH EDGE CASES)
    // =========================================================================

    /**
     * Helper: Convert HitChance enum to float for telemetry
     */
    inline float hit_chance_to_float(HitChance hc)
    {
        switch(hc)
        {
            case HitChance::Impossible:    return 0.0f;
            case HitChance::Clone:         return 0.0f;
            case HitChance::SpellShielded: return 0.0f;
            case HitChance::Windwalled:    return 0.0f;
            case HitChance::MinionBlocked: return 0.0f;
            case HitChance::Low:           return 0.35f;
            case HitChance::Medium:        return 0.55f;
            case HitChance::High:          return 0.75f;
            case HitChance::VeryHigh:      return 0.90f;
            case HitChance::Undodgeable:   return 1.0f;
            case HitChance::Immobile:      return 1.0f;
            case HitChance::Stasis:        return 1.0f;
            case HitChance::Channeling:    return 0.95f;
            case HitChance::Dashing:       return 0.80f;
            default:                       return 0.5f;
        }
    }

    /**
     * CONTINUOUS CONFIDENCE CALCULATION
     * Converts reaction window (time to dodge) to smooth 0-1 confidence
     *
     * This provides MUCH better granularity than discrete levels:
     * - 0.10s → 0.90 (very hard to dodge)
     * - 0.25s → 0.75 (medium-hard)
     * - 0.40s → 0.55 (medium)
     * - 0.50s → 0.35 (easy to dodge)
     *
     * Uses piecewise linear interpolation between thresholds for smooth gradient.
     */
    inline float reaction_window_to_confidence(float reaction_window)
    {
        // Negative or zero = impossible to dodge
        if (reaction_window <= 0.0f)
            return 1.0f;

        // Piecewise linear interpolation between threshold points
        // This gives smooth confidence instead of discrete steps

        // Range 1: 0.0s - 0.1s → confidence 1.0 - 0.90
        if (reaction_window <= REACTION_VERY_HIGH)  // 0.1s
        {
            float t = reaction_window / REACTION_VERY_HIGH;
            return 1.0f - (t * 0.10f);  // Interpolate from 1.0 to 0.90
        }

        // Range 2: 0.1s - 0.25s → confidence 0.90 - 0.75
        if (reaction_window <= REACTION_HIGH)  // 0.25s
        {
            float t = (reaction_window - REACTION_VERY_HIGH) / (REACTION_HIGH - REACTION_VERY_HIGH);
            return 0.90f - (t * 0.15f);  // Interpolate from 0.90 to 0.75
        }

        // Range 3: 0.25s - 0.4s → confidence 0.75 - 0.55
        if (reaction_window <= REACTION_MEDIUM)  // 0.4s
        {
            float t = (reaction_window - REACTION_HIGH) / (REACTION_MEDIUM - REACTION_HIGH);
            return 0.75f - (t * 0.20f);  // Interpolate from 0.75 to 0.55
        }

        // Range 4: 0.4s - 0.6s → confidence 0.55 - 0.35
        constexpr float LOW_THRESHOLD = 0.6f;
        if (reaction_window <= LOW_THRESHOLD)
        {
            float t = (reaction_window - REACTION_MEDIUM) / (LOW_THRESHOLD - REACTION_MEDIUM);
            return 0.55f - (t * 0.20f);  // Interpolate from 0.55 to 0.35
        }

        // Range 5: 0.6s+ → confidence 0.35 - 0.20 (floor at 0.20)
        // Very long reaction window = very easy to dodge
        constexpr float VERY_LOW_THRESHOLD = 1.0f;
        if (reaction_window <= VERY_LOW_THRESHOLD)
        {
            float t = (reaction_window - LOW_THRESHOLD) / (VERY_LOW_THRESHOLD - LOW_THRESHOLD);
            return 0.35f - (t * 0.15f);  // Interpolate from 0.35 to 0.20
        }

        // 1.0s+ reaction window = floor at 0.20 (always some chance of hitting distracted players)
        return 0.20f;
    }

    /**
     * THE DRIVER: Main entry point for prediction
     *
     * Fully integrated with EdgeCases, Settings, and Telemetry:
     * - Uses EdgeCases::analyze_target() for stasis, dash, windwall, minion, clone detection
     * - Respects PredictionSettings for user configuration
     * - Logs to PredictionTelemetry for performance tracking
     */
    inline PredictionResult get_prediction(const PredictionInput& input)
    {
        // Start timing for telemetry
        auto start_time = std::chrono::high_resolution_clock::now();

        PredictionResult result;

        // Validate input
        if (!input.source || !input.target)
        {
            result.block_reason = "Invalid source or target";
            result.hit_chance = HitChance::Impossible;
            result.should_cast = false;
            if (PredictionSettings::get().enable_telemetry)
            {
                PredictionTelemetry::TelemetryLogger::log_rejection_invalid_target();
            }
            return result;
        }

        // Store current positions for telemetry
        result.target_current_pos = input.target->get_server_position();
        result.distance_to_target = Utils::distance_2d(input.source->get_position(), result.target_current_pos);
        result.target_is_moving = input.target->is_moving();
        result.target_velocity = input.target->get_move_speed();

        // =======================================================================================
        // EDGE CASE ANALYSIS (Uses EdgeCaseDetection.h)
        // =======================================================================================
        auto edge_analysis = EdgeCases::analyze_target(input.target, input.source);

        // 1. CLONE DETECTION
        if (edge_analysis.is_clone)
        {
            result.block_reason = "Target is a clone";
            result.hit_chance = HitChance::Clone;
            result.should_cast = false;
            result.is_clone_target = true;
            return result;
        }

        // 2. SPELL SHIELD DETECTION
        // Report shield presence but let champion script decide whether to cast
        if (edge_analysis.has_shield)
        {
            result.spell_shield_detected = true;
            // Don't block - continue with normal prediction
        }

        // 3. UNTARGETABILITY DETECTION (Fizz E, Vlad W, Yi Q, etc.)
        // CRITICAL: Cannot hit untargetable targets - spell will fail
        if (edge_analysis.untargetability.is_untargetable)
        {
            result.block_reason = "Target is untargetable (" + edge_analysis.untargetability.ability_name + ")";
            result.hit_chance = HitChance::Impossible;
            result.should_cast = false;  // Spell will fail
            return result;
        }

        // 4. WINDWALL DETECTION
        if (edge_analysis.blocked_by_windwall)
        {
            result.block_reason = "Windwall blocks spell path";
            result.hit_chance = HitChance::Windwalled;
            result.should_cast = false;
            result.windwall_detected = true;
            return result;
        }

        // 5. STASIS HANDLING
        if (edge_analysis.stasis.is_in_stasis)
        {
            float current_time = g_sdk->clock_facade ? g_sdk->clock_facade->get_game_time() : 0.f;

            // Calculate spell travel time to stasis position
            float travel_time = Utils::compute_arrival_time(
                input.source->get_position(),
                edge_analysis.stasis.exit_position,
                input.missile_speed,
                input.cast_delay,
                input.proc_delay
            );

            // Calculate optimal timing
            float wait_time = EdgeCases::calculate_stasis_cast_timing(
                edge_analysis.stasis,
                travel_time,
                current_time
            );

            if (wait_time < 0.f)
            {
                result.block_reason = "Stasis timing impossible (spell too slow)";
                result.hit_chance = HitChance::Impossible;
                result.should_cast = false;
                return result;
            }

            // Stasis prediction
            result.cast_position = edge_analysis.stasis.exit_position;
            result.predicted_position = edge_analysis.stasis.exit_position;
            result.hit_chance = HitChance::Stasis;
            result.should_cast = (wait_time <= 0.f);  // Cast now or wait
            result.stasis_wait_time = wait_time;
            result.is_stasis = true;
            result.edge_case_type = "stasis";
            result.hit_chance_float = 1.0f;  // Guaranteed if timed correctly
            return result;
        }

        // 6. DASH HANDLING (if enabled)
        if (PredictionSettings::get().enable_dash_prediction && edge_analysis.dash.is_dashing)
        {
            float current_time = g_sdk->clock_facade ? g_sdk->clock_facade->get_game_time() : 0.f;

            // Predict to dash endpoint
            math::vector3 dash_end = edge_analysis.dash.dash_end_position;

            // Calculate spell arrival time to dash endpoint
            float spell_arrival = Utils::compute_arrival_time(
                input.source->get_position(),
                dash_end,
                input.missile_speed,
                input.cast_delay,
                input.proc_delay
            );

            // Every dash has an endpoint - predict to it regardless of timing
            // Champion script can decide if confidence is high enough to cast
            result.cast_position = dash_end;
            result.predicted_position = dash_end;
            result.hit_chance = HitChance::Dashing;
            result.should_cast = true;
            result.is_dash = true;
            result.dash_confidence = edge_analysis.dash.confidence_multiplier;
            result.edge_case_type = "dash";
            result.hit_chance_float = edge_analysis.dash.confidence_multiplier;
            return result;
        }

        // 7. CHANNEL/RECALL DETECTION
        if (edge_analysis.channel.is_channeling || edge_analysis.channel.is_recalling)
        {
            // High priority stationary target
            result.cast_position = edge_analysis.channel.position;
            result.predicted_position = edge_analysis.channel.position;
            result.hit_chance = HitChance::Channeling;
            result.should_cast = true;
            result.is_channeling = true;
            result.edge_case_type = "channeling";
            result.hit_chance_float = 0.95f;
            return result;
        }

        // 8. FORCED MOVEMENT DETECTION (Charm, Taunt, Fear)
        // These CCs force target to walk in predictable direction - very easy to hit
        if (edge_analysis.forced_movement.has_forced_movement)
        {
            // Calculate arrival time
            float time_to_impact = Utils::compute_arrival_time(
                input.source->get_position(),
                input.target->get_position(),
                input.missile_speed,
                input.cast_delay,
                input.proc_delay
            );

            // Predict forced movement
            float move_speed = input.target->get_move_speed();
            float movement_time = std::min(time_to_impact, edge_analysis.forced_movement.duration_remaining);
            float movement_distance = move_speed * movement_time;

            // Target walks in forced direction
            math::vector3 forced_pos = input.target->get_server_position() +
                edge_analysis.forced_movement.forced_direction * movement_distance;

            result.cast_position = forced_pos;
            result.predicted_position = forced_pos;
            result.hit_chance = HitChance::VeryHigh;  // Very predictable movement
            result.should_cast = true;
            result.hit_chance_float = 0.92f;  // Slightly lower than guaranteed (they might flash)

            if (edge_analysis.forced_movement.is_charm)
                result.edge_case_type = "charm";
            else if (edge_analysis.forced_movement.is_taunt)
                result.edge_case_type = "taunt";
            else if (edge_analysis.forced_movement.is_fear)
                result.edge_case_type = "fear";

            return result;
        }

        // =======================================================================================
        // NORMAL PREDICTION (Linear path following + geometric TTE)
        // =======================================================================================

        // Get animation lock time
        float animation_lock = Utils::get_animation_lock_time(input.target);
        result.is_animation_locked = (animation_lock > 0.f);

        // Predict position
        float prediction_time = Utils::compute_arrival_time(
            input.source->get_position(),
            input.target->get_position(),  // Initial guess
            input.missile_speed,
            input.cast_delay,
            input.proc_delay
        );

        math::vector3 predicted_pos = predict_linear_path(
            input.target,
            prediction_time,
            animation_lock
        );

        result.cast_position = predicted_pos;
        result.predicted_position = predicted_pos;
        result.prediction_offset = Utils::distance_2d(result.target_current_pos, predicted_pos);

        // Check if target is slowed or CSing (confidence boost)
        result.is_slowed = edge_analysis.is_slowed;
        result.is_csing = edge_analysis.is_csing;

        // =======================================================================================
        // VECTOR SPELL OPTIMIZATION (Viktor E, Rumble R, Taliyah W)
        // =======================================================================================
        math::vector3 spell_start_pos = input.source->get_position();  // Default: spell originates from champion

        if (input.shape == SpellShape::Vector)
        {
            // For vector spells, optimize the start position (first_cast_position)
            float max_cast_range = input.cast_range > 0.f ? input.cast_range : input.spell_range;

            // If first_cast_position not provided, calculate optimal placement
            if (Utils::magnitude_2d(input.first_cast_position) < EPSILON)
            {
                // Calculate direction from source to predicted target position
                math::vector3 source_to_target = predicted_pos - input.source->get_position();
                float dist_to_target = Utils::magnitude_2d(source_to_target);

                if (dist_to_target > EPSILON)
                {
                    // Normalize direction
                    math::vector3 direction = source_to_target / dist_to_target;

                    // Place start position as far forward as possible (within cast_range)
                    // This maximizes the chance of hitting the target
                    float forward_distance = std::min(max_cast_range, dist_to_target);
                    result.first_cast_position = input.source->get_position() + direction * forward_distance;
                }
                else
                {
                    // Target very close, place at source
                    result.first_cast_position = input.source->get_position();
                }
            }
            else
            {
                // Use provided first_cast_position, but clamp to max range
                math::vector3 to_first = input.first_cast_position - input.source->get_position();
                float dist_to_first = Utils::magnitude_2d(to_first);

                if (dist_to_first > max_cast_range)
                {
                    // Clamp to max cast range
                    math::vector3 direction = to_first / dist_to_first;
                    result.first_cast_position = input.source->get_position() + direction * max_cast_range;
                }
                else
                {
                    result.first_cast_position = input.first_cast_position;
                }
            }

            // Update spell start position for collision checks and escape distance
            spell_start_pos = result.first_cast_position;

            // Ensure the vector endpoint (predicted_pos) is within spell_range from first_cast
            math::vector3 vector_direction = predicted_pos - spell_start_pos;
            float vector_length = Utils::magnitude_2d(vector_direction);

            if (vector_length > input.spell_range)
            {
                // Clamp cast_position to spell_range from first_cast_position
                if (vector_length > EPSILON)
                {
                    vector_direction = vector_direction / vector_length;
                    result.cast_position = spell_start_pos + vector_direction * input.spell_range;
                }
            }
        }

        // =======================================================================================
        // MINION COLLISION CHECK (uses EdgeCaseDetection.h)
        // =======================================================================================
        // Check minion collision for line skillshots (Capsule, Cone, Vector)
        if (input.shape == SpellShape::Capsule ||
            input.shape == SpellShape::Line ||
            input.shape == SpellShape::Cone ||
            input.shape == SpellShape::Vector)
        {
            // Use EdgeCases minion collision with health prediction
            // For vector spells, use first_cast_position as origin
            float minion_clear = EdgeCases::compute_minion_block_probability(
                spell_start_pos,  // Vector uses first_cast_position, others use source position
                predicted_pos,
                input.spell_width,
                true  // This spell collides with minions
            );

            // Deterministic check: 0.0 = blocked, 1.0 = clear
            if (minion_clear == 0.0f)
            {
                result.block_reason = "Minion blocks spell path";
                result.hit_chance = HitChance::MinionBlocked;
                result.should_cast = false;
                result.minion_collision = true;

                if (PredictionSettings::get().enable_telemetry)
                {
                    PredictionTelemetry::TelemetryLogger::log_rejection_collision();
                }
                return result;
            }
        }

        // =======================================================================================
        // GEOMETRIC HIT CHANCE CALCULATION
        // =======================================================================================

        // Calculate time to impact
        // For Vector spells, missile travels from first_cast_position
        float time_to_impact = Utils::compute_arrival_time(
            spell_start_pos,  // Vector uses first_cast_position, others use source position
            predicted_pos,
            input.missile_speed,
            input.cast_delay,
            input.proc_delay
        );
        result.time_to_impact = time_to_impact;

        // Distance to exit calculation (shape-dependent)
        float target_radius = input.target->get_bounding_radius();
        float distance_to_exit = 0.f;

        if (input.shape == SpellShape::Circle)
        {
            distance_to_exit = Utils::calculate_escape_distance_circle(
                predicted_pos,
                result.cast_position,
                input.spell_width,
                target_radius
            );
        }
        else if (input.shape == SpellShape::Cone)
        {
            // Cone: spell_width stores cone angle in radians
            math::vector3 spell_direction = (predicted_pos - input.source->get_position());
            float spell_length = Utils::magnitude_2d(spell_direction);
            if (spell_length > EPSILON)
                spell_direction = spell_direction / spell_length;
            else
                spell_direction = math::vector3(0.f, 0.f, 1.f);

            float cone_half_angle = input.spell_width / 2.f;  // Convert full angle to half angle
            distance_to_exit = Utils::calculate_escape_distance_cone(
                predicted_pos,
                input.source->get_position(),
                spell_direction,
                cone_half_angle,
                input.spell_range,
                target_radius
            );
        }
        else  // Capsule, Line, Vector
        {
            // All treated as linear skillshots for single-target escape distance
            // For Vector, spell_start_pos is first_cast_position (calculated above)
            math::vector3 spell_direction = (predicted_pos - spell_start_pos);
            float spell_length = Utils::magnitude_2d(spell_direction);
            if (spell_length > EPSILON)
                spell_direction = spell_direction / spell_length;
            else
                spell_direction = math::vector3(0.f, 0.f, 1.f);

            distance_to_exit = Utils::calculate_escape_distance_capsule(
                predicted_pos,
                spell_start_pos,  // Vector uses first_cast_position, others use source position
                spell_direction,
                input.spell_width,
                input.spell_range,
                target_radius
            );
        }
        result.distance_to_exit = distance_to_exit;

        // Time needed to dodge
        float move_speed = input.target->get_move_speed();
        if (move_speed <= 0.0f)
        {
            result.hit_chance = HitChance::Immobile;
            result.should_cast = true;
            result.is_immobile = true;
            result.hit_chance_float = 1.0f;
            return result;
        }

        float time_needed_to_dodge = animation_lock + (distance_to_exit / move_speed);

        // FOW visibility adjustment (uses EdgeCases logic)
        float effective_reaction_reduction = 0.f;
        if (input.source && g_sdk && g_sdk->nav_mesh)
        {
            int enemy_team = input.target->get_team_id();
            bool hidden = g_sdk->nav_mesh->is_in_fow_for_team(
                input.source->get_position(), enemy_team);

            if (!hidden)
            {
                effective_reaction_reduction = input.cast_delay;
            }
        }

        // Reaction window calculation
        float reaction_window = time_to_impact - time_needed_to_dodge - effective_reaction_reduction;
        result.reaction_window = reaction_window;

        // Terrain blocking multiplier
        float terrain_multiplier = Utils::get_terrain_blocking_multiplier(
            predicted_pos,
            result.cast_position,
            distance_to_exit
        );

        // Trapped = forced Undodgeable
        if (terrain_multiplier >= 1.9f)
        {
            result.hit_chance = HitChance::Undodgeable;
            result.should_cast = true;
            result.hit_chance_float = 1.0f;
            return result;
        }

        // Apply terrain and behavioral multipliers
        reaction_window /= terrain_multiplier;

        // NOTE: Slows are FULLY captured by move_speed in time_needed_to_dodge calculation
        // Example: 25% slow → 34% more dodge time → 57% less reaction window (geometric amplification)
        // NO additional multiplier needed - the physics naturally makes slows very effective

        // CSing: Attention factor (NOT captured in physics)
        if (result.is_csing)
        {
            reaction_window /= 1.08f;  // Modest boost for distracted targets (CSing)
        }

        // Grade reaction window to hit chance (discrete enum for backwards compatibility)
        if (reaction_window <= REACTION_UNDODGEABLE)
            result.hit_chance = HitChance::Undodgeable;
        else if (reaction_window <= REACTION_VERY_HIGH)
            result.hit_chance = HitChance::VeryHigh;
        else if (reaction_window <= REACTION_HIGH)
            result.hit_chance = HitChance::High;
        else if (reaction_window <= REACTION_MEDIUM)
            result.hit_chance = HitChance::Medium;
        else
            result.hit_chance = HitChance::Low;

        // CONTINUOUS CONFIDENCE: Use smooth interpolation instead of discrete levels
        // This provides much better granularity (0.39s vs 0.41s are now distinguishable)
        result.hit_chance_float = reaction_window_to_confidence(reaction_window);

        // Champion script controls casting decision based on user-configured thresholds
        // Typical thresholds: Medium = 50%+, High = 75%+, VeryHigh = 90%+
        // System provides hit_chance and hit_chance_float - script decides when to cast
        result.should_cast = true;  // Let champion script make the final decision

        // =======================================================================================
        // TELEMETRY LOGGING
        // =======================================================================================
        auto end_time = std::chrono::high_resolution_clock::now();
        result.computation_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

        if (PredictionSettings::get().enable_telemetry)
        {
            PredictionTelemetry::PredictionEvent event;
            event.timestamp = g_sdk->clock_facade ? g_sdk->clock_facade->get_game_time() : 0.f;
            event.target_name = input.target->get_char_name();

            // Log spell shape type
            if (input.shape == SpellShape::Circle)
                event.spell_type = "Circle";
            else if (input.shape == SpellShape::Cone)
                event.spell_type = "Cone";
            else if (input.shape == SpellShape::Vector)
                event.spell_type = "Vector";
            else  // Capsule or Line
                event.spell_type = "Capsule";
            event.hit_chance = result.hit_chance_float;
            event.confidence = result.hit_chance_float;  // Geometric has no separate confidence
            event.distance = result.distance_to_target;
            event.was_dash = result.is_dash;
            event.was_stationary = !result.target_is_moving;
            event.was_animation_locked = result.is_animation_locked;
            event.collision_detected = result.minion_collision || result.windwall_detected;
            event.computation_time_ms = result.computation_time_ms;
            event.edge_case = result.edge_case_type;

            // Spell configuration
            event.spell_range = input.spell_range;
            event.spell_radius = input.spell_width;
            event.spell_delay = input.cast_delay;
            event.spell_speed = input.missile_speed;

            // Movement data
            event.target_velocity = result.target_velocity;
            event.prediction_offset = result.prediction_offset;
            event.target_is_moving = result.target_is_moving;

            // Position data
            event.source_pos_x = input.source->get_position().x;
            event.source_pos_z = input.source->get_position().z;
            event.target_server_pos_x = result.target_current_pos.x;
            event.target_server_pos_z = result.target_current_pos.z;
            event.predicted_pos_x = result.predicted_position.x;
            event.predicted_pos_z = result.predicted_position.z;
            event.cast_pos_x = result.cast_position.x;
            event.cast_pos_z = result.cast_position.z;

            // Timing data
            event.final_arrival_time = result.time_to_impact;
            event.dodge_time = result.reaction_window;
            event.effective_move_speed = move_speed;

            PredictionTelemetry::TelemetryLogger::log_prediction(event);
        }

        return result;
    }

} // namespace GeometricPred

    // =========================================================================
    // MULTI-TARGET AOE PREDICTION
    // =========================================================================

    /**
     * Multi-target AOE prediction result
     * Contains cast position and list of targets that will be hit
     */
    struct AOEPredictionResult
    {
        math::vector3 cast_position;                  // Optimal cast location
        std::vector<game_object*> hit_targets;        // Targets that will be hit
        std::vector<float> individual_hit_chances;    // Hit chance for each target
        float expected_hits;                          // Sum of hit chances (2.7 = likely 2-3 hits)
        float min_hit_chance;                         // Weakest link
        float avg_hit_chance;                         // Average confidence
        bool is_valid;                                // Whether this is a good cast

        AOEPredictionResult()
            : expected_hits(0.f), min_hit_chance(0.f),
              avg_hit_chance(0.f), is_valid(false)
        {}
    };

    /**
     * MULTI-TARGET AOE PREDICTION
     *
     * Find optimal cast position for circular AOE spells (Brand R, Orianna R, Annie R)
     * Returns position that maximizes expected hits × average hit chance
     *
     * Algorithm:
     * 1. Get individual predictions for all enemies in range
     * 2. Filter by minimum hit chance threshold
     * 3. Find centroid (center of mass) of predicted positions
     * 4. Clamp centroid to spell range
     * 5. Calculate expected hits at that position
     *
     * Example:
     *   - 3 enemies with 0.8, 0.7, 0.6 hit chance
     *   - Expected hits = 2.1 (likely 2 hits, possibly 3)
     *   - Avg confidence = 0.70
     */
    inline AOEPredictionResult predict_aoe_circle(
        game_object* source,
        float spell_radius,
        float spell_range,
        float missile_speed,
        float cast_delay,
        float proc_delay = 0.f,
        float min_targets = 2.0f,        // Minimum expected hits to be valid
        float min_individual_hc = 0.3f)  // Filter out very low confidence targets
    {
        AOEPredictionResult result;

        if (!source || !g_sdk || !g_sdk->object_manager)
        {
            result.is_valid = false;
            return result;
        }

        // 1. Get all valid enemy champions in range
        std::vector<game_object*> candidates;
        std::vector<math::vector3> predicted_positions;
        std::vector<float> hit_chances;

        // Calculate movement buffer based on spell parameters
        // Buffer = max_move_speed * (cast_delay + max_travel_time)
        float max_travel_time = (missile_speed > 0.f) ? (spell_range / missile_speed) : 0.f;
        float movement_buffer = AOE_MAX_MOVE_SPEED * (cast_delay + max_travel_time + proc_delay);

        auto enemies = g_sdk->object_manager->get_enemy_heroes();
        for (auto* enemy : enemies)
        {
            if (!enemy || !enemy->is_valid() || enemy->is_dead())
                continue;

            // Skip enemies out of range (accounting for movement during cast + travel)
            float dist = Utils::distance_2d(source->get_position(), enemy->get_position());
            if (dist > spell_range + movement_buffer)
                continue;

            // Get individual prediction
            PredictionInput input;
            input.source = source;
            input.target = enemy;
            input.shape = SpellShape::Circle;
            input.spell_width = spell_radius;
            input.spell_range = spell_range;
            input.missile_speed = missile_speed;
            input.cast_delay = cast_delay;
            input.proc_delay = proc_delay;

            auto pred = get_prediction(input);

            // Filter by minimum hit chance (exclude impossible/blocked/very low)
            if (pred.hit_chance_float >= min_individual_hc &&
                pred.hit_chance != HitChance::Impossible &&
                pred.hit_chance != HitChance::Clone &&
                pred.hit_chance != HitChance::SpellShielded)
            {
                candidates.push_back(enemy);
                predicted_positions.push_back(pred.predicted_position);
                hit_chances.push_back(pred.hit_chance_float);
            }
        }

        // Check if we have enough candidates
        if (candidates.empty())
        {
            result.is_valid = false;
            return result;
        }

        // 2. Find centroid (center of mass) of predicted positions
        // This is a fast approximation of minimum enclosing circle
        math::vector3 centroid{};
        for (const auto& pos : predicted_positions)
        {
            centroid.x += pos.x;
            centroid.y += pos.y;
            centroid.z += pos.z;
        }
        centroid.x /= predicted_positions.size();
        centroid.y /= predicted_positions.size();
        centroid.z /= predicted_positions.size();

        // 3. Clamp centroid to spell range
        math::vector3 source_pos = source->get_position();
        math::vector3 to_centroid = centroid - source_pos;
        float dist_to_centroid = Utils::magnitude_2d(to_centroid);

        // CRITICAL FIX: Prevent division by zero if centroid == source
        if (dist_to_centroid < EPSILON)
        {
            // All enemies at source position - cast at max range in arbitrary direction
            centroid = source_pos + math::vector3(spell_range, 0.f, 0.f);
        }
        else if (dist_to_centroid > spell_range)
        {
            // Normalize and scale to max range
            to_centroid = to_centroid / dist_to_centroid;  // Safe - dist_to_centroid > 0
            centroid = source_pos + to_centroid * spell_range;
        }

        // 4. Calculate expected hits at centroid position
        float expected = 0.f;
        float min_hc = 1.0f;
        float sum_hc = 0.f;
        int hits = 0;

        for (size_t i = 0; i < predicted_positions.size(); ++i)
        {
            // Check if predicted position is within AOE radius
            float dist_to_center = Utils::distance_2d(predicted_positions[i], centroid);
            float target_radius = candidates[i]->get_bounding_radius();

            // Target is hit if their center is within spell_radius + their_hitbox
            if (dist_to_center <= spell_radius + target_radius)
            {
                result.hit_targets.push_back(candidates[i]);
                result.individual_hit_chances.push_back(hit_chances[i]);
                expected += hit_chances[i];
                sum_hc += hit_chances[i];
                min_hc = std::min(min_hc, hit_chances[i]);
                hits++;
            }
        }

        // 5. Populate result
        result.cast_position = centroid;
        result.expected_hits = expected;
        result.min_hit_chance = min_hc;
        result.avg_hit_chance = hits > 0 ? sum_hc / hits : 0.f;
        result.is_valid = (expected >= min_targets);

        return result;
    }

    /**
     * LINEAR/CAPSULE AOE PREDICTION
     *
     * Find optimal cast direction for line AOE spells (Vel'Koz W, Ezreal R, MF R, Lux R)
     * A capsule is a rectangle with semicircular ends (line segment swept by a circle)
     *
     * Algorithm:
     * 1. Get individual predictions for all enemies in range
     * 2. Try multiple cast angles (e.g., every 15 degrees)
     * 3. For each angle, calculate expected hits by checking if predicted positions intersect the line
     * 4. Return angle with maximum expected hits
     *
     * Parameters:
     *   - spell_width: Radius of the line (half-width)
     *   - spell_range: Length of the line
     */
    inline AOEPredictionResult predict_aoe_linear(
        game_object* source,
        float spell_width,
        float spell_range,
        float missile_speed,
        float cast_delay,
        float proc_delay = 0.f,
        float min_targets = 2.0f,
        float min_individual_hc = 0.3f,
        int angle_samples = 24)  // Test every 15 degrees (360/24)
    {
        AOEPredictionResult result;

        if (!source || !g_sdk || !g_sdk->object_manager)
        {
            result.is_valid = false;
            return result;
        }

        // 1. Get all valid enemy champions in range
        std::vector<game_object*> candidates;
        std::vector<math::vector3> predicted_positions;
        std::vector<float> hit_chances;

        // Calculate movement buffer based on spell parameters
        float max_travel_time = (missile_speed > 0.f) ? (spell_range / missile_speed) : 0.f;
        float movement_buffer = AOE_MAX_MOVE_SPEED * (cast_delay + max_travel_time + proc_delay);

        auto enemies = g_sdk->object_manager->get_enemy_heroes();
        for (auto* enemy : enemies)
        {
            if (!enemy || !enemy->is_valid() || enemy->is_dead())
                continue;

            // Skip enemies out of range (accounting for movement during cast + travel)
            float dist = Utils::distance_2d(source->get_position(), enemy->get_position());
            if (dist > spell_range + spell_width + movement_buffer)
                continue;

            // Get individual prediction
            PredictionInput input;
            input.source = source;
            input.target = enemy;
            input.shape = SpellShape::Line;
            input.spell_width = spell_width;
            input.spell_range = spell_range;
            input.missile_speed = missile_speed;
            input.cast_delay = cast_delay;
            input.proc_delay = proc_delay;

            auto pred = get_prediction(input);

            // Filter by minimum hit chance
            if (pred.hit_chance_float >= min_individual_hc &&
                pred.hit_chance != HitChance::Impossible &&
                pred.hit_chance != HitChance::Clone &&
                pred.hit_chance != HitChance::SpellShielded)
            {
                candidates.push_back(enemy);
                predicted_positions.push_back(pred.predicted_position);
                hit_chances.push_back(pred.hit_chance_float);
            }
        }

        if (candidates.empty())
        {
            result.is_valid = false;
            return result;
        }

        // 2. Try multiple cast angles and find best
        math::vector3 source_pos = source->get_position();
        float best_expected = 0.f;
        math::vector3 best_cast_pos{};
        std::vector<int> best_hit_indices;

        constexpr float PI = 3.14159265359f;
        float angle_step = (2.f * PI) / angle_samples;

        for (int angle_idx = 0; angle_idx < angle_samples; ++angle_idx)
        {
            float angle = angle_idx * angle_step;
            math::vector3 direction(std::cos(angle), 0.f, std::sin(angle));

            // Cast endpoint
            math::vector3 line_end = source_pos + direction * spell_range;

            // Check how many targets this angle would hit
            float expected = 0.f;
            std::vector<int> hit_indices;

            for (size_t i = 0; i < predicted_positions.size(); ++i)
            {
                // Check if predicted position intersects the capsule
                // Capsule = line segment + circular ends + thickness

                math::vector3 to_target = predicted_positions[i] - source_pos;
                float proj = to_target.dot(direction);  // Projection onto line direction

                // Clamp projection to [0, spell_range]
                proj = std::max(0.f, std::min(spell_range, proj));

                // Closest point on line segment to target
                math::vector3 closest_point = source_pos + direction * proj;
                float dist = Utils::distance_2d(predicted_positions[i], closest_point);
                float target_radius = candidates[i]->get_bounding_radius();

                // Hit if within width + target hitbox
                if (dist <= spell_width + target_radius)
                {
                    expected += hit_chances[i];
                    hit_indices.push_back(static_cast<int>(i));
                }
            }

            // Track best angle
            if (expected > best_expected)
            {
                best_expected = expected;
                best_cast_pos = line_end;
                best_hit_indices = hit_indices;
            }
        }

        // 3. Populate result with best angle
        if (best_hit_indices.empty())
        {
            result.is_valid = false;
            return result;
        }

        float sum_hc = 0.f;
        float min_hc = 1.0f;
        for (int idx : best_hit_indices)
        {
            result.hit_targets.push_back(candidates[idx]);
            result.individual_hit_chances.push_back(hit_chances[idx]);
            sum_hc += hit_chances[idx];
            min_hc = std::min(min_hc, hit_chances[idx]);
        }

        result.cast_position = best_cast_pos;
        result.expected_hits = best_expected;
        result.min_hit_chance = min_hc;
        result.avg_hit_chance = sum_hc / best_hit_indices.size();
        result.is_valid = (best_expected >= min_targets);

        return result;
    }

    /**
     * CONE AOE PREDICTION
     *
     * Find optimal cast direction for cone AOE spells (Annie W, Cassiopeia Q, Rumble E)
     * A cone is a wedge-shaped area from source
     *
     * Algorithm:
     * 1. Get individual predictions for all enemies in range
     * 2. Try multiple cast angles
     * 3. For each angle, check if predicted positions fall within the cone
     * 4. Return angle with maximum expected hits
     *
     * Parameters:
     *   - spell_width: Cone angle in radians (e.g., 50 degrees = 0.873 radians)
     *   - spell_range: Cone length
     */
    inline AOEPredictionResult predict_aoe_cone(
        game_object* source,
        float cone_angle_radians,
        float spell_range,
        float missile_speed,
        float cast_delay,
        float proc_delay = 0.f,
        float min_targets = 2.0f,
        float min_individual_hc = 0.3f,
        int angle_samples = 24)
    {
        AOEPredictionResult result;

        if (!source || !g_sdk || !g_sdk->object_manager)
        {
            result.is_valid = false;
            return result;
        }

        // 1. Get all valid enemy champions in range
        std::vector<game_object*> candidates;
        std::vector<math::vector3> predicted_positions;
        std::vector<float> hit_chances;

        // Calculate movement buffer based on spell parameters
        float max_travel_time = (missile_speed > 0.f) ? (spell_range / missile_speed) : 0.f;
        float movement_buffer = AOE_MAX_MOVE_SPEED * (cast_delay + max_travel_time + proc_delay);

        auto enemies = g_sdk->object_manager->get_enemy_heroes();
        for (auto* enemy : enemies)
        {
            if (!enemy || !enemy->is_valid() || enemy->is_dead())
                continue;

            float dist = Utils::distance_2d(source->get_position(), enemy->get_position());
            if (dist > spell_range + movement_buffer)
                continue;

            PredictionInput input;
            input.source = source;
            input.target = enemy;
            input.shape = SpellShape::Cone;
            input.spell_width = cone_angle_radians;
            input.spell_range = spell_range;
            input.missile_speed = missile_speed;
            input.cast_delay = cast_delay;
            input.proc_delay = proc_delay;

            auto pred = get_prediction(input);

            if (pred.hit_chance_float >= min_individual_hc &&
                pred.hit_chance != HitChance::Impossible &&
                pred.hit_chance != HitChance::Clone &&
                pred.hit_chance != HitChance::SpellShielded)
            {
                candidates.push_back(enemy);
                predicted_positions.push_back(pred.predicted_position);
                hit_chances.push_back(pred.hit_chance_float);
            }
        }

        if (candidates.empty())
        {
            result.is_valid = false;
            return result;
        }

        // 2. Try multiple cast angles
        math::vector3 source_pos = source->get_position();
        float best_expected = 0.f;
        math::vector3 best_cast_pos{};
        std::vector<int> best_hit_indices;

        constexpr float PI = 3.14159265359f;
        float angle_step = (2.f * PI) / angle_samples;
        float half_cone = cone_angle_radians / 2.f;

        for (int angle_idx = 0; angle_idx < angle_samples; ++angle_idx)
        {
            float center_angle = angle_idx * angle_step;
            math::vector3 direction(std::cos(center_angle), 0.f, std::sin(center_angle));

            math::vector3 cone_end = source_pos + direction * spell_range;

            float expected = 0.f;
            std::vector<int> hit_indices;

            for (size_t i = 0; i < predicted_positions.size(); ++i)
            {
                // Check if target is within cone
                math::vector3 to_target = predicted_positions[i] - source_pos;
                float dist = to_target.magnitude();

                if (dist < EPSILON)
                    continue;

                // Check range
                if (dist > spell_range)
                    continue;

                // Calculate angle between center direction and target direction
                math::vector3 target_dir = to_target / dist;
                float dot = direction.dot(target_dir);
                float angle_diff = std::acos(std::max(-1.f, std::min(1.f, dot)));

                // Check if within cone angle (with target radius buffer)
                float target_radius = candidates[i]->get_bounding_radius();
                float angular_tolerance = std::atan2(target_radius, dist);

                if (angle_diff <= half_cone + angular_tolerance)
                {
                    expected += hit_chances[i];
                    hit_indices.push_back(static_cast<int>(i));
                }
            }

            if (expected > best_expected)
            {
                best_expected = expected;
                best_cast_pos = cone_end;
                best_hit_indices = hit_indices;
            }
        }

        // 3. Populate result
        if (best_hit_indices.empty())
        {
            result.is_valid = false;
            return result;
        }

        float sum_hc = 0.f;
        float min_hc = 1.0f;
        for (int idx : best_hit_indices)
        {
            result.hit_targets.push_back(candidates[idx]);
            result.individual_hit_chances.push_back(hit_chances[idx]);
            sum_hc += hit_chances[idx];
            min_hc = std::min(min_hc, hit_chances[idx]);
        }

        result.cast_position = best_cast_pos;
        result.expected_hits = best_expected;
        result.min_hit_chance = min_hc;
        result.avg_hit_chance = sum_hc / best_hit_indices.size();
        result.is_valid = (best_expected >= min_targets);

        return result;
    }

    /**
     * VECTOR AOE PREDICTION
     *
     * Find optimal cast for vector spells (Viktor E, Rumble R, Taliyah W)
     * Vector spells require TWO positions: start and end
     *
     * Algorithm:
     * 1. Get individual predictions
     * 2. Try different line segments (varying start + end positions)
     * 3. Find line that hits most targets
     *
     * This is more complex - simplified version just finds best line through targets
     */
    inline AOEPredictionResult predict_aoe_vector(
        game_object* source,
        float spell_width,
        float max_range,
        float line_length,
        float missile_speed,
        float cast_delay,
        float proc_delay = 0.f,
        float min_targets = 2.0f,
        float min_individual_hc = 0.3f)
    {
        AOEPredictionResult result;

        if (!source || !g_sdk || !g_sdk->object_manager)
        {
            result.is_valid = false;
            return result;
        }

        // 1. Get all valid enemy champions
        std::vector<game_object*> candidates;
        std::vector<math::vector3> predicted_positions;
        std::vector<float> hit_chances;

        // Calculate movement buffer based on spell parameters
        float max_travel_time = (missile_speed > 0.f) ? ((max_range + line_length) / missile_speed) : 0.f;
        float movement_buffer = AOE_MAX_MOVE_SPEED * (cast_delay + max_travel_time + proc_delay);

        auto enemies = g_sdk->object_manager->get_enemy_heroes();
        for (auto* enemy : enemies)
        {
            if (!enemy || !enemy->is_valid() || enemy->is_dead())
                continue;

            float dist = Utils::distance_2d(source->get_position(), enemy->get_position());
            if (dist > max_range + line_length + movement_buffer)
                continue;

            PredictionInput input;
            input.source = source;
            input.target = enemy;
            input.shape = SpellShape::Line;  // Similar to line
            input.spell_width = spell_width;
            input.spell_range = max_range;
            input.missile_speed = missile_speed;
            input.cast_delay = cast_delay;
            input.proc_delay = proc_delay;

            auto pred = get_prediction(input);

            if (pred.hit_chance_float >= min_individual_hc &&
                pred.hit_chance != HitChance::Impossible &&
                pred.hit_chance != HitChance::Clone &&
                pred.hit_chance != HitChance::SpellShielded)
            {
                candidates.push_back(enemy);
                predicted_positions.push_back(pred.predicted_position);
                hit_chances.push_back(pred.hit_chance_float);
            }
        }

        if (candidates.empty())
        {
            result.is_valid = false;
            return result;
        }

        // 2. Simplified: Try line segments starting from various positions
        // For full implementation: Would try all combinations of start/end positions
        // Here: Try lines at different angles from near the centroid

        math::vector3 source_pos = source->get_position();
        float best_expected = 0.f;
        math::vector3 best_start{};
        math::vector3 best_end{};
        std::vector<int> best_hit_indices;

        // Calculate rough centroid to orient lines toward targets
        math::vector3 centroid{};
        for (const auto& pos : predicted_positions)
        {
            centroid.x += pos.x;
            centroid.z += pos.z;
        }
        centroid.x /= predicted_positions.size();
        centroid.z /= predicted_positions.size();
        centroid.y = source_pos.y;

        // Try multiple angles
        constexpr float PI = 3.14159265359f;
        int angle_samples = 16;
        float angle_step = (2.f * PI) / angle_samples;

        for (int angle_idx = 0; angle_idx < angle_samples; ++angle_idx)
        {
            float angle = angle_idx * angle_step;
            math::vector3 direction(std::cos(angle), 0.f, std::sin(angle));

            // Line starts near centroid, extends in direction
            math::vector3 line_start = centroid;
            math::vector3 line_end = centroid + direction * line_length;

            // Clamp to max range
            if (Utils::distance_2d(source_pos, line_start) > max_range)
                continue;

            float expected = 0.f;
            std::vector<int> hit_indices;

            for (size_t i = 0; i < predicted_positions.size(); ++i)
            {
                // Check if target intersects the line segment
                math::vector3 to_start = predicted_positions[i] - line_start;
                math::vector3 line_vec = line_end - line_start;
                float line_len = line_vec.magnitude();

                if (line_len < EPSILON)
                    continue;

                math::vector3 line_dir = line_vec / line_len;
                float proj = to_start.dot(line_dir);
                proj = std::max(0.f, std::min(line_len, proj));

                math::vector3 closest = line_start + line_dir * proj;
                float dist = Utils::distance_2d(predicted_positions[i], closest);
                float target_radius = candidates[i]->get_bounding_radius();

                if (dist <= spell_width + target_radius)
                {
                    expected += hit_chances[i];
                    hit_indices.push_back(static_cast<int>(i));
                }
            }

            if (expected > best_expected)
            {
                best_expected = expected;
                best_start = line_start;
                best_end = line_end;
                best_hit_indices = hit_indices;
            }
        }

        // 3. Populate result
        if (best_hit_indices.empty())
        {
            result.is_valid = false;
            return result;
        }

        float sum_hc = 0.f;
        float min_hc = 1.0f;
        for (int idx : best_hit_indices)
        {
            result.hit_targets.push_back(candidates[idx]);
            result.individual_hit_chances.push_back(hit_chances[idx]);
            sum_hc += hit_chances[idx];
            min_hc = std::min(min_hc, hit_chances[idx]);
        }

        // For vector spells, cast_position represents the END point
        // Caller would need to know the START point separately
        result.cast_position = best_end;
        result.expected_hits = best_expected;
        result.min_hit_chance = min_hc;
        result.avg_hit_chance = sum_hc / best_hit_indices.size();
        result.is_valid = (best_expected >= min_targets);

        return result;
    }

} // namespace GeometricPred
