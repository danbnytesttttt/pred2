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
 * Replaces the 4000-line hybrid system with ~500 lines of focused logic.
 *
 * Core Components:
 * 1. predict_linear_path()  - The Engine (where will they be?)
 * 2. calculate_hitchance()  - The Brain  (should we cast?)
 * 3. get_prediction()       - The Driver (main entry point)
 * ============================================================================
 */

#include "sdk.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

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

    // Path prediction heuristics
    constexpr float PATH_START_DAMPENING = 0.85f;  // Slow start (acceleration phase)
    constexpr float PATH_START_DURATION = 0.1f;    // 100ms ramp to full speed
    constexpr float PATH_STALE_THRESHOLD = 0.2f;   // Animation locks > 200ms make paths stale
    constexpr float PATH_STALE_RANGE = 0.4f;       // Staleness ramp duration
    constexpr float PATH_STALE_MAX_REDUCTION = 0.5f; // Max 50% distance reduction

    // Minion collision constants
    constexpr float MINION_SEARCH_RADIUS = 150.f;  // Search radius around spell path
    constexpr float MINION_HITBOX_RADIUS = 65.f;   // Average minion collision radius

    // =========================================================================
    // ENUMS
    // =========================================================================

    /**
     * Spell collision shapes
     * Circle: AoE explosions (Annie R, Lux E, Ziggs Q/W/E/R, Orianna R)
     * Capsule: Linear skillshots (Morgana Q, Blitz Hook, Xerath E, Lux Q)
     *          Also covers "rectangles" (Xerath Q = capsule with length)
     */
    enum class SpellShape
    {
        Circle,   // Point-target AoE
        Capsule   // Line + width (missiles, most skillshots)
    };

    /**
     * Graded hit chance levels
     * Maps reaction_window to confidence levels
     */
    enum class HitChance
    {
        Impossible,     // Target is dead, invulnerable, out of range, or spell blocked
        SpellShielded,  // Target has spell shield (Sivir E, Banshee's, Malz passive)
        Low,            // >400ms reaction window - easily dodgeable
        Medium,         // 250-400ms window - requires attention
        High,           // 100-250ms window - difficult to dodge
        VeryHigh,       // <100ms window - requires quick reaction
        Undodgeable,    // Physically impossible to escape or no time to react
        Immobile        // CC'd and cannot move
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
        float spell_range;          // Spell max range (for capsules)
        float missile_speed;        // Projectile speed (0 = instant)
        float cast_delay;           // Windup/cast time before spell launches
        float proc_delay;           // Additional delay before damage (e.g., Syndra Q = 0.6s)

        PredictionInput()
            : source(nullptr), target(nullptr), shape(SpellShape::Capsule),
              spell_width(70.f), spell_range(1000.f), missile_speed(1500.f),
              cast_delay(0.25f), proc_delay(0.f)
        {}
    };

    /**
     * Prediction result with cast position and confidence
     */
    struct PredictionResult
    {
        math::vector3 cast_position;  // Where to aim
        HitChance hit_chance;         // Confidence level
        bool should_cast;             // Simple yes/no recommendation

        // Debug information
        float reaction_window;        // Time enemy has to dodge (seconds)
        float time_to_impact;         // Spell arrival time (seconds)
        float distance_to_exit;       // Distance enemy must travel to escape
        const char* block_reason;     // Why we can't cast (if Impossible)

        PredictionResult()
            : cast_position{}, hit_chance(HitChance::Impossible), should_cast(false),
              reaction_window(0.f), time_to_impact(0.f), distance_to_exit(0.f),
              block_reason(nullptr)
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

                // Check collision: spell radius + minion hitbox
                float collision_radius = spell_width * 0.5f + MINION_HITBOX_RADIUS;
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

                // Backward escape distance (off the back of spell)
                float backward_escape = proj_length + target_radius;

                // Return minimum (easiest escape path)
                return std::max(0.f, std::min(lateral_escape, backward_escape));
            }
            else
            {
                // Target is outside spell - already safe
                return 0.f;
            }
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

        // CC CHECK: Immobilized targets can't move
        if (target->has_buff_of_type(buff_type::stun) ||
            target->has_buff_of_type(buff_type::snare) ||
            target->has_buff_of_type(buff_type::charm) ||
            target->has_buff_of_type(buff_type::fear) ||
            target->has_buff_of_type(buff_type::taunt) ||
            target->has_buff_of_type(buff_type::suppression) ||
            target->has_buff_of_type(buff_type::knockup) ||
            target->has_buff_of_type(buff_type::asleep))
        {
            return position;  // CC'd - stay at current position
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

            // PATH STALENESS: Long locks make paths unreliable
            if (animation_lock_time > PATH_STALE_THRESHOLD)
            {
                float staleness_factor = 1.0f - std::min(
                    (animation_lock_time - PATH_STALE_THRESHOLD) / PATH_STALE_RANGE,
                    PATH_STALE_MAX_REDUCTION);
                effective_movement_time *= staleness_factor;
            }
        }

        // HEURISTIC 1: Start-of-Path Dampening (smooth acceleration ramp)
        // League accelerates to max speed in ~23ms, but we use 100ms conservative window
        // TODO: Get path age from a tracker if available (for now assume fresh path)
        float speed_multiplier = 1.0f;
        // For now, skip dampening unless we track path age
        // Can add later if needed: speed_multiplier = 0.85f + 0.15f * (path_age / 0.1f);

        float distance_to_travel = move_speed * effective_movement_time * speed_multiplier;

        // HEURISTIC 2: End-of-Path Clamping (don't overshoot destination)
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

    /**
     * THE BRAIN: Calculate hit chance using geometric Time-To-Exit (TTE)
     *
     * Core question: "Can they physically escape the hitbox in time?"
     *
     * Algorithm:
     * 1. Calculate time_to_impact (when spell arrives)
     * 2. Calculate distance_to_exit (how far they must run to safety)
     * 3. Calculate time_needed_to_dodge = animation_lock + (distance / speed)
     * 4. Calculate reaction_window = time_to_impact - time_needed_to_dodge
     * 5. Grade reaction_window → confidence level
     *
     * Environmental adjustments:
     * - FOW visibility (affects reaction time)
     * - Terrain blocking (trapped = higher chance)
     * - Minion collision (blocked = Impossible)
     */
    inline HitChance calculate_hitchance(
        const PredictionInput& input,
        const math::vector3& predicted_position,
        PredictionResult& result)  // Pass result to fill debug data
    {
        game_object* target = input.target;
        game_object* source = input.source;

        // 1. SANITY CHECKS
        if (!target || !target->is_valid() || target->is_dead())
        {
            result.block_reason = "Target is dead or invalid";
            return HitChance::Impossible;
        }

        // 2. INVULNERABILITY CHECK
        if (Utils::is_invulnerable(target))
        {
            result.block_reason = "Target is invulnerable (Zhonya's, Fizz E, etc.)";
            return HitChance::Impossible;
        }

        // 3. SPELL SHIELD CHECK
        if (Utils::has_spell_shield(target))
        {
            result.block_reason = "Target has spell shield";
            return HitChance::SpellShielded;
        }

        // 4. CC CHECK: Immobile targets
        if (target->has_buff_of_type(buff_type::stun) ||
            target->has_buff_of_type(buff_type::snare) ||
            target->has_buff_of_type(buff_type::charm) ||
            target->has_buff_of_type(buff_type::fear) ||
            target->has_buff_of_type(buff_type::taunt) ||
            target->has_buff_of_type(buff_type::suppression) ||
            target->has_buff_of_type(buff_type::knockup) ||
            target->has_buff_of_type(buff_type::asleep))
        {
            return HitChance::Immobile;
        }

        // 5. DASH HANDLING
        if (target->is_dashing())
        {
            // TODO: Predict to dash endpoint instead of giving up
            // For now, return special status
            result.block_reason = "Target is dashing (endpoint prediction not implemented)";
            return HitChance::Immobile;  // Treat as immobile for now (forced movement)
        }

        // 6. TIME TO IMPACT CALCULATION
        math::vector3 cast_position = predicted_position;  // Where we'll aim
        float time_to_impact = Utils::compute_arrival_time(
            source->get_position(),
            predicted_position,
            input.missile_speed,
            input.cast_delay,
            input.proc_delay
        );
        result.time_to_impact = time_to_impact;

        // 7. DISTANCE TO EXIT CALCULATION (shape-dependent)
        float target_radius = target->get_bounding_radius();
        float distance_to_exit = 0.f;

        if (input.shape == SpellShape::Circle)
        {
            distance_to_exit = Utils::calculate_escape_distance_circle(
                predicted_position,
                cast_position,
                input.spell_width,  // spell_width = radius for circles
                target_radius
            );
        }
        else  // Capsule
        {
            math::vector3 spell_direction = (predicted_position - source->get_position());
            float spell_length = Utils::magnitude_2d(spell_direction);
            if (spell_length > EPSILON)
                spell_direction = spell_direction / spell_length;  // Normalize
            else
                spell_direction = math::vector3(0.f, 0.f, 1.f);  // Default forward

            distance_to_exit = Utils::calculate_escape_distance_capsule(
                predicted_position,
                source->get_position(),
                spell_direction,
                input.spell_width,
                input.spell_range,
                target_radius
            );
        }
        result.distance_to_exit = distance_to_exit;

        // 8. MINION COLLISION CHECK (linear spells only)
        if (input.shape == SpellShape::Capsule)
        {
            if (Utils::has_minion_collision(
                source->get_position(),
                predicted_position,
                input.spell_width))
            {
                result.block_reason = "Minion blocks the spell path";
                return HitChance::Impossible;
            }
        }

        // 9. ANIMATION LOCK TIME
        float animation_lock = Utils::get_animation_lock_time(target);

        // 10. TIME NEEDED TO DODGE
        float move_speed = target->get_move_speed();
        if (move_speed <= 0.0f)
        {
            return HitChance::Immobile;  // Can't move
        }

        float time_needed_to_dodge = animation_lock + (distance_to_exit / move_speed);

        // 11. EFFECTIVE REACTION TIME (FOW adjustment)
        float effective_reaction_reduction = 0.f;
        if (source && g_sdk && g_sdk->nav_mesh)
        {
            int enemy_team = target->get_team_id();
            bool hidden = g_sdk->nav_mesh->is_in_fow_for_team(
                source->get_position(), enemy_team);

            if (!hidden)
            {
                // Visible cast - they react during our windup
                effective_reaction_reduction = input.cast_delay;
            }
        }

        // 12. REACTION WINDOW CALCULATION
        float reaction_window = time_to_impact - time_needed_to_dodge - effective_reaction_reduction;
        result.reaction_window = reaction_window;

        // 13. TERRAIN BLOCKING MULTIPLIER
        float terrain_multiplier = Utils::get_terrain_blocking_multiplier(
            predicted_position,
            cast_position,
            distance_to_exit
        );

        // If trapped (multiplier = 2.0), force Undodgeable
        if (terrain_multiplier >= 1.9f)
        {
            return HitChance::Undodgeable;
        }

        // Apply terrain multiplier to reaction window (smaller window = higher confidence)
        reaction_window /= terrain_multiplier;

        // 14. GRADE REACTION WINDOW TO CONFIDENCE LEVEL
        if (reaction_window <= REACTION_UNDODGEABLE)
            return HitChance::Undodgeable;

        if (reaction_window <= REACTION_VERY_HIGH)
            return HitChance::VeryHigh;

        if (reaction_window <= REACTION_HIGH)
            return HitChance::High;

        if (reaction_window <= REACTION_MEDIUM)
            return HitChance::Medium;

        return HitChance::Low;
    }

    /**
     * THE DRIVER: Main entry point for prediction
     *
     * Combines:
     * 1. Linear path prediction (where will they be?)
     * 2. Hit chance calculation (should we cast?)
     * 3. Simple recommendation (yes/no)
     */
    inline PredictionResult get_prediction(const PredictionInput& input)
    {
        PredictionResult result;

        // Validate input
        if (!input.source || !input.target)
        {
            result.block_reason = "Invalid source or target";
            result.hit_chance = HitChance::Impossible;
            result.should_cast = false;
            return result;
        }

        // Get animation lock time
        float animation_lock = Utils::get_animation_lock_time(input.target);

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

        // Calculate hit chance
        result.hit_chance = calculate_hitchance(input, predicted_pos, result);

        // Simple recommendation: Cast on Medium or higher
        result.should_cast = (
            result.hit_chance == HitChance::Medium ||
            result.hit_chance == HitChance::High ||
            result.hit_chance == HitChance::VeryHigh ||
            result.hit_chance == HitChance::Undodgeable ||
            result.hit_chance == HitChance::Immobile
        );

        return result;
    }

} // namespace GeometricPred
