#pragma once

#include "sdk.hpp"
#include "hp_sdk.hpp"  // Health prediction for ghost minion check
#include "StandalonePredictionSDK.h"  // MUST be included AFTER sdk.hpp for compatibility
#include "PredictionSettings.h"
#include <vector>
#include <string>

/**
 * =============================================================================
 * EDGE CASE DETECTION AND HANDLING
 * =============================================================================
 *
 * This module detects and handles special game states that require
 * adjusted prediction logic:
 *
 * 1. Stasis (Zhonya's, GA, Bard R) - Perfect timing for guaranteed hits
 * 2. Dash prediction - Endpoint calculation with arrival validation
 * 3. Channeling/Recall - High priority stationary targets
 * 4. Slows - Confidence boost
 * 5. Spell shields - Avoid wasting spells
 *
 * =============================================================================
 */

namespace EdgeCases
{
    // =========================================================================
    // STASIS DETECTION AND TIMING
    // =========================================================================

    /**
     * Stasis state information
     */
    struct StasisInfo
    {
        bool is_in_stasis;
        float end_time;                 // Game time when stasis ends
        math::vector3 exit_position;    // Where target will be (current pos)
        std::string stasis_type;        // "zhonyas", "ga", "bard_r", etc.

        StasisInfo() : is_in_stasis(false), end_time(0.f), exit_position{}, stasis_type("") {}
    };

    /**
     * Detect if target is in stasis and get timing info
     */
    inline StasisInfo detect_stasis(game_object* target)
    {
        StasisInfo info;

        if (!target || !target->is_valid())
            return info;

        float current_time = 0.f;
        if (g_sdk && g_sdk->clock_facade)
            current_time = g_sdk->clock_facade->get_game_time();

        auto check_buff = [&](const std::string& name, const std::string& type)
        {
            std::string buff_name = name;  // Make mutable copy for API
            auto buff = target->get_buff_by_name(buff_name);
            if (buff && buff->is_active())
            {
                float end = buff->get_end_time();
                float duration_left = end - current_time;

                // CRITICAL FIX: Ignore "permanent" buffs (like GA ready state)
                // Real stasis is short (< 4s). If duration is huge, it's an item/passive indicator.
                if (duration_left > 0.f && duration_left < 5.0f)
                {
                    info.is_in_stasis = true;
                    info.end_time = end;
                    info.exit_position = target->get_position();
                    info.stasis_type = type;
                    return true;
                }
            }
            return false;
        };

        // Zhonya's Hourglass (Standard)
        if (check_buff("zhonyasringshield", "zhonyas")) return info;

        // Guardian Angel (Reviving)
        // Note: "willrevive" is often the 'ready' buff. The actual reviving buff
        // implies stasis. We strictly filter by duration < 5s to be safe.
        if (check_buff("willrevive", "guardian_angel")) return info;

        // Bard R
        if (check_buff("bardrstasis", "bard_r")) return info;

        // Lissandra R (Self)
        if (check_buff("lissandrarstasis", "lissandra_r")) return info;

        // Zac Passive (Reviving states often use stasis logic)
        if (check_buff("zacrebirthready", "zac_passive")) return info;

        return info;
    }

    /**
     * Calculate optimal cast timing for stasis exit
     * Returns: Time to wait before casting (0 = cast now, -1 = impossible)
     */
    inline float calculate_stasis_cast_timing(
        const StasisInfo& stasis,
        float spell_travel_time,
        float current_time)
    {
        if (!stasis.is_in_stasis)
            return 0.f;  // Not in stasis, cast normally

        float time_until_exit = stasis.end_time - current_time;

        // =====================================================================
        // CRITICAL: Safety buffer for server tick rate and network jitter
        // =====================================================================
        // Server tick rate: 30Hz = 33ms per tick
        // Network jitter: ±5ms typical
        // Problem: If spell arrives exactly at stasis end, server might still
        //          register target as invulnerable (not processed yet)
        // Solution: Aim for 1 FRAME after stasis ends to catch frame-perfect escapes
        //
        // OLD (40ms): Gave full server tick + jitter → too generous, scripters/pros flash out
        // NEW (16ms): ~1 client frame @ 60fps → minimal window, catches most escapes
        // Trade-off: Tight timing may miss on high ping (>50ms), but guarantees hit on LAN/low ping
        // =====================================================================
        constexpr float SAFETY_BUFFER = 0.016f;  // ~1 frame (16ms @ 60Hz client) - tight timing for guaranteed hits
        float optimal_cast_delay = time_until_exit - spell_travel_time + SAFETY_BUFFER;

        // If we need to wait before casting
        if (optimal_cast_delay > 0.f)
            return optimal_cast_delay;  // Wait this long before casting

        // If we can cast now (spell will arrive after stasis + buffer)
        // FIXED: Include == 0 case (was causing false "impossible" when timing was exactly 0)
        if (optimal_cast_delay >= -0.1f && optimal_cast_delay <= 0.f)
            return 0.f;  // Cast immediately

        // If stasis already ended, normal prediction
        if (time_until_exit < 0.f)
            return 0.f;

        // Spell travel time too long, we'll miss the timing window
        return -1.f;  // Impossible
    }

    // =========================================================================
    // DASH DETECTION AND PREDICTION
    // =========================================================================

    /**
     * Dash prediction information
     */
    struct DashInfo
    {
        bool is_dashing;
        math::vector3 dash_end_position;
        float dash_arrival_time;        // When enemy reaches dash end
        float dash_speed;
        float confidence_multiplier;    // 0.5-1.0 based on dash length

        DashInfo() : is_dashing(false), dash_end_position{},
            dash_arrival_time(0.f), dash_speed(0.f), confidence_multiplier(1.0f) {
        }
    };

    /**
     * Detect dash and calculate endpoint
     */
    inline DashInfo detect_dash(game_object* target)
    {
        DashInfo info;

        if (!target || !target->is_valid())
            return info;

        info.is_dashing = target->is_dashing();
        if (!info.is_dashing)
            return info;

        // Get dash path
        auto path = target->get_path();
        if (path.empty())
        {
            info.confidence_multiplier = 0.3f;  // No path data, very uncertain
            return info;
        }

        // Dash endpoint = last waypoint
        info.dash_end_position = path.back();
        info.dash_speed = target->get_dash_speed();

        // Calculate arrival time at dash end
        // FIX: Use server position for accurate dash tracking (client lags behind)
        float distance = (info.dash_end_position - target->get_server_position()).magnitude();

        constexpr float EPSILON = 0.01f;
        if (info.dash_speed > EPSILON)
        {
            info.dash_arrival_time = distance / info.dash_speed;
        }
        else
        {
            // Instant dash or unknown speed - use short default
            info.dash_arrival_time = 0.1f;  // 100ms default
        }

        // Confidence based on dash travel time
        // Key insight: Longer dashes = committed animation = easier to intercept
        // Blinks are HARDER despite known endpoint (no flight interception, instant reaction time needed)
        if (info.dash_arrival_time < 0.1f)
        {
            // Instant blink (Flash, Ezreal E, Kassadin R)
            // LOWER confidence because:
            // - No interception window during flight (already completed)
            // - They can move/react immediately after landing
            // - Detection delay means they have time to dodge our incoming spell
            // - No "committed" animation period to exploit
            info.confidence_multiplier = 0.75f;
        }
        else if (info.dash_arrival_time < 0.3f)
        {
            // Short dash (Graves E, Vayne Q)
            // Brief travel time, small interception window but still catchable
            info.confidence_multiplier = 0.85f;
        }
        else if (info.dash_arrival_time < 0.6f)
        {
            // Medium dash (Tristana W, Lucian E)
            // Optimal: Committed to animation, good interception window during flight
            info.confidence_multiplier = 0.95f;
        }
        else
        {
            // Long dash (Zac E, long Tristana W)
            // Easiest: Long commitment period, multiple opportunities to intercept
            info.confidence_multiplier = 1.0f;
        }

        return info;
    }

    /**
     * Validate dash prediction timing
     * Returns true if we should predict at dash end, false if too early
     */
    inline bool validate_dash_timing(
        const DashInfo& dash,
        float spell_arrival_time,
        float current_time)
    {
        if (!dash.is_dashing)
            return true;  // Not dashing, normal prediction

        // CRITICAL: Don't cast before enemy arrives at dash endpoint
        // Our spell must arrive AFTER enemy reaches dash end
        float enemy_arrival = current_time + dash.dash_arrival_time;
        float spell_arrival = current_time + spell_arrival_time;

        // Only predict at dash end if spell arrives after enemy gets there
        return spell_arrival >= enemy_arrival;
    }

    // =========================================================================
    // CHANNELING AND RECALL DETECTION
    // =========================================================================

    /**
     * Channel information
     */
    struct ChannelInfo
    {
        bool is_channeling;
        bool is_recalling;
        float channel_end_time;
        float time_remaining;
        math::vector3 position;  // Stationary position

        ChannelInfo() : is_channeling(false), is_recalling(false),
            channel_end_time(0.f), time_remaining(0.f), position{} {
        }
    };

    /**
     * Forced movement information (charm, taunt, fear)
     * These CCs force the target to move in a predictable direction
     */
    struct ForcedMovementInfo
    {
        bool has_forced_movement;
        bool is_charm;   // Walks toward source (Ahri E, Evelynn W, Rakan W)
        bool is_taunt;   // Walks toward source (Galio E, Rammus E, Shen E)
        bool is_fear;    // Runs away from source (Fiddlesticks Q, Hecarim R, Shaco W)
        float duration_remaining;
        math::vector3 forced_direction;  // Normalized direction they're forced to walk
        game_object* cc_source;          // Who applied the CC

        ForcedMovementInfo() : has_forced_movement(false), is_charm(false),
            is_taunt(false), is_fear(false), duration_remaining(0.f),
            forced_direction{}, cc_source(nullptr) {
        }
    };

    /**
     * Detect forced movement CCs (charm, taunt, fear)
     * Returns direction target will be forced to walk
     */
    inline ForcedMovementInfo detect_forced_movement(game_object* target, game_object* source = nullptr)
    {
        ForcedMovementInfo info;

        if (!target || !target->is_valid())
            return info;

        if (!g_sdk || !g_sdk->clock_facade)
            return info;

        float current_time = g_sdk->clock_facade->get_game_time();

        // Check for charm
        if (target->has_buff_of_type(buff_type::charm))
        {
            info.has_forced_movement = true;
            info.is_charm = true;

            // Find charm source from buffs
            auto buffs = target->get_buffs();
            for (auto* buff : buffs)
            {
                if (buff && buff->is_active() && buff->get_type() == buff_type::charm)
                {
                    info.duration_remaining = std::max(0.f, buff->get_end_time() - current_time);
                    info.cc_source = buff->get_caster();

                    if (info.cc_source && info.cc_source->is_valid())
                    {
                        // Charm: walk toward caster
                        math::vector3 to_caster = info.cc_source->get_position() - target->get_position();
                        float dist = to_caster.magnitude();
                        if (dist > 0.001f)
                            info.forced_direction = to_caster / dist;  // Normalize
                    }
                    break;
                }
            }
        }
        // Check for taunt
        else if (target->has_buff_of_type(buff_type::taunt))
        {
            info.has_forced_movement = true;
            info.is_taunt = true;

            auto buffs = target->get_buffs();
            for (auto* buff : buffs)
            {
                if (buff && buff->is_active() && buff->get_type() == buff_type::taunt)
                {
                    info.duration_remaining = std::max(0.f, buff->get_end_time() - current_time);
                    info.cc_source = buff->get_caster();

                    if (info.cc_source && info.cc_source->is_valid())
                    {
                        // Taunt: walk toward caster
                        math::vector3 to_caster = info.cc_source->get_position() - target->get_position();
                        float dist = to_caster.magnitude();
                        if (dist > 0.001f)
                            info.forced_direction = to_caster / dist;
                    }
                    break;
                }
            }
        }
        // Check for fear
        else if (target->has_buff_of_type(buff_type::fear))
        {
            info.has_forced_movement = true;
            info.is_fear = true;

            auto buffs = target->get_buffs();
            for (auto* buff : buffs)
            {
                if (buff && buff->is_active() && buff->get_type() == buff_type::fear)
                {
                    info.duration_remaining = std::max(0.f, buff->get_end_time() - current_time);
                    info.cc_source = buff->get_caster();

                    if (info.cc_source && info.cc_source->is_valid())
                    {
                        // Fear: run away from caster
                        math::vector3 away_from_caster = target->get_position() - info.cc_source->get_position();
                        float dist = away_from_caster.magnitude();
                        if (dist > 0.001f)
                            info.forced_direction = away_from_caster / dist;
                    }
                    break;
                }
            }
        }

        return info;
    }

    /**
     * Detect channeling or recall
     */
    inline ChannelInfo detect_channel(game_object* target)
    {
        ChannelInfo info;

        if (!target || !target->is_valid())
            return info;

        if (!g_sdk || !g_sdk->clock_facade)
            return info;

        float current_time = g_sdk->clock_facade->get_game_time();
        info.position = target->get_position();

        // Check recall
        info.is_recalling = is_recalling(target);
        if (info.is_recalling)
        {
            std::string recall_buff_name = "recall";
            auto recall_buff = target->get_buff_by_name(recall_buff_name);
            if (recall_buff && recall_buff->is_active())
            {
                info.channel_end_time = recall_buff->get_end_time();
                info.time_remaining = info.channel_end_time - current_time;
            }
            return info;
        }

        // Check channeling
        info.is_channeling = is_channeling(target);
        if (info.is_channeling)
        {
            // Try to get channel duration from common channeled spells
            // This is approximate - exact timing depends on spell
            info.time_remaining = 2.0f;  // Default estimate
        }

        return info;
    }

    /**
     * Check if we can interrupt channel before it completes
     */
    inline bool can_interrupt_channel(
        const ChannelInfo& channel,
        float spell_travel_time)
    {
        if (!channel.is_channeling && !channel.is_recalling)
            return false;

        // Can we hit before channel completes?
        return spell_travel_time < channel.time_remaining;
    }

    // =========================================================================
    // SLOW DETECTION
    // =========================================================================

    /**
     * Check if target is slowed
     */
    inline bool is_slowed(game_object* target)
    {
        if (!target || !target->is_valid())
            return false;

        // Method 1: Check for slow buff type
        if (target->has_buff_of_type(buff_type::slow))
            return true;

        // Method 2: Compare current vs base move speed
        float current_speed = target->get_move_speed();
        float base_speed = get_base_move_speed(target);

        // Consider slowed if moving < 95% of base speed
        return current_speed < base_speed * 0.95f;
    }

    // =========================================================================
    // SPELL SHIELD DETECTION
    // =========================================================================

    /**
     * Check if target has spell shield active
     */
    inline bool has_spell_shield(game_object* target)
    {
        if (!target || !target->is_valid())
            return false;

        // Common spell shields
        std::string banshees = "bansheesveil";
        if (target->get_buff_by_name(banshees)) return true;

        std::string sivir = "sivirshield";
        if (target->get_buff_by_name(sivir)) return true;

        std::string nocturne = "nocturneshroudofdarkness";
        if (target->get_buff_by_name(nocturne)) return true;

        std::string morgana = "morganablackshield";
        if (target->get_buff_by_name(morgana)) return true;

        std::string malzahar = "malzaharpassiveshield";
        if (target->get_buff_by_name(malzahar)) return true;

        return false;
    }

    // =========================================================================
    // WINDWALL DETECTION (Yasuo W, Samira W, Braum E)
    // =========================================================================

    /**
     * Windwall information
     */
    struct WindwallInfo
    {
        bool exists;
        math::vector3 position;
        math::vector3 direction;      // Direction windwall is facing
        float width;
        float end_time;
        std::string source_champion;  // "yasuo", "samira", "braum"
        bool is_circle;               // true for Samira (360° block), false for line walls

        WindwallInfo() : exists(false), position{}, direction{},
            width(0.f), end_time(0.f), source_champion(""), is_circle(false) {
        }
    };

    /**
     * Get Yasuo W width based on ability rank
     * Width scales: 320 / 390 / 460 / 530 / 600
     */
    inline float get_yasuo_wall_width(game_object* yasuo)
    {
        if (!yasuo || !yasuo->is_valid())
            return 600.f;  // Default to max if can't determine

        // Try to get W spell level (W = slot 1: Q=0, W=1, E=2, R=3)
        auto w_spell = yasuo->get_spell(1);
        if (w_spell)
        {
            int level = w_spell->get_level();
            if (level >= 1 && level <= 5)
            {
                static const float widths[] = { 320.f, 390.f, 460.f, 530.f, 600.f };
                return widths[level - 1];
            }
        }
        return 600.f;  // Default to max width to be safe
    }

    /**
     * Detect active windwalls that can block projectiles
     * Uses game object names from actual game data dumps
     */
    inline std::vector<WindwallInfo> detect_windwalls()
    {
        std::vector<WindwallInfo> windwalls;

        if (!g_sdk || !g_sdk->object_manager)
            return windwalls;

        auto* local_player = g_sdk->object_manager->get_local_player();
        if (!local_player || !local_player->is_valid())
            return windwalls;

        int local_team = local_player->get_team_id();
        float current_time = 0.f;
        if (g_sdk->clock_facade)
            current_time = g_sdk->clock_facade->get_game_time();

        // Object names that indicate projectile-blocking walls
        // From game data dumps - these are the actual blocking objects
        static const char* YASUO_WALL_NAMES[] = {
            "Yasuo_Base_W_windwall1",
            "Yasuo_Base_W_windwall_activate",
            "YasuoWMovingWallMis"  // Alternative name
        };

        static const char* SAMIRA_WALL_NAMES[] = {
            "Samira_Base_W_windwall",
            "SamiraW"
        };

        static const char* BRAUM_SHIELD_NAMES[] = {
            "Braum_Base_E_Shield",
            "BraumShieldRaise"
        };

        // Cache enemy Yasuo for width lookup
        game_object* enemy_yasuo = nullptr;
        auto heroes = g_sdk->object_manager->get_heroes();
        for (auto* hero : heroes)
        {
            if (!hero || !hero->is_valid() || hero->is_dead())
                continue;
            if (hero->get_team_id() == local_team)
                continue;

            std::string char_name = hero->get_char_name();
            if (char_name.find("Yasuo") != std::string::npos)
            {
                enemy_yasuo = hero;
                break;
            }
        }

        // Search through minions/objects for wall entities
        auto minions = g_sdk->object_manager->get_minions();
        for (auto* obj : minions)
        {
            if (!obj || !obj->is_valid() || !obj->is_visible())
                continue;

            // Skip allied objects
            if (obj->get_team_id() == local_team)
                continue;

            std::string name = obj->get_name();
            if (name.empty())
                continue;

            WindwallInfo wall;
            wall.exists = false;

            // Check for Yasuo W
            for (const char* wall_name : YASUO_WALL_NAMES)
            {
                if (name.find(wall_name) != std::string::npos)
                {
                    wall.exists = true;
                    wall.position = obj->get_position();
                    wall.direction = obj->get_direction();
                    wall.width = enemy_yasuo ? get_yasuo_wall_width(enemy_yasuo) : 600.f;
                    wall.end_time = current_time + 4.0f;  // 4 second duration
                    wall.source_champion = "yasuo";
                    wall.is_circle = false;
                    break;
                }
            }

            // Check for Samira W
            if (!wall.exists)
            {
                for (const char* wall_name : SAMIRA_WALL_NAMES)
                {
                    if (name.find(wall_name) != std::string::npos)
                    {
                        wall.exists = true;
                        wall.position = obj->get_position();
                        wall.direction = math::vector3(0, 0, 0);  // Circle, no direction
                        wall.width = 325.f;  // Samira W radius
                        wall.end_time = current_time + 0.75f;  // Short duration
                        wall.source_champion = "samira";
                        wall.is_circle = true;
                        break;
                    }
                }
            }

            // Check for Braum E
            if (!wall.exists)
            {
                for (const char* wall_name : BRAUM_SHIELD_NAMES)
                {
                    if (name.find(wall_name) != std::string::npos)
                    {
                        wall.exists = true;
                        wall.position = obj->get_position();
                        wall.direction = obj->get_direction();
                        wall.width = 250.f;  // Braum shield width
                        wall.end_time = current_time + 4.0f;  // 4 second duration
                        wall.source_champion = "braum";
                        wall.is_circle = false;
                        break;
                    }
                }
            }

            if (wall.exists)
            {
                // Validate direction - if zero, try to estimate from nearby champion
                if (wall.direction.magnitude() < 0.1f && !wall.is_circle)
                {
                    // For Yasuo/Braum, wall faces away from caster
                    // Find the closest enemy champion to use their facing direction
                    for (auto* hero : heroes)
                    {
                        if (!hero || !hero->is_valid() || hero->is_dead())
                            continue;
                        if (hero->get_team_id() == local_team)
                            continue;

                        float dist = (hero->get_position() - wall.position).magnitude();
                        if (dist < 200.f)  // Wall should be close to caster
                        {
                            wall.direction = hero->get_direction();
                            break;
                        }
                    }

                    // Final fallback - face toward local player (worst case)
                    if (wall.direction.magnitude() < 0.1f)
                    {
                        math::vector3 to_local = local_player->get_position() - wall.position;
                        float dist = to_local.magnitude();
                        if (dist > 1.f)
                            wall.direction = to_local / dist;
                        else
                            wall.direction = math::vector3(1, 0, 0);
                    }
                }

                windwalls.push_back(wall);
            }
        }

        return windwalls;
    }

    /**
     * Check if projectile path intersects with any windwall
     * Handles both line walls (Yasuo/Braum) and circular walls (Samira)
     */
    inline bool will_hit_windwall(
        const math::vector3& start_pos,
        const math::vector3& end_pos,
        const std::vector<WindwallInfo>& windwalls)
    {
        for (const auto& wall : windwalls)
        {
            if (!wall.exists)
                continue;

            // Calculate projectile direction vector
            math::vector3 to_target = end_pos - start_pos;
            float distance_to_target = to_target.magnitude();

            constexpr float MIN_SAFE_DISTANCE = 1.0f;
            if (distance_to_target < MIN_SAFE_DISTANCE)
                continue;

            math::vector3 direction = to_target / distance_to_target;

            // Project windwall position onto the projectile path
            math::vector3 start_to_wall = wall.position - start_pos;
            float projection = start_to_wall.dot(direction);

            // Clamp projection to line segment [0, distance_to_target]
            if (projection < 0.f || projection > distance_to_target)
                continue;  // Wall is not between start and end

            // Calculate closest point on path to wall center
            math::vector3 closest_point_on_path = start_pos + direction * projection;
            float perpendicular_distance = (wall.position - closest_point_on_path).magnitude();

            if (wall.is_circle)
            {
                // CIRCULAR WALL (Samira W): Block if path passes through circle
                // Check if projectile path intersects the circular zone
                if (perpendicular_distance < wall.width)
                {
                    return true;  // Path passes through Samira's W circle
                }
            }
            else
            {
                // LINE WALL (Yasuo/Braum): Check if path crosses the wall line
                // Wall is a line segment perpendicular to wall.direction
                // Half-width on each side of wall.position

                // Calculate wall endpoints
                math::vector3 wall_perp(-wall.direction.z, 0, wall.direction.x);
                float half_width = wall.width / 2.f;

                // Check if projectile path crosses the wall line
                // The path crosses if the closest point is within half_width of wall center
                // AND the path actually goes through the wall plane

                // Check perpendicular distance to wall center
                if (perpendicular_distance < half_width)
                {
                    // Path is within wall width - check if it crosses the wall plane
                    // Wall plane is defined by wall.position and wall.direction (normal)
                    float start_side = (start_pos - wall.position).dot(wall.direction);
                    float end_side = (end_pos - wall.position).dot(wall.direction);

                    // If signs differ, projectile crosses the wall plane
                    if (start_side * end_side < 0.f)
                    {
                        return true;  // Projectile crosses Yasuo/Braum wall
                    }
                }
            }
        }

        return false;
    }

    // =========================================================================
    // MINION COLLISION DETECTION
    // =========================================================================

    /**
     * Check if projectile path is blocked by minions
     * Returns probability that spell will hit target (accounting for minion blocking)
     *
     * collides_with_minions: Whether this spell is blocked by minions
     *   - false for spells that pierce (Yasuo Q, Ezreal Q with Muramana, etc.)
     *   - true for spells that collide (Thresh Q, Blitz Q, Lux Q)
     */
    inline float compute_minion_block_probability(
        const math::vector3& source_pos,
        const math::vector3& target_pos,
        float projectile_width,
        bool collides_with_minions)
    {
        // If spell pierces minions, no collision
        if (!collides_with_minions)
            return 1.0f;

        if (!g_sdk || !g_sdk->object_manager)
            return 1.0f;

        // Check distance threshold for both enemy and ally minions
        constexpr float MIN_SAFE_DISTANCE = 1.0f;
        math::vector3 to_target = target_pos - source_pos;
        float distance_to_target = to_target.magnitude();

        if (distance_to_target < MIN_SAFE_DISTANCE)
            return 1.0f;  // Too close to worry about minions

        math::vector3 direction = to_target / distance_to_target;

        int blocking_minions = 0;

        // Get all minions (SDK uses get_minions(), not get_enemy_minions())
        if (!g_sdk || !g_sdk->object_manager)
            return 1.0f;  // Can't check minions, assume clear
        auto minions = g_sdk->object_manager->get_minions();
        for (auto* minion : minions)
        {
            if (!minion || !minion->is_valid() || !minion->is_visible())
                continue;

            // Skip wards (don't block skillshots)
            std::string name = minion->get_char_name();
            if (name.find("Ward") != std::string::npos ||
                name.find("Trinket") != std::string::npos ||
                name.find("YellowTrinket") != std::string::npos)
                continue;

            // PERFORMANCE: Skip minions too far away (only check within 2000 units)
            math::vector3 minion_pos = minion->get_position();
            float dist_to_source = (minion_pos - source_pos).magnitude();
            constexpr float MINION_RELEVANCE_RANGE = 2000.f;
            if (dist_to_source > MINION_RELEVANCE_RANGE)
                continue;

            // Check if minion is between source and target
            math::vector3 to_minion = minion_pos - source_pos;
            float distance_along_path = to_minion.dot(direction);

            // Minion must be between source and target
            if (distance_along_path < 0.f || distance_along_path > distance_to_target)
                continue;

            // Calculate travel time to minion position (assume ~1200 projectile speed if not specified)
            float travel_time = distance_along_path / 1200.f;

            // GHOST MINION FIX: Skip minions that will be DEAD when our projectile arrives
            // This is more accurate than simple HP% thresholds
            //
            // Priority 1: Use health prediction SDK if available (tracks incoming damage)
            // Priority 2: Fall back to conservative heuristic
            if (sdk::health_prediction)
            {
                // Predicted HP at arrival time - accounts for tower shots, ally autos, etc.
                float predicted_hp = sdk::health_prediction->get_predicted_health(minion, travel_time);
                if (predicted_hp <= 0.f)
                    continue;  // Minion will be dead, skip it
            }
            else
            {
                // IMPROVED FALLBACK: Time-based heuristic when HP prediction SDK unavailable
                // Estimates if minion will die before spell arrives based on current HP and typical lane DPS
                float current_hp = minion->get_hp();
                float max_hp = minion->get_max_hp();

                // Estimate incoming DPS based on minion type and game time
                // Melee minions: ~100 HP at 0min, ~1500 HP at 30min (linear scaling)
                // Typical lane DPS: ~50-150 depending on game time (minion autos, tower, champion)
                constexpr float LANE_BASE_DPS = 50.f;   // Base lane DPS
                constexpr float TOWER_DPS = 250.f;      // Tower shot DPS
                constexpr float TOWER_RANGE = 900.f;    // Tower aggro range
                float estimated_dps = LANE_BASE_DPS;

                // Higher DPS if minion is under tower (tower shots = massive burst)
                if (g_sdk && g_sdk->nav_mesh)
                {
                    // Check if minion is near an enemy tower
                    auto towers = g_sdk->object_manager->get_enemy_turrets();
                    for (auto* tower : towers)
                    {
                        if (tower && tower->is_valid() && !tower->is_dead())
                        {
                            float dist_to_tower = (minion->get_position() - tower->get_position()).magnitude();
                            if (dist_to_tower < TOWER_RANGE)
                            {
                                estimated_dps = TOWER_DPS;  // Tower shots = very high DPS
                                break;
                            }
                        }
                    }
                }

                // Estimate survival time: HP / DPS
                float estimated_survival_time = (estimated_dps > 0.f) ? (current_hp / estimated_dps) : 999.f;

                // Skip minion if it will die before spell arrives
                // Add 0.05s safety buffer to avoid edge cases
                if (estimated_survival_time < (travel_time - 0.05f))
                    continue;  // Minion will be dead when spell arrives

                // Additional conservative check: Very low HP minions (< 5%) are risky
                // Even if survival time > travel time, they might die from unexpected damage
                float health_percent = (max_hp > 0.f) ? (current_hp / max_hp) * 100.f : 100.f;
                if (health_percent < 5.f && travel_time > 0.15f)
                    continue;  // Too risky, assume it will die
            }

            if (minion->is_moving() && travel_time > 0.05f)
            {
                auto path = minion->get_path();
                if (!path.empty())
                {
                    // Last waypoint is the path end
                    math::vector3 minion_path_end = path.back();
                    math::vector3 move_dir = minion_path_end - minion_pos;
                    float move_dist = move_dir.magnitude();

                    if (move_dist > 1.0f)
                    {
                        move_dir = move_dir / move_dist;
                        float minion_speed = minion->get_move_speed();
                        float predicted_move = std::min(minion_speed * travel_time, move_dist);
                        minion_pos = minion_pos + move_dir * predicted_move;

                        // Recalculate distance along path with predicted position
                        to_minion = minion_pos - source_pos;
                        distance_along_path = to_minion.dot(direction);

                        // Skip if minion will have moved out of path
                        if (distance_along_path < 0.f || distance_along_path > distance_to_target)
                            continue;
                    }
                }
            }

            // Compute perpendicular distance from minion to projectile path
            math::vector3 closest_point_on_path = source_pos + direction * distance_along_path;
            float perpendicular_distance = (minion_pos - closest_point_on_path).magnitude();

            // Check if minion hitbox overlaps projectile
            // SAFETY BUFFER: Add ~18 units to account for:
            // - Minion movement during projectile flight
            // - Hitbox uncertainty / server-client desync
            // - "Grazing" shots that look like they should pass but don't
            // 18 units ≈ 1/3 of average champion hitbox (55 units)
            constexpr float MINION_COLLISION_SAFETY_BUFFER = 18.f;
            float minion_radius = minion->get_bounding_radius();
            if (perpendicular_distance < minion_radius + projectile_width + MINION_COLLISION_SAFETY_BUFFER)
            {
                blocking_minions++;
            }
        }

        // Each minion blocks approximately 30% chance
        // Multiple minions: P(hit) = 0.7^n
        // 1 minion: 0.70 (70% hit chance)
        // 2 minions: 0.49 (49% hit chance)
        // 3+ minions: very low (<35%)
        if (blocking_minions == 0)
            return 1.0f;

        return std::pow(0.7f, static_cast<float>(blocking_minions));
    }

    // =========================================================================
    // CLONE DETECTION
    // =========================================================================

    /**
     * Check if object is a clone (Shaco, Wukong, LeBlanc)
     * Returns true if it's a real champion, false if clone
     */
    inline bool is_real_champion(game_object* obj)
    {
        if (!obj || !obj->is_valid())
            return false;

        // Clones typically have specific patterns in their name
        std::string name = obj->get_name();

        // Shaco clone detection
        if (name.find("shaco") != std::string::npos)
        {
            // Real Shaco has different network ID than clone
            // Clone typically has "clone" in buff list
            std::string shaco_buff = "shacopassive";
            if (obj->get_buff_by_name(shaco_buff))
                return false;  // This is a clone
        }

        // Wukong clone detection
        if (name.find("monkeyking") != std::string::npos)
        {
            std::string wukong_buff = "monkeykingdecoy";
            if (obj->get_buff_by_name(wukong_buff))
                return false;  // This is a clone
        }

        // LeBlanc clone detection
        if (name.find("leblanc") != std::string::npos)
        {
            std::string leblanc_buff = "leblancpassive";
            if (obj->get_buff_by_name(leblanc_buff))
                return false;  // This is a clone
        }

        // Neeko clone detection
        std::string neeko_buff = "neekopassive";
        if (obj->get_buff_by_name(neeko_buff))
            return false;

        // Default: assume it's real
        return true;
    }

    // =========================================================================
    // COMBINED EDGE CASE ANALYSIS
    // =========================================================================

    /**
     * Complete edge case analysis for a target
     */
    struct EdgeCaseAnalysis
    {
        StasisInfo stasis;
        DashInfo dash;
        ChannelInfo channel;
        ForcedMovementInfo forced_movement;
        std::vector<WindwallInfo> windwalls;
        bool is_slowed;
        bool has_shield;
        bool is_clone;
        bool blocked_by_windwall;

        // Confidence and priority adjustments
        float confidence_multiplier;
        float priority_multiplier;

        EdgeCaseAnalysis() : is_slowed(false), has_shield(false),
            is_clone(false), blocked_by_windwall(false),
            confidence_multiplier(1.0f), priority_multiplier(1.0f) {
        }
    };

    /**
     * Analyze all edge cases for target
     */
    inline EdgeCaseAnalysis analyze_target(game_object* target, game_object* source = nullptr)
    {
        EdgeCaseAnalysis analysis;

        if (!target || !target->is_valid())
            return analysis;

        // Detect all edge cases
        analysis.stasis = detect_stasis(target);

        // Dash prediction (configurable)
        if (PredictionSettings::get().enable_dash_prediction)
        {
            analysis.dash = detect_dash(target);
        }

        analysis.channel = detect_channel(target);
        analysis.forced_movement = detect_forced_movement(target, source);
        analysis.windwalls = detect_windwalls();
        analysis.is_slowed = is_slowed(target);
        analysis.has_shield = has_spell_shield(target);
        analysis.is_clone = !is_real_champion(target);

        // Check windwall blocking (only if source provided)
        if (source && source->is_valid())
        {
            analysis.blocked_by_windwall = will_hit_windwall(
                source->get_position(),
                target->get_position(),
                analysis.windwalls
            );
        }

        // Calculate adjustments

        // PRIORITY MULTIPLIERS
        if (analysis.stasis.is_in_stasis)
            analysis.priority_multiplier *= 3.0f;  // Guaranteed hit opportunity

        if (analysis.channel.is_channeling || analysis.channel.is_recalling)
            analysis.priority_multiplier *= 2.5f;  // High value interrupts

        if (analysis.forced_movement.has_forced_movement)
            analysis.priority_multiplier *= 2.8f;  // Forced movement = very predictable

        // CONFIDENCE MULTIPLIERS
        if (analysis.stasis.is_in_stasis)
            analysis.confidence_multiplier *= 1.5f;  // Very confident on stasis exit

        if (analysis.channel.is_channeling || analysis.channel.is_recalling)
            analysis.confidence_multiplier *= 1.3f;  // Target is stationary

        if (analysis.forced_movement.has_forced_movement)
            analysis.confidence_multiplier *= 1.4f;  // Forced movement is very predictable

        if (analysis.is_slowed)
            analysis.confidence_multiplier *= 1.15f;  // Reduced mobility

        if (analysis.dash.is_dashing)
            analysis.confidence_multiplier *= analysis.dash.confidence_multiplier;

        // Spell shields: Don't penalize confidence - we want to break shields
        // Instead, reduce priority so we prefer unshielded targets when multiple options
        if (analysis.has_shield)
            analysis.priority_multiplier *= 0.7f;  // Prefer unshielded targets

        if (analysis.blocked_by_windwall)
            analysis.confidence_multiplier *= 0.2f;  // Will be blocked by windwall

        if (analysis.is_clone)
            analysis.priority_multiplier *= 0.1f;  // Don't target clones

        return analysis;
    }

} // namespace EdgeCases