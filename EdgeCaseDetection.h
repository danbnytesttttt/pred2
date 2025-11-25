#pragma once

#include "sdk.hpp"
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

        // Check Zhonya's Hourglass
        std::string zhonyas_buff = "zhonyasringshield";
        auto zhonyas = target->get_buff_by_name(zhonyas_buff);
        if (zhonyas && zhonyas->is_active())
        {
            info.is_in_stasis = true;
            info.end_time = zhonyas->get_end_time();
            info.exit_position = target->get_position();
            info.stasis_type = "zhonyas";
            return info;
        }

        // Check Guardian Angel
        std::string ga_buff = "willrevive";
        auto ga = target->get_buff_by_name(ga_buff);
        if (ga && ga->is_active())
        {
            info.is_in_stasis = true;
            info.end_time = ga->get_end_time();
            info.exit_position = target->get_position();
            info.stasis_type = "guardian_angel";
            return info;
        }

        // Check Bard R
        std::string bard_buff = "bardrstasis";
        auto bard_r = target->get_buff_by_name(bard_buff);
        if (bard_r && bard_r->is_active())
        {
            info.is_in_stasis = true;
            info.end_time = bard_r->get_end_time();
            info.exit_position = target->get_position();
            info.stasis_type = "bard_r";
            return info;
        }

        // Check Lissandra R (self-cast)
        std::string liss_buff = "lissandrarstasis";
        auto liss_r = target->get_buff_by_name(liss_buff);
        if (liss_r && liss_r->is_active())
        {
            info.is_in_stasis = true;
            info.end_time = liss_r->get_end_time();
            info.exit_position = target->get_position();
            info.stasis_type = "lissandra_r";
            return info;
        }

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
        // Solution: Aim for 40ms AFTER stasis ends (not before!)
        //
        // Old bug: Subtracted buffer → spell arrives early → wasted on invuln
        // New fix: Add buffer → spell arrives after next server tick → hits
        // =====================================================================
        constexpr float SAFETY_BUFFER = 0.04f;  // 40ms after stasis ends
        float optimal_cast_delay = time_until_exit - spell_travel_time + SAFETY_BUFFER;

        // If we need to wait before casting
        if (optimal_cast_delay > 0.f)
            return optimal_cast_delay;  // Wait this long before casting

        // If we can cast now (spell will arrive after stasis + buffer)
        if (optimal_cast_delay >= -0.1f && optimal_cast_delay < 0.f)
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
        float distance = (info.dash_end_position - target->get_position()).magnitude();

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

        WindwallInfo() : exists(false), position{}, direction{},
            width(0.f), end_time(0.f), source_champion("") {
        }
    };

    /**
     * Detect active windwalls that can block projectiles
     *
     * DISABLED: Cannot accurately track windwall position without particle API
     * - Yasuo W spawns in front of him and drifts forward
     * - Using hero position is incorrect (wall != hero location)
     * - Particle tracking would require complex object iteration
     *
     * TODO: Implement proper particle-based detection or remove entirely
     */
    inline std::vector<WindwallInfo> detect_windwalls()
    {
        // DISABLED - returns empty (no windwall detection)
        return std::vector<WindwallInfo>();

        /* DISABLED CODE - Geometry broken without particle positions
        std::vector<WindwallInfo> windwalls;

        if (!g_sdk || !g_sdk->object_manager)
            return windwalls;

        auto* local_player = g_sdk->object_manager->get_local_player();
        if (!local_player || !local_player->is_valid())
            return windwalls;

        int local_team = local_player->get_team_id();

        // Get all heroes and filter manually by team
        auto heroes = g_sdk->object_manager->get_heroes();
        for (auto* hero : heroes)
        {
            if (!hero || !hero->is_valid() || hero->is_dead())
                continue;

            // Skip allies - only check enemies
            if (hero->get_team_id() == local_team)
                continue;

            // Yasuo Windwall
            std::string yasuo_buff = "yasuowmovingwall";
            auto yasuo_wall = hero->get_buff_by_name(yasuo_buff);
            if (yasuo_wall && yasuo_wall->is_active())
            {
                WindwallInfo wall;
                wall.exists = true;
                wall.position = hero->get_position();  // BROKEN: Wall spawns in front, not at hero
                wall.width = 300.f;
                wall.end_time = yasuo_wall->get_end_time();
                wall.source_champion = "yasuo";
                windwalls.push_back(wall);
            }

            // Samira Blade Whirl
            std::string samira_buff = "samiraw";
            auto samira_wall = hero->get_buff_by_name(samira_buff);
            if (samira_wall && samira_wall->is_active())
            {
                WindwallInfo wall;
                wall.exists = true;
                wall.position = hero->get_position();
                wall.width = 325.f;
                wall.end_time = samira_wall->get_end_time();
                wall.source_champion = "samira";
                windwalls.push_back(wall);
            }

            // Braum Unbreakable
            std::string braum_buff = "braume";
            auto braum_shield = hero->get_buff_by_name(braum_buff);
            if (braum_shield && braum_shield->is_active())
            {
                WindwallInfo wall;
                wall.exists = true;
                wall.position = hero->get_position();
                wall.width = 200.f;
                wall.end_time = braum_shield->get_end_time();
                wall.source_champion = "braum";
                windwalls.push_back(wall);
            }
        }

        return windwalls;
        END DISABLED CODE */
    }

    /**
     * Check if projectile path intersects with any windwall
     * FIXED: Correct closest point calculation using dot product projection
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

            constexpr float MIN_SAFE_DISTANCE = 1.0f;  // Minimum safe distance for normalization
            if (distance_to_target < MIN_SAFE_DISTANCE)
                continue;  // Zero-length path, skip windwall check

            math::vector3 direction = to_target / distance_to_target;  // Normalized direction

            // FIXED: Project windwall position onto the projectile path using dot product
            math::vector3 start_to_wall = wall.position - start_pos;
            float projection = start_to_wall.dot(direction);

            // Clamp projection to line segment [0, distance_to_target]
            if (projection < 0.f || projection > distance_to_target)
                continue;  // Windwall is not between start and end

            // Calculate closest point on path and perpendicular distance to windwall
            math::vector3 closest_point_on_path = start_pos + direction * projection;
            float perpendicular_distance = (wall.position - closest_point_on_path).magnitude();

            // Check if windwall is close enough to block projectile
            if (perpendicular_distance < wall.width)
            {
                return true;  // Projectile will be blocked
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
            if (!minion || !minion->is_valid())
                continue;

            // Skip wards (don't block skillshots)
            std::string name = minion->get_char_name();
            if (name.find("Ward") != std::string::npos ||
                name.find("Trinket") != std::string::npos ||
                name.find("YellowTrinket") != std::string::npos)
                continue;

            // Check if minion is between source and target
            math::vector3 minion_pos = minion->get_position();
            math::vector3 to_minion = minion_pos - source_pos;
            float distance_along_path = to_minion.dot(direction);

            // Minion must be between source and target
            if (distance_along_path < 0.f || distance_along_path > distance_to_target)
                continue;

            // FIX: Predict minion position when spell arrives
            // Estimate travel time to minion position (assume ~1200 projectile speed if not specified)
            float travel_time = distance_along_path / 1200.f;

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
            float minion_radius = minion->get_bounding_radius();
            if (perpendicular_distance < minion_radius + projectile_width)
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

        // CONFIDENCE MULTIPLIERS
        if (analysis.stasis.is_in_stasis)
            analysis.confidence_multiplier *= 1.5f;  // Very confident on stasis exit

        if (analysis.channel.is_channeling || analysis.channel.is_recalling)
            analysis.confidence_multiplier *= 1.3f;  // Target is stationary

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