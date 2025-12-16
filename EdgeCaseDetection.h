#pragma once

#include "sdk.hpp"
#include "hp_sdk.hpp"  // Health prediction for ghost minion check
#include "StandalonePredictionSDK.h"  // MUST be included AFTER sdk.hpp for compatibility
#include "SDKCompatibility.h"  // SDK API compatibility layer
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
        // PING-SCALED BUFFER: Base 16ms + 50% of one-way ping
        // - Low ping (20ms): 16ms + 5ms = 21ms buffer
        // - Medium ping (60ms): 16ms + 15ms = 31ms buffer
        // - High ping (100ms): 16ms + 25ms = 41ms buffer
        // This accounts for jitter increasing with ping
        // =====================================================================
        constexpr float BASE_BUFFER = 0.016f;  // Base 16ms (1 frame @ 60Hz)

        // Get ping and calculate adaptive buffer
        float ping_buffer = BASE_BUFFER;
        if (g_sdk && g_sdk->net_client)
        {
            float ping_ms = static_cast<float>(g_sdk->net_client->get_ping());
            float one_way_ping = ping_ms / 2.0f;  // One-way delay

            // Add 50% of one-way ping as extra buffer (jitter increases with ping)
            ping_buffer = BASE_BUFFER + (one_way_ping / 1000.f) * 0.5f;

            // Cap at 50ms to avoid being too conservative
            ping_buffer = std::min(ping_buffer, 0.050f);
        }

        float optimal_cast_delay = time_until_exit - spell_travel_time + ping_buffer;

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
     * Untargetability information
     * Abilities that make champions completely immune to targeted spells
     */
    struct UntargetabilityInfo
    {
        bool is_untargetable;
        float duration_remaining;
        std::string ability_name;  // "fizz_e", "vlad_w", "yi_q", etc.

        UntargetabilityInfo() : is_untargetable(false), duration_remaining(0.f), ability_name("") {}
    };

    /**
     * Revive passive information
     * Detects if target will revive after death (GA, Zilean R, Anivia passive, etc.)
     */
    struct ReviveInfo
    {
        bool has_revive;
        std::string revive_type;  // "guardian_angel", "zilean_r", "anivia_passive", "zac_passive"

        ReviveInfo() : has_revive(false), revive_type("") {}
    };

    /**
     * Polymorph information
     * Polymorph disables abilities and limits movement control
     */
    struct PolymorphInfo
    {
        bool is_polymorphed;
        float duration_remaining;
        std::string source;  // "lulu_w", etc.

        PolymorphInfo() : is_polymorphed(false), duration_remaining(0.f), source("") {}
    };

    /**
     * Knockback/Displacement information
     * Active knockbacks make prediction difficult - wait for completion
     */
    struct KnockbackInfo
    {
        bool is_knocked_back;
        float duration_remaining;

        KnockbackInfo() : is_knocked_back(false), duration_remaining(0.f) {}
    };

    /**
     * Grounded information
     * Grounded targets cannot use movement abilities (Flash, dashes, blinks)
     * This is a HUGE confidence boost - they cannot escape except by walking!
     */
    struct GroundedInfo
    {
        bool is_grounded;
        float duration_remaining;
        std::string source;  // "cassiopeia_w", "singed_w", etc.

        GroundedInfo() : is_grounded(false), duration_remaining(0.f), source("") {}
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
     * Detect if target is untargetable (Fizz E, Vlad W, Yi Q, etc.)
     * Untargetable champions cannot be hit by ANY spells - don't cast!
     */
    inline UntargetabilityInfo detect_untargetability(game_object* target)
    {
        UntargetabilityInfo info;

        if (!target || !target->is_valid())
            return info;

        // First check SDK's is_targetable() if available
        // Note: Some SDKs have this, others don't - handle both cases
        if (!target->is_targetable())
        {
            info.is_untargetable = true;
            info.ability_name = "generic_untargetable";
            return info;
        }

        if (!g_sdk || !g_sdk->clock_facade)
            return info;

        float current_time = g_sdk->clock_facade->get_game_time();

        // Check for specific untargetability buffs
        auto check_buff = [&](const std::string& name, const std::string& ability)
        {
            std::string buff_name = name;  // Make mutable copy for API
            auto buff = target->get_buff_by_name(buff_name);
            if (buff && buff->is_active())
            {
                float end = buff->get_end_time();
                float duration_left = end - current_time;

                // Untargetability is typically short (< 3s for most abilities)
                if (duration_left > 0.f && duration_left < 5.0f)
                {
                    info.is_untargetable = true;
                    info.duration_remaining = duration_left;
                    info.ability_name = ability;
                    return true;
                }
            }
            return false;
        };

        // Fizz E (Playful/Trickster) - becomes untargetable
        if (check_buff("fizzejump", "fizz_e")) return info;
        if (check_buff("fizzetrickster", "fizz_e")) return info;

        // Vladimir W (Sanguine Pool) - becomes untargetable
        if (check_buff("vladimirw", "vlad_w")) return info;
        if (check_buff("vladimirsanguinepool", "vlad_w")) return info;

        // Master Yi Q (Alpha Strike) - becomes untargetable
        if (check_buff("alphastrike", "yi_q")) return info;
        if (check_buff("yialphastrike", "yi_q")) return info;

        // Elise E (Rappel) - becomes untargetable
        if (check_buff("elisespideredescent", "elise_e")) return info;
        if (check_buff("elisee", "elise_e")) return info;

        // Maokai W (Twisted Advance) - briefly untargetable during dash
        if (check_buff("maokaiwdash", "maokai_w")) return info;

        // Kayn R (Umbral Trespass) - untargetable inside enemy
        if (check_buff("kaynr", "kayn_r")) return info;
        if (check_buff("kaynrjumpinside", "kayn_r")) return info;

        // Xayah R (Featherstorm) - becomes untargetable
        if (check_buff("xayahr", "xayah_r")) return info;
        if (check_buff("xayahrknockup", "xayah_r")) return info;

        // Shaco R (briefly untargetable during cast)
        if (check_buff("shacorcast", "shaco_r")) return info;

        // Kled E (Jousting) - very brief untargetability
        if (check_buff("klede", "kled_e")) return info;

        // Camille R (very brief during dash)
        if (check_buff("camillerdash", "camille_r")) return info;

        return info;
    }

    /**
     * Detect if target has revive passive ready (GA, Zilean R, Anivia, Zac)
     * These targets will revive after death - affects target priority
     */
    inline ReviveInfo detect_revive(game_object* target)
    {
        ReviveInfo info;

        if (!target || !target->is_valid())
            return info;

        if (!g_sdk || !g_sdk->clock_facade)
            return info;

        float current_time = g_sdk->clock_facade->get_game_time();

        // Check for revive buffs
        auto check_buff = [&](const std::string& name, const std::string& type, float min_duration = 5.0f)
        {
            std::string buff_name = name;  // Make mutable copy for API
            auto buff = target->get_buff_by_name(buff_name);
            if (buff && buff->is_active())
            {
                float end = buff->get_end_time();
                float duration_left = end - current_time;

                // Revive passives have LONG duration when ready (> 5s)
                // This distinguishes "GA ready" from "GA reviving" (which is short stasis)
                if (duration_left > min_duration)
                {
                    info.has_revive = true;
                    info.revive_type = type;
                    return true;
                }
            }
            return false;
        };

        // Guardian Angel (item) - long duration "willrevive" buff when ready
        if (check_buff("willrevive", "guardian_angel", 5.0f)) return info;

        // Zilean R (Chronoshift) - prevents next death
        if (check_buff("chronoshift", "zilean_r", 3.0f)) return info;
        if (check_buff("zileanrrewind", "zilean_r", 3.0f)) return info;

        // Check champion-specific passives by champion name
        std::string char_name = target->get_char_name();

        // Anivia passive (Rebirth) - becomes egg then revives
        if (char_name.find("Anivia") != std::string::npos)
        {
            // Anivia passive is ready if she doesn't have "rebirthcooldown" buff
            std::string cooldown_buff = "rebirthcooldown";
            auto cd_buff = target->get_buff_by_name(cooldown_buff);
            if (!cd_buff || !cd_buff->is_active())
            {
                info.has_revive = true;
                info.revive_type = "anivia_passive";
                return info;
            }
        }

        // Zac passive (Cell Division) - splits into bloblets
        if (char_name.find("Zac") != std::string::npos)
        {
            // Zac passive is ready if he doesn't have cooldown buff
            std::string cooldown_buff = "zacrebirthready";
            auto ready_buff = target->get_buff_by_name(cooldown_buff);
            // "zacrebirthready" means passive is UP (confusing naming)
            if (ready_buff && ready_buff->is_active())
            {
                info.has_revive = true;
                info.revive_type = "zac_passive";
                return info;
            }
        }

        // Renata Glasc W (Bailout) - ally revive
        if (check_buff("renatawactive", "renata_w", 2.0f)) return info;

        // Sion passive (Glory in Death) - zombie form
        // Note: Sion passive always activates on death, no cooldown
        if (char_name.find("Sion") != std::string::npos)
        {
            // Check if not already dead/passive used
            if (!target->is_dead())
            {
                info.has_revive = true;
                info.revive_type = "sion_passive";
                return info;
            }
        }

        return info;
    }

    /**
     * Detect polymorph effects (Lulu W)
     * Polymorphed targets have limited movement control - easier to hit
     */
    inline PolymorphInfo detect_polymorph(game_object* target)
    {
        PolymorphInfo info;

        if (!target || !target->is_valid())
            return info;

        // Check for polymorph buff type
        if (target->has_buff_of_type(buff_type::polymorph))
        {
            info.is_polymorphed = true;

            if (g_sdk && g_sdk->clock_facade)
            {
                float current_time = g_sdk->clock_facade->get_game_time();

                // Try to find the specific polymorph buff for duration
                auto buffs = target->get_buffs();
                for (auto* buff : buffs)
                {
                    if (buff && buff->is_active() && buff->get_type() == buff_type::polymorph)
                    {
                        info.duration_remaining = std::max(0.f, buff->get_end_time() - current_time);

                        // Try to identify source
                        std::string buff_name = buff->get_name();
                        if (buff_name.find("lulu") != std::string::npos || buff_name.find("Lulu") != std::string::npos)
                            info.source = "lulu_w";
                        else
                            info.source = "unknown";

                        break;
                    }
                }
            }
        }

        return info;
    }

    /**
     * Detect active knockback/displacement effects
     * Knockbacks are brief but make prediction unreliable
     */
    inline KnockbackInfo detect_knockback(game_object* target)
    {
        KnockbackInfo info;

        if (!target || !target->is_valid())
            return info;

        // Check for knockback buff type
        if (target->has_buff_of_type(buff_type::knockback) ||
            target->has_buff_of_type(buff_type::knockup))
        {
            info.is_knocked_back = true;

            if (g_sdk && g_sdk->clock_facade)
            {
                float current_time = g_sdk->clock_facade->get_game_time();

                // Find knockback buff for duration
                auto buffs = target->get_buffs();
                for (auto* buff : buffs)
                {
                    if (buff && buff->is_active())
                    {
                        auto type = buff->get_type();
                        if (type == buff_type::knockback || type == buff_type::knockup)
                        {
                            info.duration_remaining = std::max(0.f, buff->get_end_time() - current_time);
                            break;
                        }
                    }
                }
            }
        }

        return info;
    }

    /**
     * Detect grounded buff (Cassiopeia W, Singed W)
     * CRITICAL: Grounded targets CANNOT use movement abilities!
     * This includes: Flash, all dashes, all blinks
     * Confidence boost: They can ONLY escape by walking
     */
    inline GroundedInfo detect_grounded(game_object* target)
    {
        GroundedInfo info;

        if (!target || !target->is_valid())
            return info;

        if (!g_sdk || !g_sdk->clock_facade)
            return info;

        float current_time = g_sdk->clock_facade->get_game_time();

        // Check for specific grounded buffs
        auto check_buff = [&](const std::string& name, const std::string& source)
        {
            std::string buff_name = name;  // Make mutable copy for API
            auto buff = target->get_buff_by_name(buff_name);
            if (buff && buff->is_active())
            {
                float end = buff->get_end_time();
                float duration_left = end - current_time;

                // Grounded effects are typically short (< 5s)
                if (duration_left > 0.f && duration_left < 6.0f)
                {
                    info.is_grounded = true;
                    info.duration_remaining = duration_left;
                    info.source = source;
                    return true;
                }
            }
            return false;
        };

        // Cassiopeia W (Miasma) - 5 second duration
        if (check_buff("cassiopeiamiasma", "cassiopeia_w")) return info;
        if (check_buff("cassiopeiaw", "cassiopeia_w")) return info;

        // Singed W (Mega Adhesive) - 3 second duration
        if (check_buff("megaadhesive", "singed_w")) return info;
        if (check_buff("singedw", "singed_w")) return info;

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
     *
     * Note: Slow decay is automatically handled by using real-time move speed.
     * Decaying slows (99% -> 20% over time) are captured via current_speed check.
     */
    inline bool is_slowed(game_object* target)
    {
        if (!target || !target->is_valid())
            return false;

        // Method 1: Check for slow buff type
        if (target->has_buff_of_type(buff_type::slow))
            return true;

        // Method 2: Compare current vs base move speed
        // This automatically accounts for slow decay since we check real-time speed
        float current_speed = target->get_move_speed();
        float base_speed = get_base_move_speed(target);

        // Consider slowed if moving < 95% of base speed
        return current_speed < base_speed * 0.95f;
    }

    // =========================================================================
    // CSING DETECTION (Last-Hit Attention)
    // =========================================================================

    /**
     * Detect if target is likely last-hitting a minion
     *
     * Indicators:
     * - Low HP minion in attack range
     * - Minion killable in 1-2 autos
     * - No immediate threat (not being harassed)
     *
     * When CSing, attention is divided = slightly worse dodging
     */
    inline bool is_likely_csing(game_object* target)
    {
        if (!target || !target->is_valid() || !g_sdk || !g_sdk->object_manager)
            return false;

        // Get target's attack range
        float attack_range = target->get_attack_range();
        math::vector3 target_pos = target->get_position();

        // Look for low-HP minions nearby
        auto minions = g_sdk->object_manager->get_minions();
        for (auto* minion : minions)
        {
            if (!minion || !minion->is_valid() || minion->is_dead())
                continue;

            // Only check enemy minions (target would CS these)
            if (minion->get_team_id() == target->get_team_id())
                continue;

            // Check if minion is in attack range
            float dist_to_minion = (minion->get_position() - target_pos).magnitude();
            if (dist_to_minion > attack_range + 100.f)  // +100 buffer for moving into range
                continue;

            // Check if minion is low HP (in last-hit range)
            float minion_hp = minion->get_hp();
            float minion_max_hp = minion->get_max_hp();
            float hp_percent = minion_hp / std::max(1.f, minion_max_hp);

            // Minion below 30% HP = likely CS target
            if (hp_percent > 0.3f)
                continue;

            // Additionally check if target can kill it in 1-2 autos
            float target_ad = target->get_attack_damage();
            if (minion_hp < target_ad * 2.5f)  // Killable in ~2 autos
            {
                // Found low-HP minion in range = likely CSing
                return true;
            }
        }

        return false;
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
    // TERRAIN CREATION DETECTION (Anivia W, Jarvan R, Trundle E, Azir R)
    // =========================================================================

    /**
     * Terrain information
     * Terrain restricts enemy movement and can trap them
     */
    struct TerrainInfo
    {
        bool exists;
        math::vector3 position;
        float radius;                  // Affected area radius
        float end_time;
        std::string source_champion;   // "anivia", "jarvan", "trundle", "azir"
        bool blocks_projectiles;       // true for Azir R, false for most terrain

        TerrainInfo() : exists(false), position{}, radius(0.f),
            end_time(0.f), source_champion(""), blocks_projectiles(false) {
        }
    };

    /**
     * Detect active terrain that restricts movement
     * Terrain can trap enemies or block pathing, increasing hit chance
     */
    inline std::vector<TerrainInfo> detect_terrain()
    {
        std::vector<TerrainInfo> terrains;

        if (!g_sdk || !g_sdk->object_manager)
            return terrains;

        auto* local_player = g_sdk->object_manager->get_local_player();
        if (!local_player || !local_player->is_valid())
            return terrains;

        int local_team = local_player->get_team_id();
        float current_time = 0.f;
        if (g_sdk->clock_facade)
            current_time = g_sdk->clock_facade->get_game_time();

        // Object names for terrain-creating abilities
        static const char* ANIVIA_WALL_NAMES[] = {
            "Anivia_Base_W_Wall",
            "AniviaWall"
        };

        static const char* JARVAN_ARENA_NAMES[] = {
            "Jarvan_Base_R_TerraArena",
            "JarvanCataclysm"
        };

        static const char* TRUNDLE_PILLAR_NAMES[] = {
            "Trundle_Base_E_Pillar",
            "TrundlePillar"
        };

        static const char* AZIR_WALL_NAMES[] = {
            "Azir_Base_R_SoldierWall",
            "AzirWall"
        };

        static const char* TALIYAH_WALL_NAMES[] = {
            "Taliyah_Base_R_Wall",
            "TaliyahWall"
        };

        static const char* VEIGAR_CAGE_NAMES[] = {
            "Veigar_Base_E_Cage",
            "VeigarEventHorizon"
        };

        // Search through minions/objects for terrain entities
        auto minions = g_sdk->object_manager->get_minions();
        for (auto* obj : minions)
        {
            if (!obj || !obj->is_valid() || !obj->is_visible())
                continue;

            // Skip allied terrain (unless we also want to track it)
            if (obj->get_team_id() == local_team)
                continue;

            std::string name = obj->get_name();
            if (name.empty())
                continue;

            TerrainInfo terrain;
            terrain.exists = false;

            // Check for Anivia W (Crystallize)
            for (const char* terrain_name : ANIVIA_WALL_NAMES)
            {
                if (name.find(terrain_name) != std::string::npos)
                {
                    terrain.exists = true;
                    terrain.position = obj->get_position();
                    terrain.radius = 400.f;  // Wall width
                    terrain.end_time = current_time + 5.0f;  // 5 second duration
                    terrain.source_champion = "anivia";
                    terrain.blocks_projectiles = false;
                    break;
                }
            }

            // Check for Jarvan R (Cataclysm)
            if (!terrain.exists)
            {
                for (const char* terrain_name : JARVAN_ARENA_NAMES)
                {
                    if (name.find(terrain_name) != std::string::npos)
                    {
                        terrain.exists = true;
                        terrain.position = obj->get_position();
                        terrain.radius = 325.f;  // Arena radius
                        terrain.end_time = current_time + 3.5f;  // 3.5 second duration
                        terrain.source_champion = "jarvan";
                        terrain.blocks_projectiles = false;
                        break;
                    }
                }
            }

            // Check for Trundle E (Pillar of Ice)
            if (!terrain.exists)
            {
                for (const char* terrain_name : TRUNDLE_PILLAR_NAMES)
                {
                    if (name.find(terrain_name) != std::string::npos)
                    {
                        terrain.exists = true;
                        terrain.position = obj->get_position();
                        terrain.radius = 188.f;  // Pillar radius
                        terrain.end_time = current_time + 6.0f;  // 6 second duration
                        terrain.source_champion = "trundle";
                        terrain.blocks_projectiles = false;
                        break;
                    }
                }
            }

            // Check for Azir R (Emperor's Divide) - BLOCKS PROJECTILES
            if (!terrain.exists)
            {
                for (const char* terrain_name : AZIR_WALL_NAMES)
                {
                    if (name.find(terrain_name) != std::string::npos)
                    {
                        terrain.exists = true;
                        terrain.position = obj->get_position();
                        terrain.radius = 520.f;  // Wall length
                        terrain.end_time = current_time + 5.0f;  // 5 second duration
                        terrain.source_champion = "azir";
                        terrain.blocks_projectiles = true;  // Azir wall blocks projectiles!
                        break;
                    }
                }
            }

            // Check for Taliyah R (Weaver's Wall)
            if (!terrain.exists)
            {
                for (const char* terrain_name : TALIYAH_WALL_NAMES)
                {
                    if (name.find(terrain_name) != std::string::npos)
                    {
                        terrain.exists = true;
                        terrain.position = obj->get_position();
                        terrain.radius = 2000.f;  // Very long wall
                        terrain.end_time = current_time + 5.0f;  // 5 second duration
                        terrain.source_champion = "taliyah";
                        terrain.blocks_projectiles = true;  // Taliyah wall blocks projectiles
                        break;
                    }
                }
            }

            // Check for Veigar E (Event Horizon) - not true terrain but restricts movement
            if (!terrain.exists)
            {
                for (const char* terrain_name : VEIGAR_CAGE_NAMES)
                {
                    if (name.find(terrain_name) != std::string::npos)
                    {
                        terrain.exists = true;
                        terrain.position = obj->get_position();
                        terrain.radius = 375.f;  // Cage radius
                        terrain.end_time = current_time + 3.0f;  // 3 second duration
                        terrain.source_champion = "veigar";
                        terrain.blocks_projectiles = false;
                        break;
                    }
                }
            }

            if (terrain.exists)
            {
                terrains.push_back(terrain);
            }
        }

        return terrains;
    }

    /**
     * Check if target is trapped by terrain (limited escape options)
     * Returns confidence boost if target is near terrain
     */
    inline float get_terrain_confidence_boost(
        game_object* target,
        const std::vector<TerrainInfo>& terrains)
    {
        if (!target || !target->is_valid() || terrains.empty())
            return 1.0f;  // No boost

        math::vector3 target_pos = target->get_position();
        float min_distance = 999999.f;

        for (const auto& terrain : terrains)
        {
            if (!terrain.exists)
                continue;

            float dist = (target_pos - terrain.position).magnitude();

            // If target is inside/near terrain, they have restricted movement
            if (dist < terrain.radius + 200.f)  // Within terrain + buffer
            {
                min_distance = std::min(min_distance, dist);
            }
        }

        // Closer to terrain = more restricted movement = higher confidence
        if (min_distance < 300.f)
            return 1.3f;  // 30% boost - very restricted
        else if (min_distance < 600.f)
            return 1.15f;  // 15% boost - somewhat restricted
        else
            return 1.0f;  // No boost
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
     * Windwall cache for performance optimization
     * Windwall detection scans all minions - cache results to avoid repeated scans
     */
    struct WindwallCache
    {
        std::vector<WindwallInfo> cached_windwalls;
        float last_update_time;
        constexpr static float CACHE_DURATION = 0.1f;  // 100ms cache (refresh 10x/sec)

        WindwallCache() : last_update_time(-999.f) {}

        bool is_valid(float current_time) const
        {
            return (current_time - last_update_time) < CACHE_DURATION;
        }

        void update(const std::vector<WindwallInfo>& windwalls, float current_time)
        {
            cached_windwalls = windwalls;
            last_update_time = current_time;
        }
    };

    // Global windwall cache (static for persistence across calls)
    static WindwallCache g_windwall_cache;

    /**
     * Get Yasuo W width based on ability rank
     * Width scales: 320 / 390 / 460 / 530 / 600
     *
     * NOTE: Patch 14.X values - may need update if Riot changes ability scaling
     * SDK doesn't expose spell width directly, so we use level-based lookup
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
                // Patch 14.X - Yasuo W width per level
                static const float widths[] = { 320.f, 390.f, 460.f, 530.f, 600.f };
                return widths[level - 1];
            }
        }
        return 600.f;  // Default to max width to be safe
    }

    /**
     * Detect active windwalls that can block projectiles
     * Uses game object names from actual game data dumps
     * PERFORMANCE: Cached for 100ms to avoid repeated minion scans
     */
    inline std::vector<WindwallInfo> detect_windwalls()
    {
        // Get current time first for cache validation
        float current_time = 0.f;
        if (g_sdk && g_sdk->clock_facade)
            current_time = g_sdk->clock_facade->get_game_time();

        // Check cache validity
        if (g_windwall_cache.is_valid(current_time))
        {
            return g_windwall_cache.cached_windwalls;
        }

        // Cache miss - perform full detection
        std::vector<WindwallInfo> windwalls;

        if (!g_sdk || !g_sdk->object_manager)
            return windwalls;

        auto* local_player = g_sdk->object_manager->get_local_player();
        if (!local_player || !local_player->is_valid())
            return windwalls;

        int local_team = local_player->get_team_id();

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

        // Update cache with new results
        g_windwall_cache.update(windwalls, current_time);

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
                constexpr float LANE_BASE_DPS = 50.f;   // Base lane DPS (fallback)
                float estimated_dps = LANE_BASE_DPS;

                // Higher DPS if minion is under tower (tower shots = massive burst)
                // SDK LIMITATION: get_enemy_turrets() not available - conservative estimate
                if (false && g_sdk && g_sdk->object_manager)
                {
                    // Disabled: SDK doesn't provide get_enemy_turrets() method
                    // Using conservative minion DPS estimate instead
                    std::vector<game_object*> towers;  // Empty - no tower check
                    for (auto* tower : towers)
                    {
                        if (tower && tower->is_valid() && !tower->is_dead())
                        {
                            float tower_range = tower->get_attack_range();  // Actual tower range from SDK
                            float dist_to_tower = (minion->get_position() - tower->get_position()).magnitude();

                            if (dist_to_tower < tower_range)
                            {
                                // Calculate actual tower DPS from SDK
                                float tower_damage = tower->get_attack_damage();
                                float tower_attack_delay = tower->get_attack_delay();
                                float tower_dps = (tower_attack_delay > 0.f) ? (tower_damage / tower_attack_delay) : 250.f;

                                estimated_dps = tower_dps;
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

            // Check if minion hitbox overlaps projectile using actual bounding boxes
            // Already accounted for minion movement prediction above
            float minion_radius = minion->get_bounding_radius();
            float collision_radius = minion_radius + projectile_width;

            if (perpendicular_distance < collision_radius)
            {
                blocking_minions++;
            }
        }

        // Deterministic collision: either blocked or clear
        // Geometry check already accounts for hitboxes and predicted movement
        if (blocking_minions > 0)
            return 0.0f;  // Blocked by minion(s)

        return 1.0f;  // Clear path
    }

    // =========================================================================
    // CLONE DETECTION
    // =========================================================================

    // =========================================================================
    // JUKE / ERRATIC MOVEMENT DETECTION
    // =========================================================================

    /**
     * Juke detection via velocity direction variance
     * Tracks recent movement patterns to detect zigzag / erratic dodging
     */
    struct JukeInfo
    {
        bool is_juking;                   // Currently exhibiting juke behavior
        bool is_oscillating;              // Periodic pattern (left-right-left) vs random jitter
        float direction_variance;         // Variance in movement direction (0-1, higher = more erratic)
        float confidence_penalty;         // Hit chance reduction factor (0.9-1.0)
        math::vector3 predicted_velocity; // Average velocity (juking) or current (straight movement)

        JukeInfo() : is_juking(false), is_oscillating(false), direction_variance(0.f),
                     confidence_penalty(1.0f), predicted_velocity{} {}
    };

    /**
     * Per-target movement history for juke detection
     * Tracks last N velocity samples to detect direction changes
     */
    struct MovementHistory
    {
        static constexpr int MAX_SAMPLES = 30;  // Last 30 samples (~1.5s window for 5+ juke cycles)
        static constexpr int MIN_SAMPLES_FOR_PREDICTION = 8;  // Need at least 400ms for initial detection
        static constexpr float SAMPLE_INTERVAL = 0.05f;  // 50ms between samples

        std::vector<math::vector3> velocity_samples;
        float last_sample_time;
        float last_velocity_magnitude;  // Track previous magnitude for dash detection

        MovementHistory() : last_sample_time(-999.f), last_velocity_magnitude(0.f) {}

        // Clear all history (called on vision loss, CC, death)
        void clear()
        {
            velocity_samples.clear();
            last_sample_time = -999.f;
            last_velocity_magnitude = 0.f;
        }

        void update(const math::vector3& velocity, float current_time)
        {
            // Only sample every 50ms to avoid noise
            if (current_time - last_sample_time < SAMPLE_INTERVAL)
                return;

            // DASH FILTER: Detect and skip dash/blink samples
            // Dashes have massive velocity spikes that create false reversals
            float velocity_magnitude = std::sqrt(velocity.x * velocity.x + velocity.z * velocity.z);

            if (last_velocity_magnitude > 0.f)
            {
                // Velocity spike detection:
                // - Absolute speed > 1500 u/s = likely dash
                // - Speed increased by >2x = likely dash start
                // - Speed decreased by >2x = likely dash end
                bool is_dash_speed = velocity_magnitude > 1500.f;
                bool is_dash_accel = velocity_magnitude > last_velocity_magnitude * 2.0f;
                bool is_dash_decel = last_velocity_magnitude > velocity_magnitude * 2.0f;

                if (is_dash_speed || is_dash_accel || is_dash_decel)
                {
                    // Skip this sample - it's a dash/blink artifact
                    last_velocity_magnitude = velocity_magnitude;
                    last_sample_time = current_time;
                    return;
                }
            }

            // Valid sample - add to history
            velocity_samples.push_back(velocity);
            last_sample_time = current_time;
            last_velocity_magnitude = velocity_magnitude;

            // Keep only last N samples
            if (velocity_samples.size() > MAX_SAMPLES)
                velocity_samples.erase(velocity_samples.begin());
        }

        // Calculate AVERAGE velocity (center of oscillation pattern)
        // This is where juking targets spend their time on average
        math::vector3 calculate_average_velocity() const
        {
            if (velocity_samples.empty())
                return math::vector3(0.f, 0.f, 0.f);

            math::vector3 sum(0.f, 0.f, 0.f);
            for (const auto& vel : velocity_samples)
            {
                sum.x += vel.x;
                sum.z += vel.z;
            }

            float count = static_cast<float>(velocity_samples.size());
            return math::vector3(sum.x / count, 0.f, sum.z / count);
        }

        // Calculate direction variance (0 = straight line, 1 = completely erratic)
        float calculate_direction_variance() const
        {
            if (velocity_samples.size() < 3)
                return 0.f;  // Not enough data

            // Calculate average direction
            math::vector3 avg_direction(0.f, 0.f, 0.f);
            int valid_samples = 0;

            for (const auto& vel : velocity_samples)
            {
                float mag = std::sqrt(vel.x * vel.x + vel.z * vel.z);
                if (mag > 10.f)  // Only consider significant movement (>10 u/s)
                {
                    avg_direction.x += vel.x / mag;
                    avg_direction.z += vel.z / mag;
                    valid_samples++;
                }
            }

            if (valid_samples == 0)
                return 0.f;

            avg_direction.x /= valid_samples;
            avg_direction.z /= valid_samples;

            float avg_mag = std::sqrt(avg_direction.x * avg_direction.x + avg_direction.z * avg_direction.z);

            // avg_mag close to 1.0 = consistent direction
            // avg_mag close to 0.0 = directions cancel out (zigzag)
            return SDKCompat::clamp(1.f - avg_mag, 0.f, 1.f);
        }

        // Detect alternating oscillation pattern (left-right-left) vs random jitter
        // Oscillating patterns are MORE predictable (aim at center)
        // Now includes period validation - must be in human juke range (150-600ms)
        bool is_oscillating_pattern() const
        {
            if (velocity_samples.size() < 4)
                return false;

            // Check last 4 samples for alternating pattern
            int alternations = 0;
            std::vector<float> reversal_times;

            for (size_t i = 0; i < velocity_samples.size() - 1 && i < 3; ++i)
            {
                const auto& v1 = velocity_samples[velocity_samples.size() - 1 - i];
                const auto& v2 = velocity_samples[velocity_samples.size() - 2 - i];

                float dot = v1.x * v2.x + v1.z * v2.z;

                // Opposite directions (negative dot product) = reversal
                if (dot < 0.f)
                {
                    alternations++;
                    float time_ago = i * SAMPLE_INTERVAL;
                    reversal_times.push_back(time_ago);
                }
            }

            // At least 2 out of 3 transitions are reversals = oscillating
            if (alternations < 2)
                return false;

            // PERIOD VALIDATION: Check if period is in human juke range
            // True juking has 150-600ms period (half-period = 75-300ms)
            // Pathing has much longer periods (1-3 seconds)
            if (reversal_times.size() >= 2)
            {
                float half_period = reversal_times[0] - reversal_times[1];

                constexpr float MIN_JUKE_HALF_PERIOD = 0.075f;  // 75ms (very fast juking)
                constexpr float MAX_JUKE_HALF_PERIOD = 0.35f;   // 350ms (slow juking)

                // Period outside human juke range = probably pathing or context change
                if (half_period < MIN_JUKE_HALF_PERIOD || half_period > MAX_JUKE_HALF_PERIOD)
                    return false;
            }

            return true;
        }

        // ADVANCED: Predict position at future time by extrapolating oscillation
        // Uses weighted period, amplitude detection, and pattern quality
        math::vector3 predict_position_with_oscillation(float delta_time, const math::vector3& current_pos,
                                                         const math::vector3& current_velocity) const
        {
            if (!is_oscillating_pattern() || velocity_samples.size() < 4)
            {
                // Fall back to average velocity * time
                return current_pos + calculate_average_velocity() * delta_time;
            }

            // Find reversal times and calculate adaptive weighted period
            std::vector<float> reversal_times;
            std::vector<float> periods;

            for (size_t i = 0; i < velocity_samples.size() - 1; ++i)
            {
                const auto& v1 = velocity_samples[i];
                const auto& v2 = velocity_samples[i + 1];
                float dot = v1.x * v2.x + v1.z * v2.z;

                if (dot < 0.f)  // Reversal detected
                {
                    float time_ago = (velocity_samples.size() - 1 - i) * SAMPLE_INTERVAL;
                    reversal_times.push_back(time_ago);
                }
            }

            if (reversal_times.size() < 2)
                return current_pos + calculate_average_velocity() * delta_time;

            // Calculate periods between reversals
            for (size_t i = 0; i < reversal_times.size() - 1; ++i)
            {
                periods.push_back(reversal_times[i] - reversal_times[i + 1]);
            }

            // IMPROVEMENT 1: ADAPTIVE WEIGHTED PERIOD
            // Recent reversals weighted more heavily (exponential decay)
            float weighted_period = 0.f;
            float total_weight = 0.f;
            for (size_t i = 0; i < periods.size(); ++i)
            {
                // More recent = higher weight (0.5, 0.3, 0.2, etc.)
                float weight = std::pow(0.6f, static_cast<float>(i));
                weighted_period += periods[i] * weight;
                total_weight += weight;
            }
            weighted_period /= total_weight;

            // IMPROVEMENT 2: PATTERN QUALITY SCORING
            // Calculate period variance (consistency)
            float period_variance = 0.f;
            float avg_period = 0.f;
            for (float p : periods)
                avg_period += p;
            avg_period /= periods.size();

            for (float p : periods)
            {
                float diff = p - avg_period;
                period_variance += diff * diff;
            }
            period_variance = std::sqrt(period_variance / periods.size());

            // Pattern quality: 0 = perfect regularity, 1 = highly irregular
            float irregularity = std::min(period_variance / avg_period, 1.0f);

            // If pattern is irregular, blend toward average velocity
            if (irregularity > 0.3f)
            {
                // High irregularity - use hybrid between extrapolation and average
                float blend = irregularity;  // 0.3-1.0
                math::vector3 extrap_pos = current_pos + current_velocity * delta_time;
                math::vector3 avg_pos = current_pos + calculate_average_velocity() * delta_time;

                return math::vector3(
                    extrap_pos.x * (1.f - blend) + avg_pos.x * blend,
                    extrap_pos.y * (1.f - blend) + avg_pos.y * blend,
                    extrap_pos.z * (1.f - blend) + avg_pos.z * blend
                );
            }

            // Prevent division by zero
            if (weighted_period < 0.01f)
                return current_pos + calculate_average_velocity() * delta_time;

            // Time since last reversal
            float time_since_reversal = reversal_times[0];
            float time_to_next_reversal = weighted_period - time_since_reversal;

            // Integrate position through reversals (INSTANT DIRECTION CHANGES)
            // League has NO turn rate - direction changes are instant!
            math::vector3 position = current_pos;
            math::vector3 velocity = current_velocity;
            float remaining_time = delta_time;

            // First segment: travel until next reversal
            float dt = std::min(remaining_time, time_to_next_reversal);
            position.x += velocity.x * dt;
            position.z += velocity.z * dt;
            remaining_time -= dt;

            // Continue through reversals (instant velocity flip)
            while (remaining_time > 0.001f)
            {
                // Instant reversal (no acceleration/deceleration in League)
                velocity.x = -velocity.x;
                velocity.z = -velocity.z;

                // Integrate next segment (full period or remaining time)
                dt = std::min(remaining_time, weighted_period);
                position.x += velocity.x * dt;
                position.z += velocity.z * dt;
                remaining_time -= dt;
            }

            return position;
        }

        // Calculate pattern quality score (0 = random, 1 = perfect oscillation)
        float get_pattern_quality() const
        {
            if (!is_oscillating_pattern() || velocity_samples.size() < 4)
                return 0.f;

            // Find reversal times
            std::vector<float> periods;
            float prev_reversal_time = -1.f;

            for (size_t i = 0; i < velocity_samples.size() - 1; ++i)
            {
                const auto& v1 = velocity_samples[i];
                const auto& v2 = velocity_samples[i + 1];
                float dot = v1.x * v2.x + v1.z * v2.z;

                if (dot < 0.f)
                {
                    float time_ago = (velocity_samples.size() - 1 - i) * SAMPLE_INTERVAL;
                    if (prev_reversal_time >= 0.f)
                    {
                        periods.push_back(prev_reversal_time - time_ago);
                    }
                    prev_reversal_time = time_ago;
                }
            }

            if (periods.size() < 2)
                return 0.f;  // Not enough data = unknown quality = assume worst

            // Calculate coefficient of variation (CV) of periods
            float mean = 0.f;
            for (float p : periods)
                mean += p;
            mean /= periods.size();

            float variance = 0.f;
            for (float p : periods)
            {
                float diff = p - mean;
                variance += diff * diff;
            }
            variance /= periods.size();
            float stddev = std::sqrt(variance);

            // Coefficient of variation (lower = more consistent)
            float cv = stddev / std::max(mean, 0.01f);

            // Quality score: 1.0 = perfect (CV=0), 0.0 = random (CV>0.5)
            float quality = SDKCompat::clamp(1.0f - (cv * 2.0f), 0.f, 1.f);

            return quality;
        }
    };

    // Global movement history cache (per target)
    static std::unordered_map<uint32_t, MovementHistory> g_movement_history;

    /**
     * Predict position for juking target using oscillation extrapolation
     * Returns predicted position accounting for juke reversals
     */
    inline math::vector3 predict_juking_position(game_object* target, float prediction_time)
    {
        if (!target || !target->is_valid())
            return target ? target->get_server_position() : math::vector3(0.f, 0.f, 0.f);

        uint32_t target_id = target->get_network_id();
        auto it = g_movement_history.find(target_id);

        if (it == g_movement_history.end())
        {
            // No history - use current position + velocity
            math::vector3 pos = target->get_server_position();
            math::vector3 vel = target->get_velocity();
            return pos + vel * prediction_time;
        }

        const auto& history = it->second;
        math::vector3 current_pos = target->get_server_position();
        math::vector3 current_vel = target->get_velocity();

        // Use oscillation extrapolation if available
        return history.predict_position_with_oscillation(prediction_time, current_pos, current_vel);
    }

    /**
     * Calculate adaptive juke detection threshold based on game context
     * Higher threshold = harder to detect (fewer false positives)
     * Lower threshold = easier to detect (catch subtle juking)
     */
    inline float get_adaptive_juke_threshold(game_object* target)
    {
        // Base threshold - starts at 0.5 (moderate sensitivity)
        float threshold = 0.5f;

        // CONTEXT 1: Combat state
        // Out of combat = variance likely from pathing, not dodging
        // SDK doesn't have is_in_combat() - use conservative assumption (out of combat)
        bool in_combat = SDKCompat::is_in_combat(target);
        if (in_combat)
        {
            threshold -= 0.05f;  // In combat → easier to detect (0.45)
        }
        else
        {
            threshold += 0.20f;  // Out of combat → harder to detect (0.70)
        }

        // CONTEXT 2: Movement speed
        // Slow champions physically can't juke as effectively
        float move_speed = target->get_move_speed();
        if (move_speed < 350.f)
        {
            threshold += 0.10f;  // Slow → harder to detect
        }
        else if (move_speed > 450.f)
        {
            threshold -= 0.05f;  // Fast → easier to detect
        }

        // CONTEXT 3: Health pressure
        // Low HP in combat = more likely to be dodging desperately
        float health_percent = SDKCompat::get_health_percent(target);
        if (in_combat && health_percent < 0.3f)
        {
            threshold -= 0.10f;  // Low HP → easier to detect
        }

        // Clamp to reasonable range
        return SDKCompat::clamp(threshold, 0.35f, 0.80f);
    }

    /**
     * Detect if target is juking (zigzag erratic movement to dodge skillshots)
     * Uses velocity direction variance + returns AVERAGE velocity for prediction
     * Now with context-aware threshold and stale data prevention
     */
    inline JukeInfo detect_juke(game_object* target)
    {
        JukeInfo info;

        if (!target || !target->is_valid() || !g_sdk || !g_sdk->clock_facade)
            return info;

        uint32_t target_id = target->get_network_id();
        float current_time = g_sdk->clock_facade->get_game_time();

        // Get or create movement history for this target
        auto& history = g_movement_history[target_id];

        // STALE DATA CHECK #1: Clear history if target is CC'd
        // CC changes behavior context - old juke patterns no longer relevant
        bool is_ccd = target->has_buff_of_type(buff_type::stun) ||
                      target->has_buff_of_type(buff_type::snare) ||
                      target->has_buff_of_type(buff_type::charm) ||
                      target->has_buff_of_type(buff_type::fear) ||
                      target->has_buff_of_type(buff_type::taunt) ||
                      target->has_buff_of_type(buff_type::suppression) ||
                      target->has_buff_of_type(buff_type::knockup) ||
                      target->has_buff_of_type(buff_type::knockback);

        if (is_ccd)
        {
            history.clear();
            info.is_juking = false;
            info.predicted_velocity = target->get_velocity();
            info.confidence_penalty = 1.0f;
            return info;
        }

        // STALE DATA CHECK #2: Clear history if target is dead or recalling
        // Dead/recalling = major behavior reset
        if (target->is_dead() || SDKCompat::has_buff(target, "recall") || SDKCompat::has_buff(target, "teleport_target"))
        {
            history.clear();
            info.is_juking = false;
            info.predicted_velocity = target->get_velocity();
            info.confidence_penalty = 1.0f;
            return info;
        }

        // STALE DATA CHECK #3: Clear history if target not visible
        // Vision loss = can't trust old movement data
        if (!target->is_visible())
        {
            history.clear();
            info.is_juking = false;
            info.predicted_velocity = target->get_velocity();
            info.confidence_penalty = 1.0f;
            return info;
        }

        // Update history with current velocity (includes dash filtering)
        math::vector3 current_velocity = target->get_velocity();
        history.update(current_velocity, current_time);

        // Calculate direction variance
        float variance = history.calculate_direction_variance();
        info.direction_variance = variance;

        // ADAPTIVE THRESHOLD: Context-aware juke detection
        // In combat + fast + low HP → threshold 0.35 (sensitive)
        // Out of combat + slow → threshold 0.70 (conservative)
        float juke_threshold = get_adaptive_juke_threshold(target);

        if (variance > juke_threshold)
        {
            info.is_juking = true;

            // Check if oscillating pattern (left-right-left) vs random jitter
            info.is_oscillating = history.is_oscillating_pattern();

            // USE AVERAGE VELOCITY for prediction (center of oscillation)
            // This is the KEY FIX - we aim at where they ARE on average, not current direction
            info.predicted_velocity = history.calculate_average_velocity();

            // SAMPLE-COUNT-BASED CONFIDENCE SCALING
            // More samples = more juke cycles observed = higher confidence
            // 8 samples (400ms) = minimum for detection, but low confidence (2 cycles)
            // 15 samples (750ms) = medium confidence (3-4 cycles)
            // 30 samples (1500ms) = high confidence (5-7 cycles)
            float sample_confidence_multiplier = 1.0f;
            int num_samples = history.velocity_samples.size();

            if (num_samples < MovementHistory::MIN_SAMPLES_FOR_PREDICTION)
            {
                // Not enough data yet - very low confidence
                sample_confidence_multiplier = 0.70f;
            }
            else if (num_samples < MovementHistory::MAX_SAMPLES)
            {
                // Gradually increase confidence as we observe more cycles
                // 8 samples → 0.85x (cautious)
                // 15 samples → 0.925x (getting confident)
                // 30 samples → 1.0x (full confidence)
                float progress = (num_samples - MovementHistory::MIN_SAMPLES_FOR_PREDICTION) /
                                 (float)(MovementHistory::MAX_SAMPLES - MovementHistory::MIN_SAMPLES_FOR_PREDICTION);
                sample_confidence_multiplier = 0.85f + (0.15f * progress);
            }
            // else: num_samples >= MAX_SAMPLES → multiplier stays 1.0x (full confidence)

            // INTELLIGENT CONFIDENCE PENALTY using pattern quality
            if (info.is_oscillating)
            {
                // Get pattern quality (1.0 = perfect, 0.0 = random)
                float pattern_quality = history.get_pattern_quality();

                // ADAPTIVE PENALTY based on pattern quality:
                // Perfect pattern (quality 1.0): variance 0.8 → 0.992x (0.8% penalty)
                // Good pattern (quality 0.7): variance 0.8 → 0.982x (1.8% penalty)
                // Mediocre pattern (quality 0.5): variance 0.8 → 0.976x (2.4% penalty)
                // Poor pattern (quality 0.3): variance 0.8 → 0.970x (3.0% penalty)

                // Base penalty from variance (max 3%)
                float base_penalty = variance * 0.03f;

                // Reduce penalty based on pattern quality
                // Perfect pattern → 70% penalty reduction
                // Poor pattern → 0% penalty reduction
                float quality_reduction = pattern_quality * 0.7f;
                float adjusted_penalty = base_penalty * (1.0f - quality_reduction);

                info.confidence_penalty = 1.0f - adjusted_penalty;
                info.confidence_penalty = SDKCompat::clamp(info.confidence_penalty, 0.97f, 1.0f);

                // Apply sample-count-based scaling
                // Example: Perfect pattern with only 10 samples:
                //   Pattern penalty: 0.992x
                //   Sample multiplier: 0.864x
                //   Combined: 0.857x (needs more observation time!)
                info.confidence_penalty *= sample_confidence_multiplier;

                // Enforce absolute minimum confidence after all scaling
                // Even with worst case (poor pattern + few samples), don't go below 60%
                info.confidence_penalty = std::max(info.confidence_penalty, 0.60f);
            }
            else
            {
                // Random jitter = less predictable (use average, so larger uncertainty)
                // variance 0.5 → 0.90x
                // variance 0.8 → 0.84x
                // variance 1.0 → 0.80x (20% max penalty)
                info.confidence_penalty = 1.0f - (variance * 0.2f);
                info.confidence_penalty = SDKCompat::clamp(info.confidence_penalty, 0.80f, 1.0f);

                // Apply sample-count-based scaling (even more important for random jitter!)
                info.confidence_penalty *= sample_confidence_multiplier;

                // Enforce absolute minimum confidence after all scaling
                // Random jitter with few samples = maximum penalty, but still reasonable
                info.confidence_penalty = std::max(info.confidence_penalty, 0.50f);
            }
        }
        else
        {
            // Not juking - use current velocity (normal prediction)
            info.is_juking = false;
            info.predicted_velocity = current_velocity;
            info.confidence_penalty = 1.0f;
        }

        return info;
    }

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
        UntargetabilityInfo untargetability;
        ReviveInfo revive;
        PolymorphInfo polymorph;
        KnockbackInfo knockback;
        GroundedInfo grounded;
        JukeInfo juke;          // Erratic zigzag movement (dodging behavior)
        std::vector<WindwallInfo> windwalls;
        std::vector<TerrainInfo> terrains;
        bool is_slowed;
        bool is_csing;          // Target is likely last-hitting minion (attention divided)
        bool has_shield;
        bool is_clone;
        bool blocked_by_windwall;
        bool blocked_by_terrain;

        // Confidence and priority adjustments
        float confidence_multiplier;
        float priority_multiplier;

        EdgeCaseAnalysis() : is_slowed(false), is_csing(false), has_shield(false),
            is_clone(false), blocked_by_windwall(false), blocked_by_terrain(false),
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
        analysis.untargetability = detect_untargetability(target);
        analysis.revive = detect_revive(target);
        analysis.polymorph = detect_polymorph(target);
        analysis.knockback = detect_knockback(target);
        analysis.grounded = detect_grounded(target);
        analysis.juke = detect_juke(target);
        analysis.windwalls = detect_windwalls();
        analysis.terrains = detect_terrain();
        analysis.is_slowed = is_slowed(target);
        analysis.is_csing = is_likely_csing(target);
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

            // Check terrain blocking (Azir R, Taliyah R)
            for (const auto& terrain : analysis.terrains)
            {
                if (terrain.blocks_projectiles)
                {
                    // Simple check: if terrain is between source and target
                    math::vector3 to_target = target->get_position() - source->get_position();
                    float dist_to_target = to_target.magnitude();
                    if (dist_to_target > 1.0f)
                    {
                        math::vector3 dir = to_target / dist_to_target;
                        math::vector3 to_terrain = terrain.position - source->get_position();
                        float proj = to_terrain.dot(dir);

                        // If terrain is between us and target
                        if (proj > 0.f && proj < dist_to_target)
                        {
                            math::vector3 closest = source->get_position() + dir * proj;
                            float perp_dist = (terrain.position - closest).magnitude();

                            // If projectile path passes through terrain
                            if (perp_dist < terrain.radius)
                            {
                                analysis.blocked_by_terrain = true;
                                break;
                            }
                        }
                    }
                }
            }
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

        // NOTE: Slows do NOT need confidence multiplier
        // Slow effect is FULLY captured by reduced move_speed in TTE calculation
        // 25% slow → 34% more dodge time → 57% less reaction window (natural geometric scaling)

        if (analysis.polymorph.is_polymorphed)
            analysis.confidence_multiplier *= 1.25f;  // Limited movement control

        if (analysis.knockback.is_knocked_back)
            analysis.confidence_multiplier *= 0.6f;  // Unpredictable displacement

        // GROUNDED: MASSIVE confidence boost!
        // Target CANNOT use Flash, dashes, or blinks - only walking escape
        if (analysis.grounded.is_grounded)
            analysis.confidence_multiplier *= 1.5f;  // 50% boost - they're trapped!

        if (analysis.dash.is_dashing)
            analysis.confidence_multiplier *= analysis.dash.confidence_multiplier;

        // Spell shields: Don't penalize confidence - we want to break shields
        // Instead, reduce priority so we prefer unshielded targets when multiple options
        if (analysis.has_shield)
            analysis.priority_multiplier *= 0.7f;  // Prefer unshielded targets

        if (analysis.blocked_by_windwall)
            analysis.confidence_multiplier *= 0.2f;  // Will be blocked by windwall

        if (analysis.blocked_by_terrain)
            analysis.confidence_multiplier *= 0.1f;  // Will be blocked by terrain (Azir R, Taliyah R)

        // Terrain confidence boost (target trapped/restricted movement)
        float terrain_boost = get_terrain_confidence_boost(target, analysis.terrains);
        analysis.confidence_multiplier *= terrain_boost;

        if (analysis.is_clone)
            analysis.priority_multiplier *= 0.1f;  // Don't target clones

        // CRITICAL: Untargetable targets cannot be hit by ANY spell
        // Severely penalize to prevent wasted casts
        if (analysis.untargetability.is_untargetable)
        {
            analysis.confidence_multiplier *= 0.05f;  // Almost zero confidence
            analysis.priority_multiplier *= 0.01f;    // Extremely low priority
        }

        // Revive passives: Lower priority (they'll come back, prefer other targets)
        // But don't block entirely - sometimes want to force them to use GA/passive
        if (analysis.revive.has_revive)
        {
            analysis.priority_multiplier *= 0.6f;  // 40% lower priority
        }

        return analysis;
    }

} // namespace EdgeCases