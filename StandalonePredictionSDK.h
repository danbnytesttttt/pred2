#pragma once

/**
 * Standalone SDK Compatibility Layer
 *
 * Provides compatibility wrappers for SDK method name differences.
 * Include this BEFORE HybridPrediction.h
 *
 * This layer maps the prediction system's expected methods to your SDK's actual methods.
 */

#include "sdk.hpp"
#include <cmath>
#include <cstring>

 // Extend math::vector3 with missing methods
namespace math
{
    // Add .length() method as alias for .magnitude()
    inline float length(const vector3& v)
    {
        return v.magnitude();
    }

    // Add .is_zero() method
    inline bool is_zero(const vector3& v)
    {
        constexpr float EPSILON = 1e-6f;
        return (std::abs(v.x) < EPSILON && std::abs(v.y) < EPSILON && std::abs(v.z) < EPSILON);
    }
}

// Extend game_object with missing methods via free functions
// These can be called as if they were member functions when included

inline float get_health(game_object* obj)
{
    return obj ? obj->get_hp() : 0.f;
}

inline float get_max_health(game_object* obj)
{
    return obj ? obj->get_max_hp() : 1.f;
}

// CC Detection - SDK uses has_buff_of_type() instead of individual methods
inline bool is_stunned(game_object* obj)
{
    return obj ? obj->has_buff_of_type(buff_type::stun) : false;
}

inline bool is_rooted(game_object* obj)
{
    return obj ? obj->has_buff_of_type(buff_type::snare) : false;
}

inline bool is_charmed(game_object* obj)
{
    return obj ? obj->has_buff_of_type(buff_type::charm) : false;
}

inline bool is_feared(game_object* obj)
{
    return obj ? obj->has_buff_of_type(buff_type::fear) : false;
}

inline bool is_taunted(game_object* obj)
{
    return obj ? obj->has_buff_of_type(buff_type::taunt) : false;
}

inline bool is_suppressed(game_object* obj)
{
    return obj ? obj->has_buff_of_type(buff_type::suppression) : false;
}

inline bool is_knocked_up(game_object* obj)
{
    return obj ? obj->has_buff_of_type(buff_type::knockup) : false;
}

// Animation state detection - checks if ACTUALLY locked (in windup), not just animating
// Human reaction buffer: Players don't move instantly when animation ends ("mental follow-through")
// 25ms = ~1 server tick, covers input latency + mental processing without being sloppy
constexpr float HUMAN_REACTION_BUFFER = 0.025f;

// Helper: Get correct windup time (Scales with Attack Speed for AAs)
// CRITICAL FIX: Auto-attack windup scales with AS - a Level 18 ADC with 2.0 AS
// has ~3x faster windup than the base spell data suggests
inline float get_current_windup(game_object* obj, spell_cast* cast)
{
    if (!obj || !cast) return 0.f;

    if (cast->is_basic_attack())
    {
        // Use get_attack_cast_delay() which accounts for Attack Speed
        // The raw spell data only holds the BASE windup (e.g. 0.25s), ignoring AS items
        return obj->get_attack_cast_delay();
    }

    // For spells, use the static data delay
    return cast->get_cast_delay();
}

// Get exact time remaining in animation lock
// Returns 0 if not locked, otherwise the time until they can move again
// Uses HUMAN_REACTION_BUFFER (0.025s) as default for average players
inline float get_remaining_lock_time(game_object* obj)
{
    if (!obj) return 0.f;
    auto active_cast = obj->get_active_spell_cast();
    if (!active_cast) return 0.f;
    auto spell_cast = active_cast->get_spell_cast();
    if (!spell_cast) return 0.f;

    if (!g_sdk || !g_sdk->clock_facade) return 0.f;
    float current_time = g_sdk->clock_facade->get_game_time();
    float cast_start = active_cast->get_cast_start_time();
    float windup = get_current_windup(obj, spell_cast);

    float end_time = cast_start + windup + HUMAN_REACTION_BUFFER;
    return std::max(0.f, end_time - current_time);
}

// ADAPTIVE VERSION: Uses measured reaction buffer for this specific player
// Pass the result of tracker.get_adaptive_reaction_buffer() for personalized prediction
// Scripters: ~0.005-0.015s (near-instant cancel), Lazy: ~0.05-0.1s (let backswing play)
inline float get_remaining_lock_time_adaptive(game_object* obj, float adaptive_reaction_buffer)
{
    if (!obj) return 0.f;
    auto active_cast = obj->get_active_spell_cast();
    if (!active_cast) return 0.f;
    auto spell_cast = active_cast->get_spell_cast();
    if (!spell_cast) return 0.f;

    if (!g_sdk || !g_sdk->clock_facade) return 0.f;
    float current_time = g_sdk->clock_facade->get_game_time();
    float cast_start = active_cast->get_cast_start_time();
    float windup = get_current_windup(obj, spell_cast);

    // Use adaptive buffer instead of fixed constant
    float end_time = cast_start + windup + adaptive_reaction_buffer;
    return std::max(0.f, end_time - current_time);
}

inline bool is_auto_attacking(game_object* obj)
{
    if (!obj) return false;
    auto active_cast = obj->get_active_spell_cast();
    if (!active_cast) return false;
    auto spell_cast = active_cast->get_spell_cast();
    if (!spell_cast || !spell_cast->is_basic_attack()) return false;

    // Only locked during windup (before projectile fires)
    // After windup, champion can animation cancel and move
    if (!g_sdk || !g_sdk->clock_facade) return false;
    float current_time = g_sdk->clock_facade->get_game_time();
    float cast_start = active_cast->get_cast_start_time();

    // CRITICAL FIX: Use dynamic windup that accounts for Attack Speed
    float windup = get_current_windup(obj, spell_cast);

    // Still in windup + reaction buffer = effectively locked
    return (current_time - cast_start) < (windup + HUMAN_REACTION_BUFFER);
}

inline bool is_casting_spell(game_object* obj)
{
    if (!obj) return false;
    auto active_cast = obj->get_active_spell_cast();
    if (!active_cast) return false;
    auto spell_cast = active_cast->get_spell_cast();
    if (!spell_cast || spell_cast->is_basic_attack()) return false;

    // Only locked during cast delay (before spell releases)
    if (!g_sdk || !g_sdk->clock_facade) return false;
    float current_time = g_sdk->clock_facade->get_game_time();
    float cast_start = active_cast->get_cast_start_time();
    float cast_delay = spell_cast->get_cast_delay();

    // Still in cast delay = actually locked
    // Some spells have 0 cast delay (instant) - not locked
    if (cast_delay < 0.01f) return false;

    // Include reaction buffer for mental follow-through
    return (current_time - cast_start) < (cast_delay + HUMAN_REACTION_BUFFER);
}

inline bool is_channeling(game_object* obj)
{
    if (!obj) return false;
    auto active_cast = obj->get_active_spell_cast();
    if (!active_cast) return false;

    // Check if channeling end time is valid (> 0 means channeling)
    float channel_end = active_cast->get_cast_channeling_end_time();
    return channel_end > 0.f;
}

/**
 * MOBILE CHANNEL FIX: Detect if a channel is stationary or mobile
 *
 * Mobile Channels (allow movement): Lucian R, Varus Q, Xerath Q, Pyke Q, Vladimir E
 * Stationary Channels (lock movement): Malzahar R, Recall, Katarina R, MF R
 *
 * Heuristic: If target is channeling BUT is_moving() returns true AND
 * they have meaningful velocity, it's a mobile channel.
 * Stationary channels clamp velocity to exactly 0.
 */
inline bool is_stationary_channel(game_object* obj)
{
    if (!obj) return false;

    // If not channeling at all, return false
    if (!is_channeling(obj)) return false;

    // HEURISTIC: If they are channeling BUT moving, it's a mobile channel
    // Mobile channels (Lucian R, Varus Q charging) allow movement during channel
    // Stationary channels (Malz R, Recall, MF R) lock the champion in place
    if (obj->is_moving())
    {
        // Double-check with path - if they have valid waypoints, they're actually moving
        auto path = obj->get_path();
        if (path.size() > 1)
        {
            return false;  // Mobile channel - not stationary
        }
    }

    // Not moving during channel = stationary channel
    return true;
}

inline bool is_recalling(game_object* obj)
{
    if (!obj) return false;

    // Check for channeling (recall is a channel)
    auto active_cast = obj->get_active_spell_cast();
    if (!active_cast) return false;

    float channel_end = active_cast->get_cast_channeling_end_time();
    if (channel_end <= 0.f) return false;  // Not channeling

    // Check spell name
    auto spell_cast = active_cast->get_spell_cast();
    if (!spell_cast) return false;

    auto spell_data = spell_cast->get_spell_data();
    if (!spell_data) return false;

    auto static_data = spell_data->get_static_data();
    if (!static_data) return false;

    const char* name = static_data->get_name();
    if (name && (std::strstr(name, "Recall") || std::strstr(name, "recall")))
        return true;

    return false;
}

inline float get_base_move_speed(game_object* obj)
{
    // SDK may not have base move speed - return current move speed as fallback
    return obj ? obj->get_move_speed() : 0.f;
}