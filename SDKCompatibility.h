#pragma once

#include "sdk.hpp"
#include <algorithm>

/**
 * SDK COMPATIBILITY LAYER
 *
 * Provides compatibility wrappers for SDK API differences.
 * Maps non-existent convenience methods to actual SDK calls.
 */

namespace SDKCompat
{
    // =========================================================================
    // GAME OBJECT HELPERS
    // =========================================================================

    /**
     * Get champion name (maps to get_char_name)
     */
    inline std::string get_champion_name(game_object* obj)
    {
        return obj ? obj->get_char_name() : "";
    }

    /**
     * Get team (maps to get_team_id)
     */
    inline int get_team(game_object* obj)
    {
        return obj ? obj->get_team_id() : 0;
    }

    /**
     * Get health percent (calculated from hp/max_hp)
     */
    inline float get_health_percent(game_object* obj)
    {
        if (!obj) return 0.f;
        float max_hp = obj->get_max_hp();
        if (max_hp <= 0.f) return 0.f;
        return obj->get_hp() / max_hp;
    }

    /**
     * Check if stunned (uses has_buff_of_type)
     */
    inline bool is_stunned(game_object* obj)
    {
        return obj ? obj->has_buff_of_type(buff_type::stun) : false;
    }

    /**
     * Check if rooted (uses has_buff_of_type)
     */
    inline bool is_rooted(game_object* obj)
    {
        return obj ? obj->has_buff_of_type(buff_type::snare) : false;
    }

    /**
     * Check if charmed (uses has_buff_of_type)
     */
    inline bool is_charmed(game_object* obj)
    {
        return obj ? obj->has_buff_of_type(buff_type::charm) : false;
    }

    /**
     * Check if feared (uses has_buff_of_type)
     */
    inline bool is_feared(game_object* obj)
    {
        return obj ? obj->has_buff_of_type(buff_type::fear) : false;
    }

    /**
     * Check if taunted (uses has_buff_of_type)
     */
    inline bool is_taunted(game_object* obj)
    {
        return obj ? obj->has_buff_of_type(buff_type::taunt) : false;
    }

    /**
     * Check if immobilized (any hard CC)
     */
    inline bool is_immobilized(game_object* obj)
    {
        if (!obj) return false;
        return obj->has_buff_of_type(buff_type::stun) ||
               obj->has_buff_of_type(buff_type::snare) ||
               obj->has_buff_of_type(buff_type::suppression) ||
               obj->has_buff_of_type(buff_type::knockup) ||
               obj->has_buff_of_type(buff_type::knockback);
    }

    /**
     * Check if channeling (casting a channeled spell)
     */
    inline bool is_channeling(game_object* obj)
    {
        if (!obj) return false;

        // Check if actively casting (channeling uses active_spell_cast)
        auto* active_cast = obj->get_active_spell_cast();
        if (active_cast)
            return true;

        return false;
    }

    /**
     * Check if winding up (casting/attacking)
     */
    inline bool is_winding_up(game_object* obj)
    {
        if (!obj) return false;

        // Check if actively casting a spell
        auto* active_cast = obj->get_active_spell_cast();
        if (active_cast)
            return true;

        return false;
    }

    /**
     * Check if in combat
     */
    inline bool is_in_combat(game_object* obj)
    {
        // SDK doesn't have this method
        // Could check for damage taken recently, but that's complex
        return false;  // Conservative default
    }

    /**
     * Check if has specific buff by name
     */
    inline bool has_buff(game_object* obj, const char* buff_name)
    {
        if (!obj || !buff_name) return false;

        auto buffs = obj->get_buffs();
        for (auto* buff : buffs)
        {
            if (buff && buff->is_active())
            {
                std::string name = buff->get_name();
                if (name == buff_name)
                    return true;
            }
        }
        return false;
    }

    // =========================================================================
    // OBJECT MANAGER HELPERS
    // =========================================================================

    /**
     * Get enemy heroes
     */
    inline std::vector<game_object*> get_enemy_heroes(object_manager* mgr, game_object* local_player)
    {
        std::vector<game_object*> result;
        if (!mgr || !local_player) return result;

        int my_team = local_player->get_team_id();
        auto all_heroes = mgr->get_heroes();

        for (auto* hero : all_heroes)
        {
            if (hero && hero->is_valid() && !hero->is_dead() && hero->get_team_id() != my_team)
            {
                result.push_back(hero);
            }
        }

        return result;
    }

    /**
     * Get enemy turrets
     */
    inline std::vector<game_object*> get_enemy_turrets(object_manager* mgr, game_object* local_player)
    {
        std::vector<game_object*> result;
        if (!mgr || !local_player) return result;

        int my_team = local_player->get_team_id();
        auto all_turrets = mgr->get_turrets();

        for (auto* turret : all_turrets)
        {
            if (turret && turret->is_valid() && !turret->is_dead() && turret->get_team_id() != my_team)
            {
                result.push_back(turret);
            }
        }

        return result;
    }

    // =========================================================================
    // CORE SDK HELPERS
    // =========================================================================

    /**
     * Get local player (maps to get_local_player())
     */
    inline game_object* local_player(core_sdk* sdk)
    {
        return sdk ? sdk->object_manager->get_local_player() : nullptr;
    }

    // =========================================================================
    // STD HELPERS (for older C++ standards)
    // =========================================================================

    /**
     * std::clamp for C++14 (C++17 feature backport)
     */
    template<typename T>
    inline const T& clamp(const T& value, const T& low, const T& high)
    {
        return std::max(low, std::min(value, high));
    }

    // =========================================================================
    // CONSTANTS
    // =========================================================================

    constexpr float EPSILON = 0.001f;
    constexpr float AOE_MAX_MOVE_SPEED = 600.f;  // Max enemy move speed for AOE predictions

} // namespace SDKCompat

// =========================================================================
// PREPROCESSOR COMPATIBILITY MACROS
// =========================================================================
// These macros redirect non-existent SDK methods to compatibility wrappers

// NOTE: Only enable if needed to avoid macro pollution
#ifdef USE_SDK_COMPAT_MACROS
    #define get_champion_name() get_char_name()
    #define get_team() get_team_id()
#endif
