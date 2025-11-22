#pragma once

#include "sdk.hpp"
#include <pred_sdk.hpp>
#include <dmg_sdk.hpp>

enum spell_farm_flag: uint8_t
{
    none = 0,
    outside_of_attack_range = 1,
    lasthit = 2,
    when_attack_not_ready = 4,
};

class orb_sdk
{
public:
    enum event_type: uint8_t
    {
        before_attack = 0,
        before_move,
    };

    enum orb_forced_mode: uint32_t
    {
        forced_mode_none = 0,
        forced_mode_combo = 1 << 0,
        forced_mode_harass = 1 << 1,
        forced_mode_clear = 1 << 2,
        forced_mode_lasthit = 1 << 3,
        forced_mode_flee = 1 << 4,
        forced_mode_freeze = 1 << 5,
        forced_mode_fast_clear = 1 << 6,
    };

    class event_data
    {
    public:
        game_object* target{};
    };

    virtual bool combo() = 0;
    virtual bool harass() = 0;
    virtual bool clear() = 0;
    virtual bool lasthit() = 0;
    virtual bool flee() = 0;
    virtual bool freeze() = 0;
    virtual bool fast_clear() = 0;
    virtual bool can_spell( game_object* target, const float time = 0.f ) = 0;
    virtual bool is_attack_ready_in( const float time ) = 0;
    virtual bool would_cancel_attack() = 0;
    virtual float get_attack_cast_end_time() = 0;
    virtual game_object* get_target_in_attack_range() = 0;
    virtual bool is_in_auto_attack_range( game_object* source, game_object* target, float offset = 0.f ) = 0;
    virtual pred_sdk::pred_data spell_farm( pred_sdk::spell_data spell_data, const int aoe_hits_needed, const spell_farm_flag flags = spell_farm_flag::none, const dmg_sdk::damage_type damage_type = dmg_sdk::damage_type::physical ) = 0;
    virtual bool is_spell_farm_enabled() = 0;
    virtual bool attack( game_object* target ) = 0;
    virtual void register_callback( event_type ev, void* fn ) = 0;
    virtual void unregister_callback( event_type ev, void* fn ) = 0;
    virtual bool is_in_auto_attack_range( game_object* source, const math::vector3& source_position, game_object* target, float offset = 0.f ) = 0;
    virtual bool is_spell_farm_lasthit_enabled() = 0;
    virtual game_object* get_current_target() = 0;
    virtual void set_forced_mode( orb_forced_mode modes ) = 0;
    virtual float get_next_attack_time() = 0;
};

namespace sdk
{
    inline orb_sdk* orbwalker = nullptr;
}

namespace sdk_init
{
    inline bool orbwalker()
    {
        sdk::orbwalker = reinterpret_cast< orb_sdk* >( g_sdk->get_orb_sdk() );

        return sdk::orbwalker != nullptr;
    }
}