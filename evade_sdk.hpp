#pragma once

#include "sdk.hpp"

class evade_sdk
{
public:
    enum event_type: uint8_t
    {
        before_move = 0,
    };

    virtual bool is_evading() = 0;
    virtual bool can_spell( int spell_slot, float cast_time ) = 0;
    virtual bool can_dash( const math::vector3& pos, float dash_speed, float cast_time = 0.f ) = 0;
    virtual bool is_position_safe( const math::vector3& pos ) = 0;
    virtual bool is_dangerous_spell( sm_sdk::spell* spell ) = 0;
    virtual bool is_player_inside_dangerous_spell() = 0;
    virtual float get_spell_intersection_time( const math::vector3& start_pos, const math::vector3& end_pos, float speed, sm_sdk::spell* spell ) = 0;
    virtual void register_callback( evade_sdk::event_type ev, void* fn ) = 0;
    virtual void unregister_callback( evade_sdk::event_type ev, void* fn ) = 0;
    virtual math::vector3 get_dodge_position() = 0;
};

namespace sdk
{
    inline evade_sdk* evade = nullptr;
}

namespace sdk_init
{
    inline bool evade()
    {
        sdk::evade = reinterpret_cast< evade_sdk* >( g_sdk->get_evade_sdk() );

        return sdk::evade != nullptr;
    }
}