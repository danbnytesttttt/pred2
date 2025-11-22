#pragma once

#include "sdk.hpp"

class dmg_sdk
{
public:
    enum class damage_type
    {
        physical = 0,
        magical,
        truedamage,
    };

    virtual float calc_damage( damage_type type, game_object* source, game_object* target, float damage ) = 0;
    virtual float get_spell_damage( game_object* hero, int spell_slot ) = 0;
    virtual float get_spell_damage( game_object* source, game_object* target, int spell_slot ) = 0;
    virtual float get_aa_damage( game_object* source, game_object* target, bool next_attack = false ) = 0;
};

namespace sdk
{
    inline dmg_sdk* damage = nullptr;
}

namespace sdk_init
{
    inline bool damage()
    {
        if( sdk::damage )
            return true;

        const std::string module_name = "VEN.Damage";
        if( !g_sdk->add_dependency( "Core/" + module_name ) )
            return false;

        sdk::damage = reinterpret_cast<dmg_sdk*>(g_sdk->get_custom_sdk( module_name ));

        return true;
    }
}