#pragma once

#include "sdk.hpp"

class ts_sdk
{
public:
    virtual game_object* get_hero_target( std::function< bool( game_object* ) > fn = {} ) = 0;
    virtual game_object* get_ally_hero_target( std::function< bool( game_object* ) > fn = {} ) = 0;
    virtual game_object* get_monster_target( std::function< bool( game_object* ) > fn = {} ) = 0;
    virtual game_object* get_forced_target() = 0;
    virtual const std::span< game_object* >& get_sorted_heroes() = 0;
    virtual const std::span< game_object* >& get_sorted_monsters() = 0;
    virtual int get_manual_target_mode() = 0;
};

namespace sdk
{
    inline ts_sdk* target_selector = nullptr;
}

namespace sdk_init
{
    inline bool target_selector()
    {
        if( sdk::target_selector )
            return true;

        const std::string module_name = "VEN.TargetSelector";
        if( !g_sdk->add_dependency( "Core/" + module_name ) )
            return false;

        sdk::target_selector = reinterpret_cast<ts_sdk*>(g_sdk->get_custom_sdk( module_name ));

        return true;
    }
}