#pragma once

#include "sdk.hpp"

class hp_sdk
{
public:
    virtual float get_predicted_health( game_object* target, const float time ) = 0;
    virtual float get_predicted_health_if( game_object* target, const float time, const std::function< bool( game_object* ) >& fn ) = 0;
    virtual game_object* get_minion_focus( game_object* source ) = 0;
};

namespace sdk
{
    inline hp_sdk* health_prediction = nullptr;
}

namespace sdk_init
{
    inline bool health_prediction()
    {
        if( sdk::health_prediction )
            return true;

        const std::string module_name = "VEN.HealthPrediction";
        if( !g_sdk->add_dependency( "Core/" + module_name ) )
            return false;

        sdk::health_prediction = reinterpret_cast<hp_sdk*>(g_sdk->get_custom_sdk( module_name ));

        return true;
    }
}