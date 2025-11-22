#pragma once

#include "sdk.hpp"

class shaders_sdk_vision_manager
{
public:
    virtual void draw( color clr ) = 0;
    virtual uint64_t add( const math::vector3& pos, float radius ) = 0;
    virtual void remove( uint64_t id ) = 0;
};

class shaders_sdk
{
public:
    enum world_circle_style : int
    {
        normal = 0,
        rgb,
    };

    virtual void draw_shadow_circle( const math::vector2& pos, float radius, color clr = {}, float blur = 0.11f ) = 0;
    virtual void draw_world_circle( const math::vector3& pos, float radius, float thickness, color clr, shaders_sdk::world_circle_style style = normal ) = 0;
    virtual void draw_world_circle_filled( const math::vector3& pos, float radius, color clr ) = 0;
    virtual void draw_world_segment( const math::vector3& start, const math::vector3& end, float thickness, color clr, uint8_t rounding_style ) = 0;
    virtual void draw_percentage_circle( const math::vector2& pos, float radius, float thickness, float percent, color clr ) = 0;
    virtual std::shared_ptr<shaders_sdk_vision_manager> get_vision_manager( int team_id ) = 0;
    virtual void draw_vision_texture( color clr, void* texture ) = 0;
    virtual void draw_world_circle_segment( const math::vector3& pos, float radius, float thickness, float start, float end, color clr, shaders_sdk::world_circle_style style = normal ) = 0;
    virtual void draw_world_circle_segment( const math::vector3& pos, float radius, float thickness, const math::vector3& pos2, float radius2, color clr, shaders_sdk::world_circle_style style = normal ) = 0;
    virtual void draw_filled_world_segment( const math::vector3& start, const math::vector3& end, float thickness, color clr, uint8_t rounding_style, float center_alpha ) = 0;
};

namespace sdk
{
    inline shaders_sdk* shaders = nullptr;
}

namespace sdk_init
{
    inline bool shaders()
    {
        if( sdk::shaders )
            return true;

        const std::string module_name = "VEN.Shaders";
        if( !g_sdk->add_dependency( "Core/" + module_name ) )
            return false;

        sdk::shaders = reinterpret_cast< shaders_sdk* >(g_sdk->get_custom_sdk( module_name ));

        return true;
    }
}