#pragma once

#include "sdk.hpp"
#include <color.hpp>

class notification_sdk
{
public:
    virtual void add( const std::string& title, const std::string& content, const color& clr = 0xfff0e6d2 ) = 0;
    virtual void add_image( const std::string& title, const std::string& content, void* image, const color& clr = 0xfff0e6d2 ) = 0;
};

namespace sdk
{
    inline notification_sdk* notification = nullptr;
}

namespace sdk_init
{
    inline bool notification()
    {
        if( sdk::notification )
            return true;

        const std::string module_name = "VEN.Notification";
        if( !g_sdk->add_dependency( "Core/" + module_name ) )
            return false;

        sdk::notification = reinterpret_cast< notification_sdk* >(g_sdk->get_custom_sdk( module_name ));

        return true;
    }
}