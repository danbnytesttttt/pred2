#pragma once

#include "sdk.hpp"

class infotab_sdk
{
public:
    struct text_entry
    {
        std::string text{};
        uint32_t color = 0xFFFFFFFF;
    };

    virtual uint32_t add_text( const infotab_sdk::text_entry& title, const std::function< infotab_sdk::text_entry() >& fn ) = 0;
    virtual void remove_text( uint32_t id ) = 0;
    virtual uint32_t add_hotkey_text( std::string* hotkey, const infotab_sdk::text_entry& title, const std::function< infotab_sdk::text_entry() >& fn ) = 0;
};

namespace sdk
{
    inline infotab_sdk* infotab = nullptr;
}

namespace sdk_init
{
    inline bool infotab()
    {
        if( sdk::infotab )
            return true;

        const std::string module_name = "VEN.Infotab";
        if( !g_sdk->add_dependency( "Core/" + module_name ) )
            return false;

        sdk::infotab = reinterpret_cast< infotab_sdk* >( g_sdk->get_custom_sdk( module_name ) );

        return true;
    }
}