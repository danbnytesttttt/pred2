#pragma once

#include "sdk.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <profileapi.h>
#include <string>

#define BENCHMARK_ADD( name ) static benchmark_data* benchmark{}; if( !benchmark ) benchmark = sdk::benchmark->add( name );

class benchmark_data
{
public:
    benchmark_data( const std::string& str )
    {
        this->name = str;
        QueryPerformanceFrequency( &this->frequency );
    }

    void start()
    {
        QueryPerformanceCounter( &this->start_time );
    }

    void stop()
    {
        QueryPerformanceCounter( &this->end_time );

        long long diff = this->end_time.QuadPart - this->start_time.QuadPart;
        this->elapsed_milliseconds[ last_idx % 20 ] = static_cast< double >( diff ) * 1000.0 / static_cast< double >( frequency.QuadPart );
        last_idx++;
    }

    std::string name;
    LARGE_INTEGER start_time{};
    LARGE_INTEGER end_time{};
    LARGE_INTEGER frequency{};
    double elapsed_milliseconds[20]{};
    int last_idx{};
};

class benchmark_sdk
{
public:
    virtual benchmark_data* add( const std::string& str ) = 0;
};

namespace sdk
{
    inline benchmark_sdk* benchmark = nullptr;
}

namespace sdk_init
{
    inline bool benchmark()
    {
        if( sdk::benchmark )
            return true;

        const std::string module_name = "VEN.Benchmark";
        if( !g_sdk->add_dependency( "Core/" + module_name ) )
            return false;

        sdk::benchmark = reinterpret_cast< benchmark_sdk* >( g_sdk->get_custom_sdk( module_name ) );

        return true;
    }
}