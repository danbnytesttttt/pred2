#pragma once

class color
{
public:
    uint8_t a;
    uint8_t r;
    uint8_t g;
    uint8_t b;

    color()
    {
        this->b = 0;
        this->g = 0;
        this->r = 0;
        this->a = 0;
    }

    color( const uint32_t argb )
    {
        this->b = ( argb >> 0 );
        this->g = ( argb >> 8 );
        this->r = ( argb >> 16 );
        this->a = ( argb >> 24 );
    }

    void set_rgb( uint8_t r, uint8_t g, uint8_t b )
    {
        this->b = b;
        this->g = g;
        this->r = r;
    }

    operator uint32_t()
    {
        return static_cast<uint32_t>( a << 24 | r << 16 | g << 8 | b << 0 );
    }

    uint32_t to_argb() const
    {
        return ( this->b << 0 ) | ( this->g << 8 ) | ( this->r << 16 ) | ( this->a << 24 );
    }
};