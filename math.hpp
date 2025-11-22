#pragma once

#include <cmath>
#include <corecrt_math_defines.h>

inline bool cmpf( float x, float y, float epsilon = 0.01f )
{
    if ( fabs( x - y ) < epsilon )
        return true; //they are same

    return false; //they are not same
}

inline float deg2rad( float degrees )
{
    return degrees * ( static_cast< float >( M_PI ) / 180.f );
}

inline float rad2deg( float radians )
{
    return radians * ( 180.f / static_cast< float >( M_PI ) );
}

namespace math
{
    class vector2
    {
    public:
        float x;
        float y;

        vector2()
        {
            this->x = 0.f;
            this->y = 0.f;
        }

        template< typename X, typename Y >
        vector2( X x, Y y )
        {
            this->x = static_cast< float >( x );
            this->y = static_cast< float >( y );
        }

        bool operator==( const vector2& vec ) const
        {
            return cmpf( this->x, vec.x ) && cmpf( this->y, vec.y );
        }

        bool operator!=( const vector2& vec ) const
        {
            return !( cmpf( this->x, vec.x ) && cmpf( this->y, vec.y ) );
        }

        bool is_on_screen( const int width, const int height )
        {
            const int half_w = static_cast< int >( width * 0.5f );
            const int half_h = static_cast< int >( height * 0.5f );

            return this->x >= -half_w && this->x <= width + half_w && this->y >= -half_h && this->y <= height + half_h;
        }

        float angle_between( math::vector2 const& v2 ) const
        {
            if( *this == v2 )
                return 0.f;

            if( ( v2.x == 0.f && v2.y == 0.f ) || ( x == 0.f && y == 0.f ) )
                return std::numeric_limits<float>::quiet_NaN();

            float dot_product = x * v2.x + y * v2.y;
            float magnitude1 = std::sqrt( x * x + y * y );
            float magnitude2 = std::sqrt( v2.x * v2.x + v2.y * v2.y );
            float angle_in_radians = std::acos( dot_product / ( magnitude1 * magnitude2 ) );
            return angle_in_radians * 180.0f / 3.14159265358979323846f;
        }
    };

    class vector3
    {
    public:
        float x;
        float y;
        float z;

        vector3()
        {
            this->x = this->y = this->z = 0.f;
        }

        template< typename X, typename Y, typename Z >
        vector3( X x, Y y, Z z )
        {
            this->x = static_cast< float >( x );
            this->y = static_cast< float >( y );
            this->z = static_cast< float >( z );
        }

        vector3 operator-( const vector3& vec ) const
        {
            return vector3{ this->x - vec.x, this->y - vec.y, this->z - vec.z };
        }

        vector3 operator+( const vector3& vec ) const
        {
            return vector3{ this->x + vec.x, this->y + vec.y, this->z + vec.z };
        }

        vector3 operator+( const float& flt )  const
        {
            return vector3{ this->x + flt, this->y + flt, this->z + flt };
        }

        double operator*( const vector3& vec ) const
        {
            return this->dot( vec );
        }

        vector3 operator^( const vector3& vec ) const
        {
            return this->cross( vec );
        }

        vector3 operator*( float flt ) const
        {
            return vector3{ this->x * flt, this->y * flt, this->z * flt };
        }

        vector3 operator/( float flt ) const
        {
            if( flt == 0.f )
                return *this;

            if ( cmpf( this->y, 0 ) )
                return vector3{ this->x / flt, this->y, this->z / flt };

            return vector3{ this->x / flt, this->y / flt, this->z / flt };
        }

        bool operator==( const vector3& vec ) const
        {
            return cmpf( this->x, vec.x ) && cmpf( this->y, vec.y ) && cmpf( this->z, vec.z );
        }

        bool operator!=( const vector3& vec ) const
        {
            return !( cmpf( this->x, vec.x, 0.1f ) && cmpf( this->y, vec.y, 0.1f ) && cmpf( this->z, vec.z, 0.1f ) );
        }

        float dot( const vector3& vec ) const
        {
            float ret = 0;

            ret += this->x * vec.x;
            ret += this->y * vec.y;
            ret += this->z * vec.z;

            return ret;
        }

        vector3 cross( const vector3& vec ) const
        {
            return vector3(
                           this->z * vec.y - this->y * vec.z,
                           this->y * vec.x - this->x * vec.y,
                           this->x * vec.z - this->z * vec.x );
        }


        float sqr_magnitude() const
        {
            return ( this->x * this->x ) + ( this->y * this->y ) + ( this->z * this->z );
        }

        float magnitude() const
        {
            return std::sqrtf( this->sqr_magnitude() );
        }

        vector3 normalized() const
        {
            auto copy = *this;
            copy.normalize();
            return copy;
        }

        void normalize()
        {
            float m = this->magnitude();
            if( m == 0.f )
                return;

            this->x /= m;
            this->z /= m;
        }

        float distance( const vector3& vec ) const
        {
            float x = this->x - vec.x;
            float z = this->z - vec.z;

            return sqrtf( x * x + z * z );
        }

        vector3 extended( vector3 target, float dist, bool allow_over_extension = false ) const
        {
            vector3 d = target - *this;
            float m = d.magnitude();
            if ( cmpf( m, 0.f ) || ( !allow_over_extension && ( m < dist ) ) )
                return target;

            return *this + ( ( d * dist ) / m );
        }

        vector3 extended_direction( vector3 d, float dist ) const
        {
            vector3 ret { };

            ret.x = this->x + ( dist * d.x );
            ret.y = this->y;
            ret.z = this->z + ( dist * d.z );
            return ret;
        }

        float polar() const
        {
            if ( cmpf( this->x, 0 ) )
            {
                if ( this->z > 0.f || this->y > 0.f )
                    return 90.f;
                else if ( this->z < 0.f || this->y < 0.f )
                    return 270.f;
                else
                    return 0.f;
            }
            else
            {
                float theta = rad2deg( std::atanf( this->z / this->x ) );
                if ( this->x < 0.f )
                    theta += 180.f;
                if ( theta < 0.f )
                    theta += 360.f;

                return theta;
            }
        }

        float angle_between( const vector3& v1, const vector3& v2 ) const
        {
            vector3 negated_v0 = vector3( -this->x, -this->y, -this->z );
            vector3 p1 = negated_v0 + v1;
            vector3 p2 = negated_v0 + v2;
            float theta = p1.polar() - p2.polar();
            if ( theta < 0.f )
                theta += 360.f;

            if ( theta > 180.f )
                theta = 360.f - theta;

            return theta;
        }

        vector3 between( vector3 vec ) const
        {
            return ( ( vec - *this ) / 2 + *this );
        }

        vector3 rotate( vector3 vec, float theta ) const
        {
            const float cos_theta = cosf( theta );
            const float sin_theta = sinf( theta );

            /*vector3 self = *this;
            vector3 rotated = ( self * cos_theta ) + ( vec.cross( self ) * sin_theta ) + ( vec * vec.dot( self ) ) * ( 1.f - cos_theta );*/

            auto rotated_x = ( ( vec.x - this->x ) * cos_theta ) - ( ( vec.z - this->z ) * sin_theta ) + this->x;
            auto rotated_y = this->z + ( ( this->z - vec.z ) * cos_theta ) + ( ( vec.x - this->x ) * sin_theta );

            return vector3( rotated_x, this->y, rotated_y );
        }

        vector3 rotate( float angle ) const
        {
            auto ang = deg2rad( angle );

            auto c = cosf( ang );
            auto s = sinf( ang );

            auto rotated_x = ( this->x * c - this->z * s );
            auto rotated_y = ( this->z * c + this->x * s );

            return { rotated_x , this->y , rotated_y };
        }

        vector3 perpendicular_left() const
        {
            return vector3( -this->z, this->y, this->x );
        }

        vector3 perpendicular_right() const
        {
            return vector3( this->z, this->y, -this->x );
        }

        vector3 to_direction() const
        {
            auto self = *this;

            const auto magnitude = self.magnitude();
            if( magnitude == 0.f )
                return self;

            float length_inverse = 1.f / magnitude;

            return vector3(
                           this->x * length_inverse,
                           0.f,
                           this->z * length_inverse );
        }

        vector2 to_2d() const
        {
            return { this->x, this->z };
        }

        std::string to_string() const
        {
            return std::string( "math::vector3{ " ) + std::to_string( this->x ) + "f, " + std::to_string( this->y ) + "f, " + std::to_string( this->z ) + "f }";
        }
    };

    struct vector4
    {
        float x;
        float y;
        float z;
        float w;
    };

    class rect
    {
    public:
        float top;
        float left;
        float bottom;
        float right;

        rect operator-( const rect& r )
        {
            return rect( this->top - r.top, this->left - r.left, this->bottom - r.bottom, this->right - r.right );
        }

        rect operator+( const rect& r )
        {
            return rect( this->top + r.top, this->left + r.left, this->bottom + r.bottom, this->right + r.right );
        }

        rect operator+( const vector2& v )
        {
            return rect( this->top, this->left, this->bottom + v.y, this->right + v.x );
        }

        rect operator+=( const rect& r )
        {
            return rect( this->top + r.top, this->left + r.left, this->bottom + r.bottom, this->right + r.right );
        }

        rect operator*( const int& i )
        {
            return rect( this->top * i, this->left * i, this->bottom * i, this->right * i );
        }

        rect operator*=( const int& i )
        {
            return rect( this->top * i, this->left * i, this->bottom * i, this->right * i );
        }

        vector2 get_center()
        {
            return vector2( ( this->right - this->left ) * 0.5f, ( this->bottom - this->top ) * 0.5f );
        }

        rect top_left()
        {
            return rect( this->top, this->left, this->top, this->left );
        }

        rect top_right()
        {
            return rect( this->top, this->right, this->top, this->right );
        }

        rect bottom_left()
        {
            return rect( this->bottom, this->left, this->bottom, this->left );
        }
    };
}
