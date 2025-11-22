#pragma once

#include "sdk.hpp"

class pred_sdk
{
public:
    enum hitchance: int
    {
        automatic = -1,
        any = 0,
        low = 30,
        medium = 50,
        high = 70,
        very_high = 85,
        guaranteed_hit = 100,
    };

    enum class hit_type: uint8_t
    {
        normal = 0,
        undodgeable,
        cast,
        zhonyas,
        cc,
        cc_hard,
        dash,
    };

    enum class spell_type: uint8_t
    {
        linear = 0,
        targetted,
        circular,
        vector,
    };

    /*
        basic attack range from edge to edge
        targeted skill range from center to center (mainly not always)
        skillshot range from center to edge (mainly not always)
        and the range of self-centered area of effects are from the center of the source (blitz R, diana R..)
    */
    enum class targetting_type: uint8_t
    {
        center = 0,
        center_to_edge, // when using this, target has to be within the maximum range the missile would have had, it will not go over that range, it's just used for inside the range
        edge_to_edge,
    };

    enum class collision_type: uint8_t
    {
        unit = 0,
        hero,
        turret,
        terrain,
        yasuo_wall,
        braum_wall,
    };

    struct collision_data
    {
        game_object* object{};
        math::vector3 collided_position{};
    };

    struct collision_ret
    {
        bool collided{};
        std::vector< collision_data > collided_units{};
    };

    class spell_data
    {
    public:
        spell_type spell_type{};
        targetting_type targetting_type{};
        int expected_hitchance = hitchance::automatic; // the expected hitchance

        game_object* source{}; // source object, if none player will be taken
        math::vector3 source_position{}; // position where the skillshot originates from, if none player position will be taken

        bool bypass_anti_buffering{}; // allows casting spells while other spells are being casted (allow spell buffering)
        int spell_slot = -1; // used to build needed hitchance % if expected_hitchance is automatic
        float range{}; // max range of the spell
        float radius{}; // circle is the same as linear, use * 0.5f of the TOTAL radius (so only distance from player to one side)
        float cast_range{}; // the cast range of the spell for vector types: viktor E, ruble R..
        float delay{}; // cast delay
        float proc_delay{}; // delay until the spell hits, for example syndra Q is static 0.6
        float projectile_speed = FLT_MAX; // projectile speed if any, FLT_MAX if the spell has no projectiles
        float extension_override{}; // if we want to override the prediction extension

        std::vector< collision_type > forbidden_collisions{}; // things we dont want to skillshot to collide with
        std::vector< hit_type > expected_hit_types{}; // if we want special hit types only, fill them in here
        std::function< bool( game_object* ) > additional_target_selection_checks{}; // custom function for target selection checks, example: target with specific buff(s) only
    };

    class pred_data
    {
    public:
        bool is_valid{}; // if this is true, prediction was successful we are assured that we can use all the members
        game_object* target{}; // the target prediction is aiming at
        int hitchance{}; // use expected_hitchance field inside spell_data if you only want to check > hitchance

        math::vector3 predicted_position{}; // predicted position of the target **use cast_position** if you want to cast a spell
        math::vector3 predicted_dodge_position{}; // predicted dodge position (usually on the side of the target)
        math::vector3 cast_position{}; // position to use when casting the spell (second cast if vector type)
        math::vector3 first_cast_position{}; // for vector types: viktor E, ruble R..
        float intersection_time{}; // time until the skillshot will hit the player

        math::vector3 collision_pos{}; // the point where the spell collided
        std::vector< collision_data > collided_units{}; // the forbidden_collisions that the spell collided with

        pred_data() {};
        pred_data( game_object* target )
        {
            this->target = target;
        }
    };

    class utils
    {
    public:
        virtual float get_spell_range( pred_sdk::spell_data& data, game_object* target, game_object* source ) = 0;
        virtual bool is_in_range( pred_sdk::spell_data& data, math::vector3 cast_position, game_object* target ) = 0;
        virtual float get_spell_hit_time( pred_sdk::spell_data& data, math::vector3 pos, game_object* target = nullptr ) = 0;
        virtual float get_spell_escape_time( pred_sdk::spell_data& data, game_object* target ) = 0;
    };

    virtual pred_sdk::utils* util() = 0;
    virtual pred_sdk::pred_data targetted( pred_sdk::spell_data spell_data ) = 0;
    virtual pred_sdk::pred_data predict( pred_sdk::spell_data spell_data ) = 0;
    virtual pred_sdk::pred_data predict( game_object* obj, pred_sdk::spell_data spell_data ) = 0;
    virtual math::vector3 predict_on_path( game_object* obj, float time, bool use_server_pos = false ) = 0;
    virtual collision_ret collides( const math::vector3& end_point, pred_sdk::spell_data spell_data, const game_object* target ) = 0;
};

namespace sdk
{
    inline pred_sdk* prediction = nullptr;
}

namespace sdk_init
{
    inline bool prediction()
    {
        sdk::prediction = reinterpret_cast< pred_sdk* >( g_sdk->get_pred_sdk() );

        return sdk::prediction != nullptr;
    }
}