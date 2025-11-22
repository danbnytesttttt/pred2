#pragma once

#include <sdk.hpp>
#include <color.hpp>
#include <array>

class sm_sdk
{
public:
    enum class spell_iteration: uint8_t {
        all = 0,
        ally = 1,
        enemy = 2,
    };

    enum class spell_type: uint8_t {
        unsupported = 0,
        linear = 1,
        circular = 2,
    };

    class static_data
    {
    public:
        spell_type type{};
        game_object* hero{};
        bool valid{};
        bool has_projectile{};
        bool strict_missile_name{};
        bool is_cc{};
        int slot = -1;
        std::array< uint32_t, 3 > particle_hashes{};
        std::string missile_name{};
        float projectile_speed{};
        float travel_time{};
        float range[ 7 ]{};
        float radius[ 7 ]{};
        std::vector< char* > tags{};

        virtual float get_radius() const
        {
            if( !this->valid )
                return 0.f;

            const auto spell_entry = this->hero->get_spell( this->slot );
            if( !spell_entry )
                return 0.f;

            return this->radius[ spell_entry->get_level() ];
        }

        virtual float get_range() const
        {
            if( !this->valid )
                return 0.f;

            const auto spell_entry = this->hero->get_spell( this->slot );
            if( !spell_entry )
                return 0.f;

            return this->range[ spell_entry->get_level() ];
        }

        virtual ~static_data() = default;
    };

    class spell {
    public:
        bool operator==( const spell& a )
        {
            return a.owner == this->owner && a.slot == this->slot;
        }

        int id{}; // Unique Spell ID created by Spell manager
        game_object* owner{}; // Spell caster
        game_object* target{}; // Target (for skills like Fizz R, Hwei R and in the future targetted spells)
        game_object* missile{}; // Missile if exists (can be nullptr)
        game_object* particle{}; // Particle if exists (can be nullptr)
        math::vector3 start_pos{}; // Spell start position
        math::vector3 end_pos{}; // Spell end position
        int slot{}; // Spell slot
        spell_type type{}; // Spell type
        int team_id{}; // Caster Team ID
        bool is_drawing_only{}; // Is drawing only
        bool is_cc{}; // Is CC
        bool is_particle_on_ground{}; // Is particle on ground
        bool has_projectile{}; // If the spell will create a projectile (Does not mean it has an active missile!!)
        bool missile_created{}; // If the missile has been created
        std::array< uint32_t, 3 > particle_hashes{}; // Particle hash
        float radius{};
        float projectile_speed{}; // If the spell has a projectile speed, otherwise 0 or FLT_MAX
        float travel_time{}; // If the spell has a static travel time, otherwise 0 or FLT_MAX
        float cast_delay{}; // Spell cast delay (not always set !)
        float cast_end_time{}; // Spell cast end time in game time
        float deletion_time{}; // Spell deletion time in game time
        float creation_time{}; // Spell creation time in game time
        std::string missile_name{}; // Spell missile name
        std::string trap_name{}; // Currently unused
        spell* parent_spell{}; // Parent spell (can be nullptr)
        std::vector< spell* > additional_spells{}; // Additional spells example for Leona R (center CC) or Return spells such as Swain E, Sivir Q, Ahri Q..
        bool from_fow{}; // If the spell was casted in Fog of War

        color color{}; // For internal use
        uint8_t previous_alpha = 0; // For internal use
        bool allow_position_changes = true; // For internal use
        bool delete_on_missile_deletion = true; // For internal use
        bool pending_deletion{}; // For internal use
    };

    virtual void iterate_spells( sm_sdk::spell_iteration iter, const std::function< bool( sm_sdk::spell* spell ) >& fn ) = 0; // fn can return true; to break; and stop iterating spells further
    virtual std::array< sm_sdk::static_data, 64 > get_spells_static_data( game_object* hero ) = 0; /* /!\ only use this on script initialization, not inside an event /!\ */
    virtual math::vector3 get_missile_position( sm_sdk::spell* spell ) = 0;
    virtual float get_deletion_time( sm_sdk::spell* spell ) = 0;
    virtual bool is_casted( sm_sdk::spell* spell ) = 0;
    virtual int get_spells_count( sm_sdk::spell_iteration iter ) = 0;
};

namespace sdk
{
    inline sm_sdk* spell_manager = nullptr;
}

namespace sdk_init
{
    inline bool spell_manager()
    {
        if( sdk::spell_manager )
            return true;

        const std::string module_name = "VEN.SpellManager";
        if( !g_sdk->add_dependency( "Core/" + module_name ) )
            return false;

        sdk::spell_manager = reinterpret_cast< sm_sdk* >( g_sdk->get_custom_sdk( module_name ) );

        return true;
    }
}