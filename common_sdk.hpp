#pragma once

#include "sdk.hpp"

class common_sdk
{
public:
	struct range_interface
	{
		virtual bool is_in_range( game_object* source, game_object* target, float range ) = 0;
		virtual bool is_in_turret_range( const math::vector3& pos, int team_id = -1, game_object* specific_turret = nullptr ) = 0;
	};

	struct buffs_interface
	{
		virtual std::vector<buff_instance*> get_buffs_by_hashes( game_object* object, const std::vector< uint32_t >& hashes ) = 0;
		virtual std::vector<buff_instance*> get_buffs_by_types( game_object* object, const std::vector< buff_type >& types ) = 0;
		virtual buff_instance* get_stasis_buff( game_object* object ) = 0;
		virtual float get_revive_buff_end_time( game_object* object ) = 0;
	};

	struct physics_interface
	{
		virtual math::vector3 get_closest_point_outside_rectangle( const math::vector3& pos, const math::vector3& start, const math::vector3& end, float radius ) = 0;
		virtual math::vector3 get_closest_point_outside_circle( const math::vector3& pos, const math::vector3& center, float radius ) = 0;
		virtual math::vector3 get_closest_point_outside_cone( const math::vector3& pos, const math::vector3& start, const math::vector3& end, float angle ) = 0;
		virtual bool is_in_segment( const math::vector3& pos, const math::vector3& start, const math::vector3& end, float radius ) = 0;
		virtual bool is_in_circle( const math::vector3& pos, const math::vector3& center, float radius ) = 0;
		virtual bool is_in_cone( const math::vector3& pos, const math::vector3& start, const math::vector3& end, float angle, float additional_radius = 0.f ) = 0;
	};

	struct spell_interface
	{
		virtual bool is_spell_ready( int slot, float in = 0.f ) = 0;
		virtual bool is_spell_learned( int slot ) = 0;
		virtual bool is_spell_active() = 0;
	};

	virtual bool issue_order( game_object_order order_type, const math::vector3& position, bool move_pet = false ) = 0;
	virtual bool issue_order( game_object_order order_type, game_object* target, bool move_pet = false ) = 0;
	virtual bool cast_spell( int spell_slot ) = 0;
	virtual bool cast_spell( int spell_slot, const math::vector3& cast_position ) = 0;
	virtual bool cast_spell( int spell_slot, const math::vector3& start_position, const math::vector3& end_position ) = 0;
	virtual bool cast_spell( int spell_slot, game_object* target ) = 0;
	virtual bool update_chargeable_spell( int spell_slot, const math::vector3& position, bool release_cast ) = 0;
	virtual bool use_object( game_object* object ) = 0;
	virtual float get_ping( float multiply = 0.f ) = 0;

	range_interface* range = nullptr;
	buffs_interface* buffs = nullptr;
	physics_interface* physics = nullptr;
	spell_interface* spell = nullptr;
};

namespace sdk
{
	inline common_sdk* common = nullptr;
}

namespace sdk_init
{
	inline bool common()
	{
		if( sdk::common )
			return true;

		const std::string module_name = "VEN.Common";
		if( !g_sdk->add_dependency( "Core/" + module_name ) )
			return false;

		sdk::common = reinterpret_cast< common_sdk* >( g_sdk->get_custom_sdk( module_name ) );

		return true;
	}
}