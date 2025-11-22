#pragma once

#include <functional>
#include <memory>
#include <string>
#include <span>
#include <map>

#include "math.hpp"

enum class module_type : uint8_t
{
	any,
	orb,
	pred,
	evade
};

// Enums
//
enum class game_object_order
{
	none = 0,
	hold_position,
	move_to,
	attack_unit,
	auto_attack_pet,
	auto_attack,
	move_pet,
	attack_to,
	stop = 10
};

enum class buff_type : uint8_t
{
	internal = 0,
	aura = 1,
	combat_enchancer = 2,
	combat_dehancer = 3,
	spell_shield = 4,
	stun = 5,
	invisibility = 6,
	silence = 7,
	taunt = 8,
	berserk = 9,
	polymorph = 10,
	slow = 11,
	snare = 12,
	damage = 13,
	heal = 14,
	haste = 15,
	spell_immunity = 16,
	physical_immunity = 17,
	invulnerability = 18,
	attack_speed_slow = 19,
	near_sight = 20,
	currency = 21,
	fear = 22,
	charm = 23,
	poison = 24,
	suppression = 25,
	blind = 26,
	counter = 27,
	shred = 28,
	flee = 29,
	knockup = 30,
	knockback = 31,
	disarm = 32,
	grounded = 33,
	drowsy = 34,
	asleep = 35,
	obscured = 36,
	clickproof_to_enemies = 37,
	unkillable = 38
};

enum class map_ping_type : uint8_t
{
	ping = 1,
	old_caution = 2,
	enemy_missing = 3,
	on_my_way = 4,
	caution = 5,
	assist_me = 6,
	enemy_has_vision = 7,
	invisible_assist_me_yellow = 8,
	invisible_assist_me_red = 9,
	game_mode_info = 10,
	game_mode_important = 11,
	game_mode_warning = 12,
	enemy_has_no_vision = 13,
	ask_for_vision = 14,
	push_forward = 15,
	all_in = 16,
	reset = 17,
	retreat = 18,
	hold_the_area = 19
};

enum class emote_type : uint8_t
{
	dance = 0,
	taunt = 1,
	laugh = 2,
	joke = 3,
	toggle = 4
};

enum class item_param : uint8_t
{
	item_effect_1,
	item_effect_2,
	item_effect_3,
	item_effect_4,
	item_effect_5,
	item_effect_6,
	item_effect_7,
	item_effect_8,
	item_effect_9,
	item_effect_10,
	item_effect_11,
	item_effect_12,
	item_effect_13,
	item_effect_14,
	item_effect_15,
	item_effect_16,
	item_effect_17,
	item_effect_18,
	item_effect_19,
	item_effect_20,
	item_effect_21,
	item_effect_22,
	item_effect_23,
	item_effect_24,
	item_enchantment_effect_1,
	item_enchantment_effect_2,
	item_enchantment_effect_3,
	item_enchantment_effect_4,
	item_enchantment_effect_5,
	item_enchantment_effect_6,
	item_enchantment_effect_7,
	item_enchantment_effect_8,
	item_enchantment_effect_9,
	item_enchantment_effect_10,
	item_enchantment_effect_11,
	item_enchantment_effect_12,
	item_enchantment_effect_13,
	item_enchantment_effect_14,
	item_enchantment_effect_15,
	item_enchantment_effect_16,
	item_enchantment_effect_17,
	item_enchantment_effect_18,
	item_enchantment_effect_19,
	item_enchantment_effect_20,
	item_enchantment_effect_21,
	item_enchantment_effect_22,
	item_enchantment_effect_23,
	item_enchantment_effect_24,
	flat_cooldown,
	percent_cooldown,
	ability_haste,
	flat_hp,
	percent_hp,
	flat_hp_regen,
	percent_hp_regen,
	percent_base_hp_regen,
	flat_mana,
	percent_base_mana_regen,
	percent_tenacity_item,
	flat_movement_speed,
	percent_movement_speed,
	percent_multiplicative_movement,
	flat_armor,
	percent_armor,
	flat_armor_pen,
	percent_armor_pen,
	percent_bonus_armor_pen,
	flat_magic_pen,
	percent_magic_pen,
	percent_bonus_magic_pen,
	flat_spell_block,
	percent_spell_block,
	flat_dodge,
	flat_crit_chance,
	flat_miss_chance,
	flat_crit_damage,
	percent_crit_damage,
	flat_physical_damage,
	percent_physical_damage,
	flat_magic_damage,
	percent_magic_damage,
	flat_physical_reduction,
	percent_physical_reduction,
	flat_magic_reduction,
	percent_magic_reduction,
	percent_exp_bonus,
	flat_attack_range,
	percent_attack_range,
	flat_cast_range,
	percent_cast_range,
	percent_attack_speed,
	percent_multiplicative_attack_speed,
	percent_healing_amount,
	percent_life_steal,
	percent_spell_vamp,
	percent_omnivamp,
	percent_physical_vamp,
	percent_spell_effectiveness,
	flat_bubble_radius,
	percent_bubble_radius,
	percent_slow_resist,
	item_effect_by_level_table,
	required_buff_currency_cost,
	max_stack,
	total_cost,
	physical_lethality,
	wip_stat_1,
	wip_stat_2,
	wip_stat_3,
	wip_stat_4,
	wip_stat_5,
	wip_stat_6
};

enum class location_id : uint8_t
{
	top_lane = 0,
	mid_lane = 1,
	bot_lane = 2,
	jungle = 3,
	base = 4
};

// SDK classes
//
class game_object;
class active_spell_cast;
class spell_cast;

class spell_static_data
{
public:
	virtual int get_flags() = 0;
	virtual char* get_name() = 0;
	virtual char* get_alternate_name() = 0;
	virtual char* get_animation_name() = 0;
	virtual float* get_cooldown() = 0;
	virtual void* get_missile_spec() = 0;
	virtual float get_missile_speed() = 0;
	virtual uint32_t get_missile_effect_key() = 0;
	virtual int get_cast_type() = 0;
	virtual std::vector< char* > get_tags() = 0;
	virtual float get_radius() = 0;
	virtual float* get_cast_range() = 0;
	virtual float* get_cast_radius() = 0;
	virtual float get_travel_time() = 0;
	virtual float get_cast_time() = 0;
	virtual float* get_channel_duration() = 0;
	virtual float* get_cast_range_growth_max() = 0;
	virtual float* get_cast_range_growth_duration() = 0;
	virtual float get_charge_update_interval() = 0;
	virtual bool can_cast_while_disabled() = 0;
	virtual bool can_cast_or_queue_while_casting() = 0;
	virtual bool cant_cancel_while_winding_up() = 0;
	virtual bool cant_cancel_while_channeling() = 0;
	virtual bool cant_cast_while_rooted() = 0;
	virtual bool channel_is_interrupted_by_disables() = 0;
	virtual bool project_target_to_cast_range() = 0;
	virtual bool cast_range_use_bounding_boxes() = 0;
	virtual bool is_chargeable() = 0;
	virtual bool can_move_while_channeling() = 0;
	virtual bool have_hit_effect() = 0;
	virtual bool have_hit_bone() = 0;
	virtual bool is_toggle_spell() = 0;
	virtual bool ignore_range_check() = 0;
	virtual float get_override_attack_time() = 0;
	virtual bool use_autoattack_cast_time_data() = 0;
	virtual bool considered_as_auto_attack() = 0;
	virtual float get_cast_cone_angle() = 0;
	virtual float get_cast_cone_distance() = 0;
	virtual float get_cast_target_additional_units_radius() = 0;
	virtual char* get_missile_effect_name() = 0;
	virtual char* get_hit_bone_name() = 0;
};

class spell_data
{
public:
	virtual void* get_icon() = 0;
	virtual spell_static_data* get_static_data() = 0;
};

class spell_entry
{
public:
	virtual uint8_t get_type() = 0;
	virtual int get_level() = 0;
	virtual spell_data* get_data() = 0;
	virtual float get_cooldown() = 0;
	virtual float get_cast_range() = 0;
	virtual uint8_t get_toggle_state() = 0;
	virtual uint32_t get_charges() = 0;
};

class buff_instance
{
public:
	virtual buff_type get_type() = 0;
	virtual float get_start_time() = 0;
	virtual float get_end_time() = 0;
	virtual int get_count() = 0;
	virtual uint8_t get_stacks() = 0;
	virtual std::string get_name() = 0;
	virtual uint32_t get_hash() = 0;
	virtual bool is_active() = 0;
	virtual game_object* get_caster() = 0;
};

class game_object
{
public:
	virtual uint32_t get_id() = 0;
	virtual int get_team_id() = 0;
	virtual std::string get_name() = 0;
	virtual uint32_t get_network_id() = 0;
	virtual math::vector3 get_min_bounding_box() = 0;
	virtual math::vector3 get_max_bounding_box() = 0;
	virtual math::vector3 get_position() = 0;
	virtual float get_par() = 0;
	virtual float get_max_par() = 0;
	virtual int get_par_enabled() = 0;
	virtual int get_par_state() = 0;
	virtual float get_sar() = 0;
	virtual float get_max_sar() = 0;
	virtual int get_sar_enabled() = 0;
	virtual int get_sar_state() = 0;
	virtual float get_hp() = 0;
	virtual float get_max_hp() = 0;
	virtual float get_hp_max_penalty() = 0;
	virtual float get_all_shield() = 0;
	virtual float get_physical_shield() = 0;
	virtual float get_magical_shield() = 0;
	virtual float get_champ_specific_health() = 0;
	virtual float get_stop_shield_fade() = 0;
	virtual math::vector3 get_direction() = 0;
	virtual std::string get_char_name() = 0;
	virtual int get_evolve_points() = 0;
	virtual int get_evolve_flag() = 0;
	virtual int get_level() = 0;
	virtual float get_experience() = 0;
	virtual float get_experience_percent() = 0;
	virtual int get_skill_points() = 0;
	virtual int get_current_plates() = 0;
	virtual int get_max_plates() = 0;

	virtual float get_percent_cooldown_mod() = 0;
	virtual float get_ability_haste_mod() = 0;
	virtual float get_percent_cooldown_cap_mod() = 0;
	virtual float get_passive_cooldown_end_time() = 0;
	virtual float get_passive_cooldown_total_time() = 0;
	virtual float get_percent_damage_to_barracks_minion_mod() = 0;
	virtual float get_flat_damage_reduction_from_barracks_minion_mod() = 0;
	virtual float get_increased_move_speed_minion_mod() = 0;
	virtual float get_flat_physical_damage_mod() = 0;
	virtual float get_percent_physical_damage_mod() = 0;
	virtual float get_percent_bonus_physical_damage_mod() = 0;
	virtual float get_percent_base_physical_damage_as_flat_bonus_mod() = 0;
	virtual float get_flat_magic_damage_mod() = 0;
	virtual float get_percent_magic_damage_mod() = 0;
	virtual float get_flat_magic_reduction() = 0;
	virtual float get_percent_magic_reduction() = 0;
	virtual float get_flat_cast_range_mod() = 0;
	virtual float get_attack_speed_mod() = 0;
	virtual float get_percent_attack_speed_mod() = 0;
	virtual float get_percent_multiplicative_attack_speed_mod() = 0;
	virtual float get_base_attack_damage() = 0;
	virtual float get_base_attack_damage_sans_percent_scale() = 0;
	virtual float get_flat_base_attack_damage_mod() = 0;
	virtual float get_percent_base_attack_damage_mod() = 0;
	virtual float get_base_ability_damage() = 0;
	virtual float get_crit_damage_multiplier() = 0;
	virtual float get_scale_skin_coef() = 0;
	virtual float get_dodge() = 0;
	virtual float get_crit() = 0;
	virtual float get_flat_base_hp_pool_mod() = 0;
	virtual float get_armor() = 0;
	virtual float get_bonus_armor() = 0;
	virtual float get_magic_resist() = 0;
	virtual float get_bonus_magic_resist() = 0;
	virtual float get_hp_regen_rate() = 0;
	virtual float get_base_hp_regen_rate() = 0;
	virtual float get_move_speed() = 0;
	virtual float get_move_speed_base_increase() = 0;
	virtual float get_attack_range() = 0;
	virtual float get_flat_bubble_radius_mod() = 0;
	virtual float get_percent_bubble_radius_mod() = 0;
	virtual float get_flat_armor_penetration() = 0;
	virtual float get_physical_lethality() = 0;
	virtual float get_percent_armor_penetration() = 0;
	virtual float get_percent_bonus_armor_penetration() = 0;
	virtual float get_percent_crit_bonus_armor_penetration() = 0;
	virtual float get_percent_crit_total_armor_penetration() = 0;
	virtual float get_flat_magic_penetration() = 0;
	virtual float get_magic_lethality() = 0;
	virtual float get_percent_magic_penetration() = 0;
	virtual float get_percent_bonus_magic_penetration() = 0;
	virtual float get_percent_life_steal_mod() = 0;
	virtual float get_percent_spell_vamp_mod() = 0;
	virtual float get_percent_omnivamp_mod() = 0;
	virtual float get_percent_physical_vamp() = 0;
	virtual float get_pathfinding_radius_mod() = 0;
	virtual float get_percent_cc_reduction() = 0;
	virtual float get_percent_exp_bonus() = 0;
	virtual float get_primary_ar_regen_rate_rep() = 0;
	virtual float get_primary_ar_base_regen_rate_rep() = 0;
	virtual float get_secondary_ar_regen_rate_rep() = 0;
	virtual float get_secondary_ar_base_regen_rate_rep() = 0;

	virtual float get_attack_damage() = 0;
	virtual float get_ability_power() = 0;

	virtual bool is_melee() = 0;
	virtual bool is_ranged() = 0;

	virtual bool is_valid() = 0;
	virtual bool is_ai() = 0;
	virtual bool is_hero() = 0;
	virtual bool is_minion() = 0;
	virtual bool is_missile() = 0;
	virtual bool is_turret() = 0;
	virtual bool is_nexus() = 0;
	virtual bool is_inhibitor() = 0;
	virtual bool is_particle() = 0;
	virtual bool is_champion_clone() = 0;
	virtual bool is_lane_minion() = 0;
	virtual bool is_lane_minion_melee() = 0;
	virtual bool is_lane_minion_ranged() = 0;
	virtual bool is_lane_minion_siege() = 0;
	virtual bool is_lane_minion_super() = 0;
	virtual bool is_monster() = 0;
	virtual bool is_epic_monster() = 0;
	virtual bool is_large_monster() = 0;
	virtual bool is_medium_monster() = 0;
	virtual bool is_buff_monster() = 0;
	virtual bool is_trap() = 0;
	virtual bool is_ward() = 0;
	virtual bool is_plant() = 0;
	virtual bool is_dead() = 0;
	virtual bool is_zombie() = 0;
	virtual bool is_visible() = 0;
	virtual bool is_targetable() = 0;
	virtual float get_attack_delay() = 0;
	virtual float get_attack_cast_delay() = 0;
	virtual game_object* get_attacher() = 0;
	virtual game_object* get_owner() = 0;
	virtual float get_bounding_radius() = 0;
	virtual spell_entry* get_spell( int spell_slot ) = 0;
	virtual spell_data* get_basic_attack() = 0;
	virtual float get_basic_attack_cooldown_expiration() = 0;
	virtual math::vector2 get_health_bar_position() = 0;
	virtual void* get_icon_circle() = 0;
	virtual void* get_icon_square() = 0;
	virtual float get_raw_spell_value( int spell_slot, uint32_t value_index ) = 0;
	virtual std::vector< spell_static_data* > get_child_spells( int spell_slot ) = 0;
	virtual int get_spell_cast_state( int spell_slot ) = 0;
	virtual active_spell_cast* get_active_spell_cast() = 0;
	virtual void set_skin( int skin_id ) = 0;
	virtual void issue_order( game_object_order order_type, math::vector3 position, bool move_pet = false ) = 0;
	virtual void issue_order( game_object_order order_type, game_object* target, bool move_pet = false ) = 0;
	virtual void cast_spell( int spell_slot ) = 0;
	virtual void cast_spell( int spell_slot, math::vector3 cast_position ) = 0;
	virtual void cast_spell( int spell_slot, math::vector3 start_position, math::vector3 end_position ) = 0;
	virtual void cast_spell( int spell_slot, game_object* target ) = 0;
	virtual void update_chargeable_spell( int spell_slot, math::vector3 position, bool release_cast ) = 0;
	virtual bool use_object( game_object* target ) = 0;

	virtual bool has_buff_of_type( buff_type type ) = 0;
	virtual std::vector< buff_instance* > get_buffs() = 0;
	virtual buff_instance* get_buff_by_hash( uint32_t hash ) = 0;
	virtual buff_instance* get_buff_by_name( std::string& name ) = 0;

	virtual bool is_moving() = 0;
	virtual bool is_dashing() = 0;
	virtual float get_dash_speed() = 0;
	virtual math::vector3 get_server_position() = 0;
	virtual uint8_t get_current_path_index() = 0;
	virtual std::span< math::vector3 > get_path() = 0;
	virtual std::vector< math::vector3 > calculate_path( math::vector3 const& end_position ) = 0;

	virtual spell_cast* get_missile_spell_cast() = 0;
	virtual math::vector3 get_missile_start_pos() = 0;
	virtual math::vector3 get_missile_end_pos() = 0;

	virtual math::vector3 get_particle_direction() = 0;

	virtual bool is_in_bush() = 0;
	virtual bool is_near_bush() = 0;

	virtual char* get_search_tags() = 0;
	virtual char* get_search_tags_secondary() = 0;
	virtual float get_pathfinding_collision_radius() = 0;

	virtual bool has_item( int item_id, int* found_slot ) = 0;
	virtual int get_item_id( int slot ) = 0;
	virtual bool buy_item( uint32_t item_id, uint8_t slot = 0xFF ) = 0;
	virtual void sell_item( uint8_t slot ) = 0;
	virtual bool undo_item() = 0;
	virtual bool swap_item( uint8_t slot, uint8_t new_slot ) = 0;

	virtual bool draw_outline( uint32_t color ) = 0;
	virtual bool draw_glow( uint32_t color ) = 0;

	virtual bool can_level_spell( int spell_slot ) = 0;
	virtual bool level_spell( int spell_slot ) = 0;

	virtual float get_respawn_time() = 0;
	virtual bool is_clone() = 0;
	virtual bool has_rune( uint32_t rune_id ) = 0;
	virtual uint32_t get_rune_id( uint8_t slot ) = 0;
	virtual bool can_evolve_spell( int spell_slot ) = 0;
	virtual game_object* get_turret_aggro_target() = 0;
	virtual float get_rune_value( uint8_t slot, uint32_t value_index ) = 0;
	virtual bool cast_hwei_mood( int mood_slot ) = 0;
	virtual float get_missile_current_speed() = 0;
	virtual math::vector3 get_velocity() = 0;
	virtual float get_lifetime() = 0;
	virtual float get_max_lifetime() = 0;

	virtual float get_spell_mana_cost( int spell_slot ) = 0;
	virtual float get_item_param_value( int slot, item_param param ) = 0;
};

class active_spell_cast
{
public:
	virtual float get_cast_end_time() = 0;
	virtual float get_cast_completion_time() = 0;
	virtual float get_cast_channeling_end_time() = 0;
	virtual float get_cast_start_time() = 0;
	virtual spell_cast* get_spell_cast() = 0;
};

class spell_cast
{
public:
	virtual spell_data* get_spell_data() = 0;
	virtual float get_process_time() = 0;
	virtual int get_attack_id() = 0;
	virtual math::vector3 get_start_pos() = 0;
	virtual math::vector3 get_end_pos() = 0;
	virtual math::vector3 get_cast_pos() = 0;
	virtual math::vector3 get_direction() = 0;
	virtual float get_cast_delay() = 0;
	virtual float get_cast_delay_mod() = 0;
	virtual float get_delay() = 0;
	virtual float get_cooldown() = 0;
	virtual bool is_basic_attack() = 0;
	virtual bool is_special_attack() = 0;
	virtual bool is_stopped() = 0;
	virtual int get_spell_slot() = 0;
	virtual game_object* get_source() = 0;
	virtual game_object* get_target() = 0;
	virtual bool is_network_synced() = 0;
	virtual uint32_t get_network_id() = 0;
};

class object_manager
{
public:
	virtual game_object* get_local_player() = 0;
	virtual std::span< game_object* > get_turrets() = 0;
	virtual std::span< game_object* > get_heroes() = 0;
	virtual std::span< game_object* > get_minions() = 0;
	virtual std::span< game_object* > get_nexuses() = 0;
	virtual std::span< game_object* > get_inhibitors() = 0;
	virtual std::span< game_object* > get_monsters() = 0;
	virtual std::span< game_object* > get_traps() = 0;
	virtual std::span< game_object* > get_wards() = 0;
	virtual std::span< game_object* > get_plants() = 0;

	virtual game_object* get_object_by_network_id( uint32_t network_id ) = 0;
};

class event_manager
{
public:
	// void __fastcall present();
	// void __fastcall wndproc( uint32_t msg, uint32_t wparam, uint32_t lparam );
	// void __fastcall game_update();
	// void __fastcall create_object( game_object* object );
	// void __fastcall delete_object( game_object* object );
	// void __fastcall create_missile( game_object* missile );
	// void __fastcall pre_create_particle( game_object* owner, char* name, uint32_t hash, char** overriden_name, uint32_t* overriden_hash );
	// void __fastcall create_particle( game_object* particle, game_object* owner, math::vector3 start_pos, math::vector3 end_pos );
	// void __fastcall basic_attack( game_object* object, game_object* target, spell_cast* cast );
	// void __fastcall stop_cast( game_object* object, spell_cast* cast, bool was_attack_processed );
	// void __fastcall process_cast( game_object* object, spell_cast* cast );
	// void __fastcall buff_gain( game_object* object, buff_instance* buff );
	// void __fastcall buff_loss( game_object* object, buff_instance* buff );
	// void __fastcall draw_world();
	// void __fastcall neutral_minion_kill( game_object* object, game_object* minion, int camp_side_team_id );
	// void __fastcall new_path( game_object* object, bool is_dash, float dash_speed );
	// void __fastcall execute_cast( game_object* object, spell_cast* cast );
	// bool __fastcall issue_order( game_object* object, game_object_order order_type, math::vector3 position, game_object* target, bool move_pet );
	// bool __fastcall cast_spell( game_object* object, int spell_slot, math::vector3 position, game_object* target );
	// void __fastcall packet( uint16_t packet_opcode, uint64_t packet_data, uint32_t rpc_network_id );
	// void __fastcall animation( game_object* object, uint32_t animation_hash );
	// void __fastcall cast_heal( game_object* object, game_object* target, float amount );
	// void __fastcall kill( game_object* object, game_object* killer );
	// void __fastcall draw_environment();
	// void __fastcall spell_hit( game_object* object, spell_cast* cast );

	enum event : uint8_t
	{
		present,
		wndproc,
		game_update,
		create_object,
		delete_object,
		create_missile,
		basic_attack,
		stop_cast,
		process_cast,
		buff_gain,
		buff_loss,
		draw_world,
		neutral_minion_kill,
		new_path,
		execute_cast,
		issue_order,
		cast_spell,
		packet,
		animation,
		cast_heal,
		kill,
		create_particle,
		pre_create_particle,
		draw_environment,
		spell_hit
	};

	virtual void register_callback( event id, void* callback ) = 0;
	virtual void unregister_callback( event id, void* callback ) = 0;
};

class clock_facade
{
public:
	virtual float get_game_time() = 0;
};

class hud_manager
{
public:
	virtual math::vector3 get_cursor_position() = 0;
	virtual game_object* get_hovered_target() = 0;

	virtual float get_camera_pitch() = 0;
	virtual float get_camera_yaw() = 0;
	virtual float get_camera_fov() = 0;
	virtual void set_camera_pitch( float pitch ) = 0;
	virtual void set_camera_yaw( float yaw ) = 0;
	virtual void set_camera_fov( float fov ) = 0;

	virtual void show_action_marker( math::vector3 const& position ) = 0;
};

class shader
{
public:
	virtual void set_constant_data( void* data, uint32_t size ) = 0;
	virtual void draw( math::vector3* vertices, uint32_t vertices_size ) = 0;
	virtual void draw_textured( math::vector3* vertices, uint32_t vertices_size, void* texture ) = 0;
};

class renderer
{
public:
	virtual void add_circle_2d( math::vector2 const& position, float radius, float thickness, uint32_t color ) = 0;
	virtual void add_circle_filled_2d( math::vector2 const& position, float radius, uint32_t color ) = 0;
	virtual void add_circle_3d( math::vector3 const& position, float radius, float thickness, uint32_t color ) = 0;
	virtual void add_circle_minimap( math::vector3 const& position, float radius, float thickness, uint32_t color ) = 0;
	virtual void add_rectangle( math::rect const& rectangle, uint32_t color, float thickness ) = 0;
	virtual void add_rectangle_filled( math::rect const& rectangle, uint32_t color ) = 0;
	virtual void add_text( std::string const& text, float size, math::vector2 const& position, uint32_t flags, uint32_t color ) = 0;
	virtual void add_line_2d( math::vector2 const& point1, math::vector2 const& point2, float thickness, uint32_t color ) = 0;
	virtual void add_sprite( void* sprite, math::vector2 const& position, uint32_t color, math::vector2 const& scale = { 1.f , 1.f } ) = 0;
	virtual void add_sprite_circle( void* sprite, math::vector2 const& position, uint32_t color, math::vector2 const& scale = { 1.f , 1.f } ) = 0;

	virtual int get_window_width() = 0;
	virtual int get_window_height() = 0;

	virtual math::vector2 world_to_screen( math::vector3 world_position ) = 0;
	virtual math::vector2 world_to_minimap( math::vector3 world_position ) = 0;

	virtual void add_damage_indicator( game_object* object, float damage ) = 0;
	virtual void add_heal_indicator( game_object* object, float heal ) = 0;
	virtual void add_shield_indicator( game_object* object, float shield ) = 0;

	virtual float* get_view_matrix() = 0;
	virtual float* get_projection_matrix() = 0;
	virtual shader* create_shader( void* shader_bytecode, size_t shader_size, uint32_t max_vertices, uint32_t constant_buffer_size = 0 ) = 0;
	virtual shader* create_shader( std::string const& shader_code, uint32_t max_vertices, uint32_t constant_buffer_size = 0 ) = 0;
	virtual std::tuple< void*, void* > create_texture( void* buffer, uint32_t width, uint32_t height, uint8_t channels ) = 0;
	virtual std::tuple< void*, uint32_t, uint32_t > lock_texture( void* texture ) = 0;
	virtual void unlock_texture( void* texture ) = 0;
	virtual void* create_effect( std::vector< uint8_t > const& vs_blob, std::vector< uint8_t > const& ps_blob ) = 0;

	virtual void* create_sprite( void* buffer, size_t size ) = 0;
};

class net_client
{
public:
	virtual int get_ping() = 0;
};

class nav_mesh
{
public:
	virtual bool is_pathable( math::vector3& position ) = 0;

	virtual math::vector3 get_terrain_start() = 0;
	virtual math::vector3 get_terrain_end() = 0;
	virtual uint32_t get_horizontal_chunks_count() = 0;
	virtual uint32_t get_vertical_chunks_count() = 0;
	virtual float get_chunk_width() = 0;
	virtual float get_chunk_scale() = 0;
	virtual float get_chunk_radius() = 0;

	virtual bool is_in_fow( math::vector3 const& position ) = 0;

	virtual float get_terrain_height( float x, float z ) = 0;

	virtual bool is_in_fow_for_team( math::vector3 const& position, int team_id ) = 0;
	virtual bool is_in_fow_ex( math::vector3 const& position, int team_id, bool skip_minions, bool skip_heroes ) = 0;
	virtual bool is_pathable_ex( math::vector3& position, uint32_t team ) = 0;

	virtual math::vector3 get_nearest_passable_position( math::vector3& position ) = 0;

	virtual bool is_stone( math::vector3 const& position ) = 0;
	virtual bool is_water( math::vector3 const& position ) = 0;
	virtual bool is_bush( math::vector3 const& position ) = 0;
	virtual location_id get_location( math::vector3 const& position ) = 0;
};

class menu_category
{
public:
	virtual void add_label( std::string const& text ) = 0;
	virtual void add_checkbox( std::string const& element_name, std::string const& element_display_name, bool default_value, std::function< void( bool ) > const& callback ) = 0;
	virtual void add_hotkey( std::string const& element_name, std::string const& element_display_name, unsigned char default_key, bool default_value = false,
							 bool toggle = false, std::function< void( std::string*, bool ) > const& callback = nullptr ) = 0;
	virtual void add_separator() = 0;
	virtual void add_slider_int( std::string const& element_name, std::string const& element_display_name, int min, int max, int step, int default_value,
								 std::function< void( int ) > const& callback = nullptr ) = 0;
	virtual void add_slider_float( std::string const& element_name, std::string const& element_display_name, float min, float max, float step, float default_value,
								   std::function< void( float ) > const& callback = nullptr ) = 0;
	virtual void add_combo( std::string const& element_name, std::string const& element_display_name, std::vector< std::string > const& items,
							int default_value, const std::function< void( int ) >& callback = nullptr ) = 0;
	virtual void add_colorpicker( std::string const& element_name, std::string const& element_display_name, uint32_t default_color,
								  std::function< void( uint32_t ) > const& callback = nullptr ) = 0;

	virtual menu_category* add_sub_category( std::string const& element_name, std::string const& element_display_name ) = 0;
};

class menu_manager
{
public:
	virtual menu_category* add_category( std::string const& name, std::string const& display_name ) = 0;
};

class game_info
{
public:
	virtual char* get_game_mode() = 0;
	virtual int get_map_id() = 0;
};

class game_settings_manager
{
public:
	virtual uint32_t get_show_summoner_names() = 0;
	virtual float get_minimap_scale() = 0;
};

class game_gui
{
public:
	virtual void send_chat( std::string const& message ) = 0;
	virtual void send_map_ping( math::vector3 const& position, map_ping_type type ) = 0;
	virtual void send_map_ping( game_object* target, map_ping_type type ) = 0;
	virtual bool send_emote( emote_type type ) = 0;
};

class core_sdk
{
public:
	object_manager* object_manager = nullptr;
	event_manager* event_manager = nullptr;
	clock_facade* clock_facade = nullptr;
	hud_manager* hud_manager = nullptr;
	renderer* renderer = nullptr;
	net_client* net_client = nullptr;
	nav_mesh* nav_mesh = nullptr;
	menu_manager* menu_manager = nullptr;
	game_info* game_info = nullptr;
	game_settings_manager* game_settings_manager = nullptr;
	game_gui* game_gui = nullptr;

	virtual void set_package( std::string const& name ) = 0;
	virtual bool add_dependency( std::string const& name ) = 0;

	virtual void* get_custom_sdk( std::string const& name ) = 0;

	virtual void* get_orb_sdk() = 0;
	virtual void* get_pred_sdk() = 0;
	virtual void* get_evade_sdk() = 0;

	virtual void log_console( const char* format, ... ) = 0;

	virtual std::string get_username() = 0;
	virtual uint32_t get_remaining_sub_days() = 0;

	virtual bool is_replay_mode() = 0;
	virtual char* get_engine_string( std::string const& engine_string ) = 0;
	virtual bool is_chat_open() = 0;
	virtual bool is_shop_open() = 0;

	virtual bool get_height_precise_drawings_state() = 0;

	virtual std::map< uint32_t, char* > get_buffs_hash_map() = 0;
};

inline core_sdk* g_sdk = nullptr;

constexpr auto SDK_VERSION = 2;
