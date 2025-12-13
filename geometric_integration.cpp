/**
 * COMPLETE INTEGRATED get_prediction() for GeometricPrediction.h
 *
 * This is the complete rewrite that integrates:
 * - EdgeCaseDetection.h (stasis, dash, windwall, minion, clone)
 * - PredictionSettings.h (user settings)
 * - PredictionTelemetry.h (performance tracking)
 *
 * Replace the existing get_prediction() and calculate_hitchance() with this code.
 */

namespace GeometricPred
{
    /**
     * Helper: Convert HitChance enum to float for telemetry
     */
    inline float hit_chance_to_float(HitChance hc)
    {
        switch(hc)
        {
            case HitChance::Impossible:    return 0.0f;
            case HitChance::Clone:         return 0.0f;
            case HitChance::SpellShielded: return 0.0f;
            case HitChance::Windwalled:    return 0.0f;
            case HitChance::MinionBlocked: return 0.0f;
            case HitChance::Low:           return 0.35f;
            case HitChance::Medium:        return 0.55f;
            case HitChance::High:          return 0.75f;
            case HitChance::VeryHigh:      return 0.90f;
            case HitChance::Undodgeable:   return 1.0f;
            case HitChance::Immobile:      return 1.0f;
            case HitChance::Stasis:        return 1.0f;
            case HitChance::Channeling:    return 0.95f;
            case HitChance::Dashing:       return 0.80f;
            default:                       return 0.5f;
        }
    }

    /**
     * THE DRIVER: Main entry point for prediction
     *
     * Fully integrated with EdgeCases, Settings, and Telemetry
     */
    inline PredictionResult get_prediction(const PredictionInput& input)
    {
        // Start timing for telemetry
        auto start_time = std::chrono::high_resolution_clock::now();

        PredictionResult result;

        // Validate input
        if (!input.source || !input.target)
        {
            result.block_reason = "Invalid source or target";
            result.hit_chance = HitChance::Impossible;
            result.should_cast = false;
            if (PredictionSettings::get().enable_telemetry)
            {
                PredictionTelemetry::TelemetryLogger::log_rejection_invalid_target();
            }
            return result;
        }

        // Store current positions for telemetry
        result.target_current_pos = input.target->get_server_position();
        result.distance_to_target = Utils::distance_2d(input.source->get_position(), result.target_current_pos);
        result.target_is_moving = input.target->is_moving();
        result.target_velocity = input.target->get_move_speed();

        // =======================================================================================
        // EDGE CASE ANALYSIS (Uses EdgeCaseDetection.h)
        // =======================================================================================
        auto edge_analysis = EdgeCases::analyze_target(input.target, input.source);

        // 1. CLONE DETECTION
        if (edge_analysis.is_clone)
        {
            result.block_reason = "Target is a clone";
            result.hit_chance = HitChance::Clone;
            result.should_cast = false;
            result.is_clone_target = true;
            return result;
        }

        // 2. SPELL SHIELD DETECTION
        if (edge_analysis.has_shield)
        {
            result.block_reason = "Target has spell shield";
            result.hit_chance = HitChance::SpellShielded;
            result.should_cast = false;  // Don't waste spell
            result.spell_shield_detected = true;
            return result;
        }

        // 3. WINDWALL DETECTION
        if (edge_analysis.blocked_by_windwall)
        {
            result.block_reason = "Windwall blocks spell path";
            result.hit_chance = HitChance::Windwalled;
            result.should_cast = false;
            result.windwall_detected = true;
            return result;
        }

        // 4. STASIS HANDLING
        if (edge_analysis.stasis.is_in_stasis)
        {
            float current_time = g_sdk->clock_facade ? g_sdk->clock_facade->get_game_time() : 0.f;

            // Calculate spell travel time to stasis position
            float travel_time = Utils::compute_arrival_time(
                input.source->get_position(),
                edge_analysis.stasis.exit_position,
                input.missile_speed,
                input.cast_delay,
                input.proc_delay
            );

            // Calculate optimal timing
            float wait_time = EdgeCases::calculate_stasis_cast_timing(
                edge_analysis.stasis,
                travel_time,
                current_time
            );

            if (wait_time < 0.f)
            {
                result.block_reason = "Stasis timing impossible (spell too slow)";
                result.hit_chance = HitChance::Impossible;
                result.should_cast = false;
                return result;
            }

            // Stasis prediction
            result.cast_position = edge_analysis.stasis.exit_position;
            result.predicted_position = edge_analysis.stasis.exit_position;
            result.hit_chance = HitChance::Stasis;
            result.should_cast = (wait_time <= 0.f);  // Cast now or wait
            result.stasis_wait_time = wait_time;
            result.is_stasis = true;
            result.edge_case_type = "stasis";
            result.hit_chance_float = 1.0f;  // Guaranteed if timed correctly
            return result;
        }

        // 5. DASH HANDLING (if enabled)
        if (PredictionSettings::get().enable_dash_prediction && edge_analysis.dash.is_dashing)
        {
            float current_time = g_sdk->clock_facade ? g_sdk->clock_facade->get_game_time() : 0.f;

            // Predict to dash endpoint
            math::vector3 dash_end = edge_analysis.dash.dash_end_position;

            // Calculate spell arrival time to dash endpoint
            float spell_arrival = Utils::compute_arrival_time(
                input.source->get_position(),
                dash_end,
                input.missile_speed,
                input.cast_delay,
                input.proc_delay
            );

            // Validate timing
            if (EdgeCases::validate_dash_timing(edge_analysis.dash, spell_arrival, current_time))
            {
                result.cast_position = dash_end;
                result.predicted_position = dash_end;
                result.hit_chance = HitChance::Dashing;
                result.should_cast = true;
                result.is_dash = true;
                result.dash_confidence = edge_analysis.dash.confidence_multiplier;
                result.edge_case_type = "dash";
                result.hit_chance_float = edge_analysis.dash.confidence_multiplier;
                return result;
            }
            else
            {
                // Spell arrives before dash ends - don't predict to endpoint
                result.block_reason = "Spell arrives before dash completes";
                result.hit_chance = HitChance::Low;
                result.should_cast = false;
                return result;
            }
        }

        // 6. CHANNEL/RECALL DETECTION
        if (edge_analysis.channel.is_channeling || edge_analysis.channel.is_recalling)
        {
            // High priority stationary target
            result.cast_position = edge_analysis.channel.position;
            result.predicted_position = edge_analysis.channel.position;
            result.hit_chance = HitChance::Channeling;
            result.should_cast = true;
            result.is_channeling = true;
            result.edge_case_type = "channeling";
            result.hit_chance_float = 0.95f;
            return result;
        }

        // =======================================================================================
        // NORMAL PREDICTION (Linear path following + geometric TTE)
        // =======================================================================================

        // Get animation lock time
        float animation_lock = Utils::get_animation_lock_time(input.target);
        result.is_animation_locked = (animation_lock > 0.f);

        // Predict position
        float prediction_time = Utils::compute_arrival_time(
            input.source->get_position(),
            input.target->get_position(),  // Initial guess
            input.missile_speed,
            input.cast_delay,
            input.proc_delay
        );

        math::vector3 predicted_pos = predict_linear_path(
            input.target,
            prediction_time,
            animation_lock
        );

        result.cast_position = predicted_pos;
        result.predicted_position = predicted_pos;
        result.prediction_offset = Utils::distance_2d(result.target_current_pos, predicted_pos);

        // Check if target is slowed (confidence boost)
        result.is_slowed = edge_analysis.is_slowed;

        // =======================================================================================
        // MINION COLLISION CHECK (uses EdgeCaseDetection.h)
        // =======================================================================================
        if (input.shape == SpellShape::Capsule)
        {
            // Use EdgeCases minion collision with health prediction
            // Champion script decides if spell collides via input parameter
            float minion_clear = EdgeCases::compute_minion_block_probability(
                input.source->get_position(),
                predicted_pos,
                input.spell_width,
                true  // This spell collides with minions
            );

            // Deterministic check: 0.0 = blocked, 1.0 = clear
            if (minion_clear == 0.0f)
            {
                result.block_reason = "Minion blocks spell path";
                result.hit_chance = HitChance::MinionBlocked;
                result.should_cast = false;
                result.minion_collision = true;

                if (PredictionSettings::get().enable_telemetry)
                {
                    PredictionTelemetry::TelemetryLogger::log_rejection_collision();
                }
                return result;
            }
        }

        // =======================================================================================
        // GEOMETRIC HIT CHANCE CALCULATION
        // =======================================================================================

        // Calculate time to impact
        float time_to_impact = Utils::compute_arrival_time(
            input.source->get_position(),
            predicted_pos,
            input.missile_speed,
            input.cast_delay,
            input.proc_delay
        );
        result.time_to_impact = time_to_impact;

        // Distance to exit calculation (shape-dependent)
        float target_radius = input.target->get_bounding_radius();
        float distance_to_exit = 0.f;

        if (input.shape == SpellShape::Circle)
        {
            distance_to_exit = Utils::calculate_escape_distance_circle(
                predicted_pos,
                result.cast_position,
                input.spell_width,
                target_radius
            );
        }
        else  // Capsule
        {
            math::vector3 spell_direction = (predicted_pos - input.source->get_position());
            float spell_length = Utils::magnitude_2d(spell_direction);
            if (spell_length > EPSILON)
                spell_direction = spell_direction / spell_length;
            else
                spell_direction = math::vector3(0.f, 0.f, 1.f);

            distance_to_exit = Utils::calculate_escape_distance_capsule(
                predicted_pos,
                input.source->get_position(),
                spell_direction,
                input.spell_width,
                input.spell_range,
                target_radius
            );
        }
        result.distance_to_exit = distance_to_exit;

        // Time needed to dodge
        float move_speed = input.target->get_move_speed();
        if (move_speed <= 0.0f)
        {
            result.hit_chance = HitChance::Immobile;
            result.should_cast = true;
            result.is_immobile = true;
            result.hit_chance_float = 1.0f;
            return result;
        }

        float time_needed_to_dodge = animation_lock + (distance_to_exit / move_speed);

        // FOW visibility adjustment (uses EdgeCases logic)
        float effective_reaction_reduction = 0.f;
        if (input.source && g_sdk && g_sdk->nav_mesh)
        {
            int enemy_team = input.target->get_team_id();
            bool hidden = g_sdk->nav_mesh->is_in_fow_for_team(
                input.source->get_position(), enemy_team);

            if (!hidden)
            {
                effective_reaction_reduction = input.cast_delay;
            }
        }

        // Reaction window calculation
        float reaction_window = time_to_impact - time_needed_to_dodge - effective_reaction_reduction;
        result.reaction_window = reaction_window;

        // Terrain blocking multiplier
        float terrain_multiplier = Utils::get_terrain_blocking_multiplier(
            predicted_pos,
            result.cast_position,
            distance_to_exit
        );

        // Trapped = forced Undodgeable
        if (terrain_multiplier >= 1.9f)
        {
            result.hit_chance = HitChance::Undodgeable;
            result.should_cast = true;
            result.hit_chance_float = 1.0f;
            return result;
        }

        // Apply terrain and slow multipliers
        reaction_window /= terrain_multiplier;
        if (result.is_slowed)
        {
            reaction_window /= 1.15f;  // Confidence boost for slowed targets
        }

        // Grade reaction window to hit chance
        if (reaction_window <= REACTION_UNDODGEABLE)
            result.hit_chance = HitChance::Undodgeable;
        else if (reaction_window <= REACTION_VERY_HIGH)
            result.hit_chance = HitChance::VeryHigh;
        else if (reaction_window <= REACTION_HIGH)
            result.hit_chance = HitChance::High;
        else if (reaction_window <= REACTION_MEDIUM)
            result.hit_chance = HitChance::Medium;
        else
            result.hit_chance = HitChance::Low;

        result.hit_chance_float = hit_chance_to_float(result.hit_chance);

        // Simple recommendation: Cast on Medium or higher
        result.should_cast = (
            result.hit_chance == HitChance::Medium ||
            result.hit_chance == HitChance::High ||
            result.hit_chance == HitChance::VeryHigh ||
            result.hit_chance == HitChance::Undodgeable
        );

        // =======================================================================================
        // TELEMETRY LOGGING
        // =======================================================================================
        auto end_time = std::chrono::high_resolution_clock::now();
        result.computation_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

        if (PredictionSettings::get().enable_telemetry)
        {
            PredictionTelemetry::PredictionEvent event;
            event.timestamp = g_sdk->clock_facade ? g_sdk->clock_facade->get_game_time() : 0.f;
            event.target_name = input.target->get_char_name();
            event.spell_type = (input.shape == SpellShape::Circle) ? "Circle" : "Capsule";
            event.hit_chance = result.hit_chance_float;
            event.confidence = result.hit_chance_float;  // Geometric has no separate confidence
            event.distance = result.distance_to_target;
            event.was_dash = result.is_dash;
            event.was_stationary = !result.target_is_moving;
            event.was_animation_locked = result.is_animation_locked;
            event.collision_detected = result.minion_collision || result.windwall_detected;
            event.computation_time_ms = result.computation_time_ms;
            event.edge_case = result.edge_case_type;

            // Spell configuration
            event.spell_range = input.spell_range;
            event.spell_radius = input.spell_width;
            event.spell_delay = input.cast_delay;
            event.spell_speed = input.missile_speed;

            // Movement data
            event.target_velocity = result.target_velocity;
            event.prediction_offset = result.prediction_offset;
            event.target_is_moving = result.target_is_moving;

            // Position data
            event.source_pos_x = input.source->get_position().x;
            event.source_pos_z = input.source->get_position().z;
            event.target_server_pos_x = result.target_current_pos.x;
            event.target_server_pos_z = result.target_current_pos.z;
            event.predicted_pos_x = result.predicted_position.x;
            event.predicted_pos_z = result.predicted_position.z;
            event.cast_pos_x = result.cast_position.x;
            event.cast_pos_z = result.cast_position.z;

            // Timing data
            event.final_arrival_time = result.time_to_impact;
            event.dodge_time = result.reaction_window;
            event.effective_move_speed = move_speed;

            PredictionTelemetry::TelemetryLogger::log_prediction(event);
        }

        return result;
    }

} // namespace GeometricPred
