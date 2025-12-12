#include "HybridPrediction.h"
#include "EdgeCaseDetection.h"
#include "PredictionSettings.h"
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <random>
#include <sstream>

// Reasoning string generation (expensive - disable for production)
// Set to 0 to disable reasoning strings (saves ~0.02ms per prediction)
#define HYBRID_PRED_ENABLE_REASONING 0

namespace HybridPred
{
    // =========================================================================
    // BEHAVIOR PDF IMPLEMENTATION
    // =========================================================================

    float BehaviorPDF::sample(const math::vector3& world_pos) const
    {
        // Convert world position to grid coordinates
        float dx = world_pos.x - origin.x;
        float dz = world_pos.z - origin.z;

        int grid_x = static_cast<int>((dx / cell_size) + GRID_SIZE / 2);
        int grid_z = static_cast<int>((dz / cell_size) + GRID_SIZE / 2);

        // Check bounds
        if (grid_x < 0 || grid_x >= GRID_SIZE || grid_z < 0 || grid_z >= GRID_SIZE)
            return 0.f;

        // Bilinear interpolation for smoother sampling
        float fx = (dx / cell_size) + GRID_SIZE / 2 - grid_x;
        float fz = (dz / cell_size) + GRID_SIZE / 2 - grid_z;

        float v00 = pdf_grid[grid_x][grid_z];
        float v10 = (grid_x + 1 < GRID_SIZE) ? pdf_grid[grid_x + 1][grid_z] : 0.f;
        float v01 = (grid_z + 1 < GRID_SIZE) ? pdf_grid[grid_x][grid_z + 1] : 0.f;
        float v11 = (grid_x + 1 < GRID_SIZE && grid_z + 1 < GRID_SIZE) ?
            pdf_grid[grid_x + 1][grid_z + 1] : 0.f;

        float v0 = v00 * (1.f - fx) + v10 * fx;
        float v1 = v01 * (1.f - fx) + v11 * fx;

        return v0 * (1.f - fz) + v1 * fz;
    }

    void BehaviorPDF::normalize()
    {
        total_probability = 0.f;

        // Sum all probabilities
        for (int i = 0; i < GRID_SIZE; ++i)
        {
            for (int j = 0; j < GRID_SIZE; ++j)
            {
                total_probability += pdf_grid[i][j];
            }
        }

        // Normalize so sum = 1
        if (total_probability > EPSILON)
        {
            float scale = 1.f / total_probability;
            for (int i = 0; i < GRID_SIZE; ++i)
            {
                for (int j = 0; j < GRID_SIZE; ++j)
                {
                    pdf_grid[i][j] *= scale;
                }
            }
            total_probability = 1.f;
        }
    }

    void BehaviorPDF::add_weighted_sample(const math::vector3& pos, float weight)
    {
        // Convert world position to grid coordinates
        float dx = pos.x - origin.x;
        float dz = pos.z - origin.z;

        int grid_x = static_cast<int>((dx / cell_size) + GRID_SIZE / 2);
        int grid_z = static_cast<int>((dz / cell_size) + GRID_SIZE / 2);

        // Gaussian kernel (spread probability to nearby cells)
        // CONSISTENCY FIX: Sigma in WORLD UNITS, not grid cells
        // Old: sigma = 1.5 cells → varying spread (37-112 units depending on cell_size)
        // New: sigma = 50 world units → consistent spread regardless of prediction distance
        constexpr float SIGMA_WORLD_UNITS = 50.f;
        float sigma = SIGMA_WORLD_UNITS / cell_size;  // Convert to grid cells
        sigma = std::max(sigma, 0.5f);  // Minimum spread of 0.5 cells
        constexpr int kernel_radius = 2;

        for (int i = -kernel_radius; i <= kernel_radius; ++i)
        {
            for (int j = -kernel_radius; j <= kernel_radius; ++j)
            {
                int gx = grid_x + i;
                int gz = grid_z + j;

                if (gx >= 0 && gx < GRID_SIZE && gz >= 0 && gz < GRID_SIZE)
                {
                    float dist_sq = static_cast<float>(i * i + j * j);
                    float kernel_value = std::exp(-dist_sq / (2.f * sigma * sigma));
                    pdf_grid[gx][gz] += weight * kernel_value;
                }
            }
        }
    }

    // =========================================================================
    // TARGET BEHAVIOR TRACKER IMPLEMENTATION
    // =========================================================================

    TargetBehaviorTracker::TargetBehaviorTracker(game_object* target)
        : network_id_(target ? target->get_network_id() : 0), last_update_time_(0.f), last_aa_time_(0.f),
        cached_prediction_time_(-1.f), cached_move_speed_(-1.f), cached_timestamp_(-1.f)
    {
    }

    void TargetBehaviorTracker::update(game_object* current_target_ptr)
    {
        // Use fresh pointer passed from manager (prevents dangling pointer crashes)
        game_object* target_ = current_target_ptr;

        if (!target_ || !target_->is_valid())
            return;

        // CRITICAL: Validate clock_facade before accessing
        if (!g_sdk || !g_sdk->clock_facade)
            return;

        float current_time = g_sdk->clock_facade->get_game_time();

        // FOG EMERGENCE DETECTION: Reset history when target becomes visible again
        // This prevents stale velocity calculations from old positions
        bool currently_visible = target_->is_visible();
        if (currently_visible && !was_visible_last_update_)
        {
            // Target just emerged from fog - clear ALL stale data
            movement_history_.clear();
            last_update_time_ = 0.f;
            has_last_path_endpoint_ = false;  // Path data is stale too
        }
        was_visible_last_update_ = currently_visible;

        // Don't track movement while in fog (data would be stale/estimated)
        if (!currently_visible)
            return;

        // EVENT-DRIVEN SAMPLING: Detect path changes for zero-latency response
        // If target clicked a new destination, sample immediately (bypass 50ms timer)
        bool force_sample = false;
        auto current_path = target_->get_path();
        size_t current_path_size = current_path.size();

        if (!current_path.empty())
        {
            math::vector3 current_endpoint = current_path.back();
            bool path_changed = false;

            if (has_last_path_endpoint_)
            {
                // Check if path endpoint changed significantly (new click)
                float endpoint_delta = (current_endpoint - last_path_endpoint_).magnitude();
                if (endpoint_delta > 50.f)  // 50 units = meaningful new destination
                {
                    path_changed = true;
                    force_sample = true;  // New click detected - sample NOW
                }

                // Also check if path structure changed (waypoint added/removed)
                if (current_path_size != last_path_size_)
                {
                    path_changed = true;
                    force_sample = true;
                }
            }

            // Update path state tracking
            last_path_endpoint_ = current_endpoint;
            has_last_path_endpoint_ = true;

            // Track path update timestamp for staleness detection
            if (path_changed || last_path_update_time_ == 0.f)
            {
                last_path_update_time_ = current_time;
            }

            // Track path progress (which waypoint they're heading to)
            last_path_size_ = current_path_size;
            last_path_index_ = target_->get_current_path_index();
        }
        else
        {
            // Path is empty (stopped, arrived, or no path)
            // Reset path state if we had a recent path
            if (has_last_path_endpoint_ && (current_time - last_path_update_time_ < 0.1f))
            {
                // Path just disappeared - they arrived or stopped
                force_sample = true;
            }

            has_last_path_endpoint_ = false;
            last_path_size_ = 0;
            last_path_index_ = 0;
        }

        // ANIMATION STATE CHANGE DETECTION: Sample immediately when AA/Cast starts
        // Don't wait 50ms to detect lock - that's 50ms of wasted opportunity
        bool current_is_attacking = is_auto_attacking(target_);
        bool current_is_casting = is_casting_spell(target_);

        if (!movement_history_.empty())
        {
            const auto& last = movement_history_.back();
            if (current_is_attacking != last.is_auto_attacking ||
                current_is_casting != last.is_casting)
            {
                force_sample = true;  // Animation state changed - sample NOW
            }
        }

        // Sample at fixed rate OR on event (path change, animation state change)
        if (!force_sample && current_time - last_update_time_ < MOVEMENT_SAMPLE_RATE)
            return;

        // Create snapshot
        MovementSnapshot snapshot;
        // FIX: Use server position for accurate tracking (client position lags behind)
        snapshot.position = target_->get_server_position();
        snapshot.timestamp = current_time;

        snapshot.is_auto_attacking = is_auto_attacking(target_);
        snapshot.is_casting = is_casting_spell(target_);
        snapshot.is_dashing = target_->is_dashing();
        snapshot.is_cced = target_->has_buff_of_type(buff_type::stun) || target_->has_buff_of_type(buff_type::charm) ||
            target_->has_buff_of_type(buff_type::fear) || target_->has_buff_of_type(buff_type::snare) || target_->has_buff_of_type(buff_type::taunt) ||
            target_->has_buff_of_type(buff_type::suppression) || target_->has_buff_of_type(buff_type::knockup);

        // Safety: Prevent division by zero
        float max_hp = target_->get_max_hp();
        snapshot.hp_percent = (max_hp > 0.f) ? (target_->get_hp() / max_hp) * 100.f : 100.f;

        // Compute velocity if we have previous snapshot
        if (!movement_history_.empty())
        {
            // TELEPORT DETECTION: If position changed by more than 2000 units in one frame,
            // target teleported (recall, TP, etc.) - reset history to prevent garbage velocity
            float position_delta = (snapshot.position - movement_history_.back().position).magnitude();
            if (position_delta > 2000.f)
            {
                // Teleportation detected - clear history and start fresh
                movement_history_.clear();
                snapshot.velocity = math::vector3(0, 0, 0);
            }
            else
            {
                snapshot.velocity = compute_velocity(movement_history_.back(), snapshot);

                // Physics measurement code removed - calibration phase complete

                // ================================================================

                // VELOCITY SANITY CAPPING: Prevent extreme values
                // Max champion speed with all boosts is ~800 units/sec
                // Anything above 1000 is likely a calculation error or micro-teleport
                constexpr float MAX_SANE_VELOCITY = 1000.f;
                float vel_magnitude = snapshot.velocity.magnitude();
                if (vel_magnitude > MAX_SANE_VELOCITY)
                {
                    // Cap velocity magnitude while preserving direction
                    snapshot.velocity = (snapshot.velocity / vel_magnitude) * MAX_SANE_VELOCITY;
                }

                // SMART VELOCITY SMOOTHING: Handle special cases before blending
                float raw_speed = snapshot.velocity.magnitude();
                float smooth_speed = smoothed_velocity_.magnitude();

                // CASE 0: ANIMATION LOCKED - Anchor them in place
                // Force velocity to zero ONLY if they are actually stationary.
                // SYNDRA/ORIANNA/VIKTOR FIX: Some champs can cast while moving!
                // If we anchor a move-caster, we'll under-predict and shoot at their feet.
                //
                // ROBUST CHECK: Use path presence, not just is_moving()
                // Velocity can dip to zero for 1 frame at cast start, but path persists.
                bool has_move_path = false;
                auto path = target_->get_path();
                if (!path.empty())
                {
                    // Verify path goes beyond current position (not just a stop command)
                    if (path.back().distance(snapshot.position) > 10.f)
                        has_move_path = true;
                }

                bool is_locked = (snapshot.is_auto_attacking || snapshot.is_casting || snapshot.is_cced);
                bool is_move_casting = snapshot.is_casting && (target_->is_moving() || has_move_path);

                if (is_locked && !is_move_casting)
                {
                    smoothed_velocity_ = math::vector3(0, 0, 0);
                    snapshot.velocity = math::vector3(0, 0, 0);  // Override raw velocity too
                }
                // CASE 1: DASHING - Reset smoother to avoid polluting history
                // Dash velocity is forced movement, not player input
                else if (snapshot.is_dashing)
                {
                    smoothed_velocity_ = math::vector3(0, 0, 0);
                }
                // CASE 2: STOPPED - Use 2-frame buffer to prevent 1-frame stop exploits
                // High-skill players tap 'S' for 1 frame to fake jukes. Requiring 2 frames
                // (100ms) filters out single-frame stops while still detecting real stops quickly.
                else if (raw_speed < 10.f)
                {
                    zero_velocity_frames_++;
                    if (zero_velocity_frames_ >= 2)
                    {
                        // Confirmed stop (2+ frames) - snap to zero
                        smoothed_velocity_ = math::vector3(0, 0, 0);
                    }
                    // else: First frame of stop - keep decaying, don't snap instantly
                }
                // CASE 3: SHARP TURN - Snap to new direction (angle > 60 degrees)
                else if (raw_speed > 100.f && smooth_speed > 100.f)
                {
                    // Reset stop counter when moving
                    zero_velocity_frames_ = 0;

                    // Check angle between new raw velocity and old smoothed velocity
                    float dot = snapshot.velocity.normalized().dot(smoothed_velocity_.normalized());

                    if (dot < 0.5f)  // Angle > 60 degrees
                    {
                        // Sharp turn detected! Snap instantly to new direction
                        smoothed_velocity_ = snapshot.velocity;
                    }
                    else
                    {
                        // CASE 4: NORMAL MOVEMENT - Balanced alpha (IIR low-pass filter)
                        // 0.3f = too sluggish, 0.7f = too jittery on noisy data
                        // 0.5f = Nyquist sweet spot, responds in ~100ms while filtering noise
                        constexpr float SMOOTH_ALPHA = 0.5f;
                        smoothed_velocity_ = smoothed_velocity_ * (1.0f - SMOOTH_ALPHA) + snapshot.velocity * SMOOTH_ALPHA;
                    }
                }
                else
                {
                    // Low speed movement - reset stop counter
                    zero_velocity_frames_ = 0;
                    smoothed_velocity_ = snapshot.velocity;
                }
                snapshot.velocity = smoothed_velocity_;  // Commit to snapshot
            }
        }
        else
        {
            snapshot.velocity = math::vector3(0, 0, 0);
            smoothed_velocity_ = math::vector3(0, 0, 0);
        }

        // Detect auto-attack for post-AA movement analysis
        if (!movement_history_.empty() && snapshot.is_auto_attacking && !movement_history_.back().is_auto_attacking)
        {
            last_aa_time_ = current_time;
        }

        // Track post-AA movement delay
        if (!movement_history_.empty() && last_aa_time_ > 0.f && snapshot.velocity.magnitude() > 10.f &&
            movement_history_.back().velocity.magnitude() < 10.f)
        {
            float delay = current_time - last_aa_time_;
            if (delay < 1.0f) // Only track reasonable delays
            {
                post_aa_movement_delays_.push_back(delay);
                if (post_aa_movement_delays_.size() > 20)
                    post_aa_movement_delays_.erase(post_aa_movement_delays_.begin());
            }
        }

        // ADAPTIVE REACTION BUFFER: Track animation cancel delay
        // Measures how quickly this player starts moving after animation locks end
        // Scripters: ~0.005-0.015s, Average: ~0.025s, Lazy: ~0.05-0.1s
        bool currently_in_animation = snapshot.is_auto_attacking || snapshot.is_casting;

        // Detect animation END (was locked, now unlocked)
        // Only start tracking if they're NOT immediately CC'd (would corrupt measurement)
        if (was_in_animation_ && !currently_in_animation && !snapshot.is_cced)
        {
            // Animation just ended - record timestamp
            animation_end_time_ = current_time;
        }

        // CANCEL TRACKING if they get CC'd after animation end
        // CC duration would corrupt our cancel delay measurement
        if (animation_end_time_ > 0.f && snapshot.is_cced)
        {
            animation_end_time_ = 0.f;  // Abort - CC invalidates measurement
        }

        // Detect movement START after animation end
        if (animation_end_time_ > 0.f && snapshot.velocity.magnitude() > 50.f)
        {
            // Movement started after animation - measure the delay
            float cancel_delay = current_time - animation_end_time_;

            // Only count reasonable delays (0.001s to 0.5s)
            // Ignore very long delays (they probably got CC'd or changed their mind)
            if (cancel_delay > 0.001f && cancel_delay < 0.5f)
            {
                // Rolling average with decay (newer samples weighted more)
                if (cancel_delay_samples_ == 0)
                    measured_cancel_delay_ = cancel_delay;
                else
                    measured_cancel_delay_ = measured_cancel_delay_ * 0.7f + cancel_delay * 0.3f;
                cancel_delay_samples_++;
            }

            // Reset - we've captured this sample
            animation_end_time_ = 0.f;
        }

        // Also reset if they've been static for too long (> 0.5s after animation end)
        if (animation_end_time_ > 0.f && (current_time - animation_end_time_) > 0.5f)
        {
            animation_end_time_ = 0.f;  // Cancel tracking - they didn't move
        }

        was_in_animation_ = currently_in_animation;

        // Add to history
        movement_history_.push_back(snapshot);

        // Limit history size
        if (movement_history_.size() > MOVEMENT_HISTORY_SIZE)
            movement_history_.pop_front();

        last_update_time_ = current_time;

        // Analyze patterns periodically
        if (movement_history_.size() >= MIN_SAMPLES_FOR_BEHAVIOR &&
            static_cast<int>(movement_history_.size()) % 20 == 0)
        {
            analyze_patterns();
        }
    }

    void TargetBehaviorTracker::analyze_patterns()
    {
        update_dodge_pattern();
        detect_direction_changes();
    }

    math::vector3 TargetBehaviorTracker::compute_velocity(
        const MovementSnapshot& prev,
        const MovementSnapshot& curr) const
    {
        float dt = curr.timestamp - prev.timestamp;
        if (dt < EPSILON)
            return math::vector3{};

        return (curr.position - prev.position) / dt;
    }

    void TargetBehaviorTracker::update_dodge_pattern()
    {
        if (movement_history_.size() < 3)
            return;

        int left_count = 0, right_count = 0, forward_count = 0, backward_count = 0;
        int total_movements = 0;

        // Analyze movement directions relative to previous direction
        for (size_t i = 2; i < movement_history_.size(); ++i)
        {
            const auto& prev = movement_history_[i - 1];
            const auto& curr = movement_history_[i];

            if (prev.velocity.magnitude() < 10.f || curr.velocity.magnitude() < 10.f)
                continue;

            // Compute perpendicular and parallel components
            math::vector3 prev_dir = prev.velocity.normalized();
            math::vector3 curr_dir = curr.velocity.normalized();

            // Cross product to determine left/right (y component)
            float cross_y = prev_dir.x * curr_dir.z - prev_dir.z * curr_dir.x;

            // Dot product to determine forward/backward
            float dot = prev_dir.dot(curr_dir);

            // Original threshold for frequency stats
            if (cross_y > 0.1f) left_count++;
            else if (cross_y < -0.1f) right_count++;

            if (dot > 0.5f) forward_count++;
            else if (dot < -0.5f) backward_count++;

            total_movements++;
        }

        if (total_movements > 0)
        {
            float inv_total = 1.f / total_movements;
            dodge_pattern_.left_dodge_frequency = left_count * inv_total;
            dodge_pattern_.right_dodge_frequency = right_count * inv_total;
            dodge_pattern_.forward_frequency = forward_count * inv_total;
            dodge_pattern_.backward_frequency = backward_count * inv_total;

            // Linear continuation probability
            dodge_pattern_.linear_continuation_prob = forward_count * inv_total;
        }

        // Update reaction delay from post-AA movement data
        dodge_pattern_.reaction_delay = 200.f; // Default 200ms

        if (!post_aa_movement_delays_.empty())
        {
            float sum = 0.f;
            for (float delay : post_aa_movement_delays_)
                sum += delay;
            dodge_pattern_.reaction_delay = (sum / post_aa_movement_delays_.size()) * 1000.f;
        }

        // PATTERN REPETITION DETECTION
        // PATTERN EXPIRATION: Reset pattern if no movement updates for 3+ seconds
        if (!movement_history_.empty())
        {
            // CRITICAL: Validate clock_facade before accessing
            if (!g_sdk || !g_sdk->clock_facade)
                return;

            float current_time = g_sdk->clock_facade->get_game_time();
            float last_movement_time = movement_history_.back().timestamp;
            constexpr float PATTERN_EXPIRY_DURATION = 3.0f;  // 3 seconds of inactivity

            if (dodge_pattern_.has_pattern &&
                (current_time - last_movement_time > PATTERN_EXPIRY_DURATION))
            {
                // Pattern is stale - reset it
                dodge_pattern_.has_pattern = false;
                dodge_pattern_.pattern_confidence = 0.f;
                dodge_pattern_.predicted_next_direction = math::vector3{};
                dodge_pattern_.juke_sequence.clear();
            }
        }

        // EVENT-BASED JUKE SEQUENCE: Only record state transitions, not every sample
        // This allows 8 sequence slots to capture multiple full juke cycles (0.6-1.0s each)
        // instead of just 0.4s of raw samples
        std::vector<int> old_sequence = dodge_pattern_.juke_sequence;  // Save for comparison
        constexpr size_t MAX_SEQUENCE_LENGTH = 8;

        // Track juke events for magnitude measurement
        // We measure RETROSPECTIVELY: when a new juke starts, we calculate the previous juke's total magnitude
        size_t juke_start_idx = 0;
        math::vector3 juke_ref_dir{};  // Direction before juke started
        bool tracking_juke = false;

        // Process recent movement history to detect state transitions
        // Note: We check ALL recent history (not just 8 samples) to properly detect transitions
        size_t start_idx = movement_history_.size() > 40 ? movement_history_.size() - 40 : 1;

        for (size_t i = start_idx; i < movement_history_.size(); ++i)
        {
            const auto& prev = movement_history_[i - 1];
            const auto& curr = movement_history_[i];

            if (prev.velocity.magnitude() < 10.f || curr.velocity.magnitude() < 10.f)
                continue;

            math::vector3 prev_dir = prev.velocity.normalized();
            math::vector3 curr_dir = curr.velocity.normalized();

            // Cross product Y component determines left (-1) or right (1)
            // Higher threshold filters out pathfinding micro-corrections vs intentional dodges
            float cross_y = prev_dir.x * curr_dir.z - prev_dir.z * curr_dir.x;

            // Threshold: 0.25 ≈ sin(14.5°) filters noise, captures deliberate jukes
            // Old: 0.15 (~8.5°) was too sensitive, caught micro-adjustments as jukes
            constexpr float JUKE_THRESHOLD = 0.25f;

            int current_move = 0;
            if (cross_y > JUKE_THRESHOLD)
                current_move = -1;  // Left
            else if (cross_y < -JUKE_THRESHOLD)
                current_move = 1;   // Right
            // else current_move = 0 (Straight)

            // JUKE MAGNITUDE TRACKING: Measure full juke distance retrospectively
            // When direction changes again (or straightens), the previous juke just ended
            int last_move = dodge_pattern_.juke_sequence.empty() ? 0 : dodge_pattern_.juke_sequence.back();
            if (tracking_juke && current_move != last_move)
            {
                // Previous juke ended - calculate its TOTAL lateral displacement
                math::vector3 juke_perp(-juke_ref_dir.z, 0.f, juke_ref_dir.x);
                math::vector3 total_displacement = curr.position - movement_history_[juke_start_idx].position;
                float total_lateral = std::abs(total_displacement.x * juke_perp.x + total_displacement.z * juke_perp.z);

                // Store magnitude (cap history at 16 entries)
                constexpr size_t MAX_MAGNITUDE_HISTORY = 16;
                if (dodge_pattern_.juke_magnitudes.size() >= MAX_MAGNITUDE_HISTORY)
                    dodge_pattern_.juke_magnitudes.pop_front();
                dodge_pattern_.juke_magnitudes.push_back(total_lateral);

                // Update running average
                float sum = 0.f;
                for (float mag : dodge_pattern_.juke_magnitudes)
                    sum += mag;
                dodge_pattern_.average_juke_magnitude = sum / dodge_pattern_.juke_magnitudes.size();

                tracking_juke = false;
            }

            // Start tracking new juke
            if (current_move != 0 && !tracking_juke)
            {
                juke_start_idx = i - 1;  // Position before the direction change
                juke_ref_dir = prev_dir;  // Direction before juke
                tracking_juke = true;
            }
            else if (current_move == 0)
            {
                tracking_juke = false;
            }

            // EVENT-BASED RECORDING: Only add to sequence when state actually CHANGES
            // This prevents filling the sequence with repeated samples of the same move
            if (current_move != 0 && current_move != dodge_pattern_.last_recorded_move)
            {
                // Update n-gram transitions BEFORE adding new move
                if (!dodge_pattern_.juke_sequence.empty())
                {
                    int prev_move = dodge_pattern_.juke_sequence.back();
                    dodge_pattern_.ngram_transitions[prev_move][current_move]++;
                }

                // Record the state transition
                dodge_pattern_.juke_sequence.push_back(current_move);
                dodge_pattern_.last_recorded_move = current_move;

                // Maintain size limit (8 state transitions, not time-based)
                if (dodge_pattern_.juke_sequence.size() > MAX_SEQUENCE_LENGTH)
                {
                    dodge_pattern_.juke_sequence.erase(dodge_pattern_.juke_sequence.begin());
                }
            }
            else if (current_move == 0 && dodge_pattern_.last_recorded_move != 0)
            {
                // Transition back to straight - record it
                dodge_pattern_.last_recorded_move = 0;
            }
        }

        // BAYESIAN TRUST UPDATE: Check if our pattern prediction was correct
        // Compare old sequence to new - if new juke appeared, check if it matches prediction
        if (dodge_pattern_.awaiting_juke_result &&
            dodge_pattern_.juke_sequence.size() > old_sequence.size() &&
            !dodge_pattern_.juke_sequence.empty())
        {
            int actual_juke = dodge_pattern_.juke_sequence.back();

            // Only update if they actually juked (not straight)
            if (actual_juke != 0)
            {
                if (actual_juke == dodge_pattern_.last_predicted_juke)
                {
                    dodge_pattern_.pattern_trust.observe_correct();
                }
                else
                {
                    dodge_pattern_.pattern_trust.observe_incorrect();
                }
            }

            dodge_pattern_.awaiting_juke_result = false;
        }

        // Detect alternating pattern (L-R-L-R or R-L-R-L)
        // EARNED CONFIDENCE: Require more evidence before trusting patterns
        // Don't give high confidence after just L-R-L-R - need sustained pattern
        // Increased thresholds to match more conservative overall approach (40 sample minimum)
        if (dodge_pattern_.juke_sequence.size() >= 8)  // Need 8+ jukes for reliability
        {
            bool is_alternating = true;
            int alternation_count = 0;

            for (size_t i = 2; i < dodge_pattern_.juke_sequence.size(); ++i)
            {
                int prev_juke = dodge_pattern_.juke_sequence[i - 2];
                int curr_juke = dodge_pattern_.juke_sequence[i];

                // Skip straights
                if (prev_juke == 0 || curr_juke == 0)
                    continue;

                // Check if opposite direction
                if (prev_juke == -curr_juke)
                    alternation_count++;
                else
                {
                    is_alternating = false;
                    break;  // No point continuing - pattern is broken
                }
            }

            // Require 5+ alternations for pattern (more conservative)
            // Confidence scales: 5 alt = 0.55, 6 = 0.625, 7 = 0.70, 8+ = 0.75 max
            if (is_alternating && alternation_count >= 5)
            {
                // Alternating pattern detected with sufficient evidence
                dodge_pattern_.has_pattern = true;
                float base_confidence = std::min(0.75f, 0.4f + alternation_count * 0.075f);
                // Scale by Bayesian trust - if our predictions have been wrong, reduce confidence
                dodge_pattern_.pattern_confidence = base_confidence * dodge_pattern_.pattern_trust.get_trust();

                // Predict next juke: opposite of last
                int last_juke = dodge_pattern_.juke_sequence.back();
                if (last_juke != 0 && !movement_history_.empty())
                {
                    const auto& latest = movement_history_.back();
                    // CRASH FIX: Check velocity magnitude before normalizing
                    float vel_mag = latest.velocity.magnitude();
                    if (vel_mag >= 0.001f)
                    {
                        math::vector3 vel_dir = latest.velocity / vel_mag;
                        // Perpendicular: 90° rotation in XZ plane
                        math::vector3 perpendicular(-vel_dir.z, 0.f, vel_dir.x);
                        // FIX: Direction = perpendicular * (-juke_value)
                        // predicted_juke = -last_juke, so direction = perpendicular * last_juke
                        dodge_pattern_.predicted_next_direction = perpendicular * static_cast<float>(last_juke);

                        // Track prediction for Bayesian trust update
                        dodge_pattern_.last_predicted_juke = -last_juke;  // Opposite of last
                        dodge_pattern_.awaiting_juke_result = true;
                    }
                }
            }
            // Detect repeating sequence (e.g., L-L-R-L-L-R)
            // Require 10+ jukes for repeating pattern (more complex, needs more evidence)
            else if (dodge_pattern_.juke_sequence.size() >= 10)
            {
                // Check if first half matches second half
                size_t half = dodge_pattern_.juke_sequence.size() / 2;
                bool is_repeating = true;

                for (size_t i = 0; i < half; ++i)
                {
                    if (dodge_pattern_.juke_sequence[i] != dodge_pattern_.juke_sequence[i + half])
                    {
                        is_repeating = false;
                        break;
                    }
                }

                if (is_repeating)
                {
                    // Repeating sequence detected with sufficient evidence
                    dodge_pattern_.has_pattern = true;
                    // Scale by sequence length: 8 = 0.6, 10 = 0.65, 12+ = 0.7 max
                    float base_confidence = std::min(0.7f, 0.5f + dodge_pattern_.juke_sequence.size() * 0.0125f);
                    // Scale by Bayesian trust
                    dodge_pattern_.pattern_confidence = base_confidence * dodge_pattern_.pattern_trust.get_trust();
                    if (g_sdk && g_sdk->clock_facade)
                        dodge_pattern_.last_pattern_update_time = g_sdk->clock_facade->get_game_time();

                    // Predict next: continues the sequence
                    // CRASH FIX: Check half != 0 before modulo
                    if (half > 0)
                    {
                        int next_in_sequence = dodge_pattern_.juke_sequence[dodge_pattern_.juke_sequence.size() % half];
                        if (next_in_sequence != 0 && !movement_history_.empty())
                        {
                            const auto& latest = movement_history_.back();
                            // CRASH FIX: Check velocity magnitude before normalizing
                            float vel_mag = latest.velocity.magnitude();
                            if (vel_mag >= 0.001f)
                            {
                                math::vector3 vel_dir = latest.velocity / vel_mag;
                                math::vector3 perpendicular(-vel_dir.z, 0.f, vel_dir.x);
                                // FIX: Direction = perpendicular * (-juke_value)
                                dodge_pattern_.predicted_next_direction = perpendicular * static_cast<float>(-next_in_sequence);

                                // Track prediction for Bayesian trust update
                                dodge_pattern_.last_predicted_juke = next_in_sequence;
                                dodge_pattern_.awaiting_juke_result = true;
                            }
                        }
                    }
                }
            }
        }

        // No pattern detected
        if (!dodge_pattern_.has_pattern)
        {
            dodge_pattern_.pattern_confidence = 0.f;
            dodge_pattern_.predicted_next_direction = math::vector3{};
        }
    }

    void TargetBehaviorTracker::detect_direction_changes()
    {
        direction_change_times_.clear();
        direction_change_angles_.clear();

        if (movement_history_.size() < 3)
        {
            // Not enough data - reset average
            average_turn_angle_ = 0.f;
            return;
        }

        // CONTEXT-AWARE JUKE DETECTION
        // Only count INTENTIONAL dodging, not CSing/kiting/orb-walking
        for (size_t i = 2; i < movement_history_.size(); ++i)
        {
            const auto& prev = movement_history_[i - 2];
            const auto& mid = movement_history_[i - 1];
            const auto& curr = movement_history_[i];

            if (prev.velocity.magnitude() < 10.f || curr.velocity.magnitude() < 10.f)
                continue;

            // Detect significant direction change
            math::vector3 prev_dir = prev.velocity.normalized();
            math::vector3 curr_dir = curr.velocity.normalized();

            float angle = std::acos(std::clamp(prev_dir.dot(curr_dir), -1.f, 1.f));
            float angle_degrees = angle * (180.f / PI);

            // CONTEXT FILTERING: Distinguish jukes from normal gameplay
            bool is_legitimate_juke = true;

            // FILTER 1: Ignore orb-walking/CSing movement (during auto-attacks)
            // If they're auto-attacking, direction changes are probably kiting, not dodging
            if (curr.is_auto_attacking || mid.is_auto_attacking)
            {
                // Only count as juke if it's a LARGE lateral movement (>45° AND high speed)
                // Small corrections during AAs are normal kiting
                if (angle_degrees < 60.f || curr.velocity.magnitude() < 200.f)
                {
                    is_legitimate_juke = false;
                }
            }

            // FILTER 2: Magnitude threshold - small corrections aren't jukes
            // Calculate actual displacement to distinguish small corrections from real jukes
            float time_delta = curr.timestamp - mid.timestamp;
            if (time_delta > 0.001f)  // Avoid division by zero
            {
                math::vector3 displacement = curr.position - mid.position;
                float lateral_displacement = displacement.magnitude();

                // If lateral movement is < 75 units, it's probably a minor correction
                // Real jukes involve significant lateral displacement (100+ units)
                if (lateral_displacement < 75.f)
                {
                    is_legitimate_juke = false;
                }
            }

            // FILTER 3: Forward path corrections vs lateral dodging
            // Calculate if movement is perpendicular (juke) vs along path (correction)
            if (angle_degrees > 30.f && angle_degrees < 150.f)  // Not a reversal or tiny turn
            {
                // Use cross product to check if it's lateral (left/right) vs forward/back
                math::vector3 cross = prev_dir.cross(curr_dir);
                float lateral_component = std::abs(cross.y);  // Y component = lateral magnitude

                // If lateral component is small, it's a forward correction, not a juke
                if (lateral_component < 0.4f)  // < ~24° lateral angle
                {
                    is_legitimate_juke = false;
                }
            }

            // Track ALL turn angles for average (for smooth runners vs dancers)
            // But weight legitimate jukes more heavily
            float angle_weight = is_legitimate_juke ? 1.0f : 0.3f;
            recent_turn_angles_.push_back(angle_degrees * angle_weight);
            if (recent_turn_angles_.size() > 8)  // Keep last 8 angles
                recent_turn_angles_.pop_front();

            // Only track LEGITIMATE jukes in direction change history
            if (angle > 0.5f && is_legitimate_juke) // ~30 degrees + context filters
            {
                direction_change_times_.push_back(curr.timestamp);
                direction_change_angles_.push_back(angle);
            }
        }

        // Calculate running average turn angle (lightweight juke detection)
        if (!recent_turn_angles_.empty())
        {
            float sum = 0.f;
            for (float angle : recent_turn_angles_)
                sum += angle;
            average_turn_angle_ = sum / recent_turn_angles_.size();
        }
        else
        {
            average_turn_angle_ = 0.f;
        }

        // HARD JUKE DETECTION: Check for sharp turns (>45°) in recent history
        // Used for confidence fast-track - bypass sample count penalty
        // NOW WITH CONTEXT FILTERING
        recent_hard_juke_ = false;
        if (movement_history_.size() >= 3)
        {
            size_t check_count = std::min(movement_history_.size(), size_t(5));
            for (size_t i = 1; i < check_count; ++i)
            {
                size_t idx = movement_history_.size() - i;
                const auto& curr = movement_history_[idx];
                const auto& prev = movement_history_[idx - 1];

                if (curr.velocity.magnitude() > 100.f && prev.velocity.magnitude() > 100.f)
                {
                    math::vector3 v1 = curr.velocity.normalized();
                    math::vector3 v2 = prev.velocity.normalized();
                    float dot = v1.dot(v2);

                    // dot < 0.707 means angle > 45 degrees (sharp turn)
                    if (dot < 0.707f)
                    {
                        // CONTEXT CHECK: Was this during an auto-attack?
                        // If yes, only count if it's a VERY sharp turn (>60°)
                        if (curr.is_auto_attacking)
                        {
                            if (dot < 0.5f)  // >60° turn during AA = legitimate dodge
                            {
                                recent_hard_juke_ = true;
                                break;
                            }
                        }
                        else
                        {
                            // Not during AA - this is a legitimate juke
                            recent_hard_juke_ = true;
                            break;
                        }
                    }
                }
            }
        }

        // Compute juke interval statistics
        if (direction_change_times_.size() >= 2)
        {
            std::vector<float> intervals;
            for (size_t i = 1; i < direction_change_times_.size(); ++i)
            {
                intervals.push_back(direction_change_times_[i] - direction_change_times_[i - 1]);
            }

            // Mean
            float sum = 0.f;
            for (float interval : intervals)
                sum += interval;
            dodge_pattern_.juke_interval_mean = sum / intervals.size();

            // Variance
            float variance_sum = 0.f;
            for (float interval : intervals)
            {
                float diff = interval - dodge_pattern_.juke_interval_mean;
                variance_sum += diff * diff;
            }
            dodge_pattern_.juke_interval_variance = variance_sum / intervals.size();
        }
    }

    bool TargetBehaviorTracker::is_animation_locked() const
    {
        if (movement_history_.empty())
            return false;

        const auto& latest = movement_history_.back();
        return latest.is_auto_attacking || latest.is_casting || latest.is_cced;
    }

    math::vector3 TargetBehaviorTracker::get_current_velocity() const
    {
        if (movement_history_.empty())
            return math::vector3{};

        // Return smoothed velocity to reduce jitter from spam clicking
        return smoothed_velocity_;
    }

    BehaviorPDF TargetBehaviorTracker::build_behavior_pdf(float prediction_time, float move_speed) const
    {
        // PDF caching: Reuse cached PDF if same frame and similar parameters
        // This avoids rebuilding for Q/W/E/R predictions on the same target in one frame
        // CRASH FIX: Check g_sdk and clock_facade before accessing
        if (!g_sdk || !g_sdk->clock_facade)
        {
            BehaviorPDF empty_pdf;
            return empty_pdf;
        }
        float current_time = g_sdk->clock_facade->get_game_time();
        constexpr float TIME_TOLERANCE = 0.05f;  // 50ms tolerance for prediction_time similarity
        constexpr float SPEED_TOLERANCE = 20.f;  // 20 units/s tolerance for move_speed

        bool same_frame = std::abs(current_time - cached_timestamp_) < EPSILON;
        bool similar_pred_time = std::abs(prediction_time - cached_prediction_time_) < TIME_TOLERANCE;
        bool similar_move_speed = std::abs(move_speed - cached_move_speed_) < SPEED_TOLERANCE;

        if (same_frame && similar_pred_time && similar_move_speed && cached_pdf_.total_probability > EPSILON)
        {
            // Cache hit - return cached PDF
            return cached_pdf_;
        }

        BehaviorPDF pdf;

        if (movement_history_.empty())
            return pdf;

        const auto& latest = movement_history_.back();

        // DYNAMIC CELL SIZE: Ensure grid covers maximum distance target can move
        // Grid radius = (GRID_SIZE / 2) * cell_size
        // Target max movement = move_speed * prediction_time
        // Add 20% margin for dodge patterns
        // Use move_speed stat instead of velocity.magnitude() (which can be 0 if target is mid-cast)
        float max_move_distance = move_speed * prediction_time * 1.2f;
        float required_grid_radius = std::max(400.f, max_move_distance);  // Minimum 400 units
        pdf.cell_size = (required_grid_radius * 2.f) / BehaviorPDF::GRID_SIZE;  // Total coverage / grid_size
        // CRASH PROTECTION: Ensure cell_size is never zero
        if (pdf.cell_size < EPSILON)
            pdf.cell_size = 25.f;  // Safe default

        // ADAPTIVE DECAY RATE: Adjust based on target mobility
        float decay_rate = get_adaptive_decay_rate(latest.velocity.magnitude());

        // If animation locked (AA, casting, or CC'd), predict stationary at current position
        if (latest.is_auto_attacking || latest.is_casting || latest.is_cced)
        {
            pdf.origin = latest.position;
            pdf.add_weighted_sample(latest.position, 1.0f);
            pdf.normalize();
            return pdf;
        }

        // TIME-BASED HISTORY WINDOW (APM Trap Fix)
        // Old: "Use last N samples" - Failed against spam-clickers (15 samples = 0.15s if 10 clicks/sec)
        // New: "Use last X seconds" - Consistent regardless of click rate
        //
        // Juking/High Variance: 0.75s (short memory, recent behavior matters)
        // Stable/Low Variance:  2.5s (long memory, capture full trajectory)

        float velocity_variance = 0.f;
        if (movement_history_.size() >= 10)
        {
            // Calculate velocity variance from last 10 samples
            float mean_vx = 0.f, mean_vz = 0.f;
            int variance_samples = std::min(static_cast<int>(movement_history_.size()), 10);

            for (int i = 0; i < variance_samples; ++i)
            {
                const auto& snap = movement_history_[movement_history_.size() - 1 - i];
                mean_vx += snap.velocity.x;
                mean_vz += snap.velocity.z;
            }
            mean_vx /= variance_samples;
            mean_vz /= variance_samples;

            for (int i = 0; i < variance_samples; ++i)
            {
                const auto& snap = movement_history_[movement_history_.size() - 1 - i];
                float dx = snap.velocity.x - mean_vx;
                float dz = snap.velocity.z - mean_vz;
                velocity_variance += dx * dx + dz * dz;
            }
            velocity_variance /= variance_samples;
        }

        // Normalize variance by move_speed squared (scale-independent)
        float normalized_variance = (move_speed > 10.f) ? velocity_variance / (move_speed * move_speed) : 0.f;

        // INSTANT JUKE DETECTION (PDF Staleness Fix)
        // Variance is a lagging indicator - if target ran straight for 5s then hard-juked,
        // low variance average would force long window for several frames, drowning out the juke.
        // Fix: Check immediate history for sharp turns (>45°) and force short window instantly.
        bool recent_hard_juke = false;
        if (movement_history_.size() >= 3)
        {
            // Check last few samples for sharp direction changes
            size_t check_count = std::min(movement_history_.size(), size_t(5));
            for (size_t i = 1; i < check_count; ++i)
            {
                size_t idx = movement_history_.size() - i;
                const auto& curr = movement_history_[idx];
                const auto& prev = movement_history_[idx - 1];

                // Only check if both samples have meaningful velocity
                if (curr.velocity.magnitude() > 100.f && prev.velocity.magnitude() > 100.f)
                {
                    float curr_mag = curr.velocity.magnitude();
                    float prev_mag = prev.velocity.magnitude();
                    math::vector3 v1 = curr.velocity / curr_mag;
                    math::vector3 v2 = prev.velocity / prev_mag;
                    float dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;

                    // dot < 0.707 means angle > 45 degrees
                    if (dot < 0.707f)
                    {
                        recent_hard_juke = true;
                        break;
                    }
                }
            }
        }

        // Determine time-based history window
        // High variance OR erratic juke timing OR recent hard juke = short memory
        // Low variance = long memory for trajectory stability
        float history_duration;
        if (recent_hard_juke || normalized_variance > 0.3f || dodge_pattern_.juke_interval_variance > 0.05f)
            history_duration = 0.75f;   // Juking: 0.75s
        else if (normalized_variance < 0.1f)
            history_duration = 2.5f;    // Straight runner: 2.5s
        else
            history_duration = 1.5f;    // Mixed: 1.5s

        float min_timestamp = current_time - history_duration;

        // First pass: Compute weighted average of predicted positions
        // This centers the grid where samples will actually fall, not just where latest velocity predicts
        math::vector3 predicted_center{};
        float total_weight = 0.f;
        float current_weight = 1.0f;  // decay_rate^0 = 1.0

        for (int i = static_cast<int>(movement_history_.size()) - 1; i >= 0; --i)
        {
            const auto& snapshot = movement_history_[i];

            // TIME-BASED CUTOFF: Stop if sample is too old
            if (snapshot.timestamp < min_timestamp)
                break;

            // Exponential decay weighting (recent data more important)
            // Use multiplicative accumulation instead of std::pow for O(1) per iteration
            float weight = current_weight;
            current_weight *= decay_rate;  // Prepare next weight

            // FIX: Scale historical velocity to current move_speed
            // Direction from history, magnitude from current move_speed stat
            // This handles both slows (scale down) and speed buffs (scale up)
            math::vector3 adjusted_velocity = snapshot.velocity;
            float snapshot_speed = snapshot.velocity.magnitude();
            if (snapshot_speed > 10.f && move_speed > 1.f)
            {
                // Scale velocity magnitude to current move_speed
                adjusted_velocity = snapshot.velocity.normalized() * move_speed;
            }
            // If snapshot_speed <= 10, target was stationary - keep as-is

            // Predict position from this snapshot with adjusted velocity
            math::vector3 predicted_pos = snapshot.position + adjusted_velocity * prediction_time;

            // Accumulate for weighted average
            predicted_center = predicted_center + predicted_pos * weight;
            total_weight += weight;
        }

        // Center grid at weighted average of all predictions
        // This handles direction changes gracefully - grid is centered where samples cluster
        if (total_weight > EPSILON)
        {
            predicted_center = predicted_center / total_weight;
        }
        else
        {
            // Fallback to simple linear prediction
            predicted_center = latest.position + latest.velocity * prediction_time;
        }

        // FIX: Grid snapping to stabilize PDF across frames
        // Snap origin to cell_size multiples to prevent jitter from micro-position changes
        float snap_x = std::round(predicted_center.x / pdf.cell_size) * pdf.cell_size;
        float snap_z = std::round(predicted_center.z / pdf.cell_size) * pdf.cell_size;
        pdf.origin = math::vector3(snap_x, predicted_center.y, snap_z);

        // Second pass: Add samples to PDF (now properly centered)
        total_weight = 0.f;
        current_weight = 1.0f;  // Reset for second pass

        for (int i = static_cast<int>(movement_history_.size()) - 1; i >= 0; --i)
        {
            const auto& snapshot = movement_history_[i];

            // TIME-BASED CUTOFF: Stop if sample is too old
            if (snapshot.timestamp < min_timestamp)
                break;

            // Exponential decay weighting (recent data more important)
            float weight = current_weight;
            current_weight *= decay_rate;

            // FIX: Scale historical velocity to current move_speed (same as first pass)
            math::vector3 adjusted_velocity = snapshot.velocity;
            float snapshot_speed = snapshot.velocity.magnitude();
            if (snapshot_speed > 10.f && move_speed > 1.f)
            {
                adjusted_velocity = snapshot.velocity.normalized() * move_speed;
            }

            // Predict position from this snapshot with adjusted velocity
            math::vector3 predicted_pos = snapshot.position + adjusted_velocity * prediction_time;

            // Add to PDF with weight
            pdf.add_weighted_sample(predicted_pos, weight);
            total_weight += weight;
        }

        // Apply dodge pattern bias ONLY if target has had time to react
        // Reaction delay gating: Skip lateral dodge samples if prediction_time < reaction_delay
        // Target cannot have juked yet if they haven't had time to react
        float reaction_delay_seconds = dodge_pattern_.reaction_delay / 1000.f;  // Convert ms to seconds
        bool can_react = prediction_time >= reaction_delay_seconds;

        if (latest.velocity.magnitude() > 10.f && can_react)
        {
            math::vector3 velocity_dir = latest.velocity.normalized();
            // Perpendicular in XZ plane (2D movement) - 90° rotation
            math::vector3 perpendicular(-velocity_dir.z, 0.f, velocity_dir.x);

            // FIX: Use current move_speed for dodge calculations
            // Direction from velocity, magnitude from move_speed stat

            // USE OBSERVED JUKE MAGNITUDE instead of inferring from angles
            // This tells us exactly how far they travel laterally during jukes
            float total_distance = move_speed * prediction_time;
            float observed_lateral = dodge_pattern_.get_juke_magnitude(move_speed);

            // Cap lateral at 90% of total distance to maintain forward momentum
            // (They can't juke further than they can travel)
            observed_lateral = std::min(observed_lateral, total_distance * 0.9f);

            // Pythagorean: forward² + lateral² = total² (distance conservation)
            float forward_distance = std::sqrt(total_distance * total_distance - observed_lateral * observed_lateral);
            math::vector3 forward = velocity_dir * forward_distance;
            math::vector3 side = perpendicular * observed_lateral;

            // Juke cadence weighting: Weight based on WHERE we are in their juke cycle
            // Key insight: We need time since LAST juke to predict NEXT juke
            float juke_cadence_weight = 1.0f;  // Default
            if (dodge_pattern_.juke_interval_variance > EPSILON && !direction_change_times_.empty())
            {
                // Get time since last direction change (juke)
                float current_time = 0.f;
                if (g_sdk && g_sdk->clock_facade)
                    current_time = g_sdk->clock_facade->get_game_time();

                float last_juke_time = direction_change_times_.back();
                float time_since_last_juke = current_time - last_juke_time;

                // Predict when next juke will happen based on their rhythm
                // If they juke every 0.5s on average and last juked 0.2s ago, next is in 0.3s
                float time_until_next_juke = dodge_pattern_.juke_interval_mean - time_since_last_juke;

                // If negative, they're "overdue" for a juke - wrap to next cycle
                // Guard against infinite loop if juke_interval_mean is invalid
                int wrap_count = 0;
                while (time_until_next_juke < 0.f && wrap_count < 100)
                {
                    if (dodge_pattern_.juke_interval_mean <= EPSILON)
                        break;
                    time_until_next_juke += dodge_pattern_.juke_interval_mean;
                    wrap_count++;
                }

                // Gaussian weight: high when spell arrives near expected juke time
                // Defensive: max(0, variance) handles floating point precision edge cases
                float sigma = std::sqrt(std::max(0.f, dodge_pattern_.juke_interval_variance));
                float time_diff = prediction_time - time_until_next_juke;
                juke_cadence_weight = std::exp(-0.5f * (time_diff * time_diff) / (sigma * sigma));
                juke_cadence_weight = std::clamp(juke_cadence_weight, 0.3f, 1.0f);
            }
            else if (dodge_pattern_.juke_interval_variance > EPSILON)
            {
                // Fallback: no juke history yet, use old method
                // Defensive: max(0, variance) handles floating point precision edge cases
                float sigma = std::sqrt(std::max(0.f, dodge_pattern_.juke_interval_variance));
                float time_diff = prediction_time - dodge_pattern_.juke_interval_mean;
                juke_cadence_weight = std::exp(-0.5f * (time_diff * time_diff) / (sigma * sigma));
                juke_cadence_weight = std::clamp(juke_cadence_weight, 0.3f, 1.0f);
            }

            // N-GRAM ENHANCED DODGE PREDICTION
            // Blend overall frequency with N-Gram probability for smarter weighting
            // N-Gram captures "given they just went Right, what's next?" vs overall "they go Left 60% of time"
            float ngram_left = dodge_pattern_.get_ngram_probability(-1);
            float ngram_right = dodge_pattern_.get_ngram_probability(1);

            // Blend: 40% N-Gram + 60% overall frequency
            // Don't overfit to last few transitions - overall frequency is more stable
            float left_weight = 0.4f * ngram_left + 0.6f * dodge_pattern_.left_dodge_frequency;
            float right_weight = 0.4f * ngram_right + 0.6f * dodge_pattern_.right_dodge_frequency;

            if (left_weight > 0.2f)
            {
                math::vector3 left_pos = latest.position + forward + side;
                pdf.add_weighted_sample(left_pos, left_weight * 0.5f * juke_cadence_weight);
            }

            if (right_weight > 0.2f)
            {
                math::vector3 right_pos = latest.position + forward - side;
                pdf.add_weighted_sample(right_pos, right_weight * 0.5f * juke_cadence_weight);
            }

            // PATTERN-BASED PREDICTION
            // If we detected a repeating pattern, weight the predicted juke position
            // But also account for the chance they DON'T juke (break pattern)
            if (dodge_pattern_.has_pattern && dodge_pattern_.pattern_confidence > 0.6f)
            {
                // Predict position based on detected pattern
                // Use same distance conservation as above
                math::vector3 pattern_predicted_pos = latest.position +
                    forward +  // Already scaled by forward_distance
                    dodge_pattern_.predicted_next_direction * observed_lateral;

                // Pattern weight: commit to predicted juke but scale by timing
                // If prediction_time matches their juke rhythm, we're more confident
                float pattern_weight = dodge_pattern_.pattern_confidence * 1.8f * juke_cadence_weight;
                pdf.add_weighted_sample(pattern_predicted_pos, pattern_weight);

                // Also add "no juke" position - they might break the pattern
                // Always significant weight - even established patterns get broken
                // Weight higher when timing is off-rhythm (low juke_cadence_weight)
                math::vector3 no_juke_pos = latest.position + velocity_dir * total_distance;
                float no_juke_weight = std::max(0.5f, (1.0f - dodge_pattern_.pattern_confidence) + (1.0f - juke_cadence_weight) * 0.5f);
                pdf.add_weighted_sample(no_juke_pos, no_juke_weight);
            }
        }

        pdf.normalize();

        // Update cache for next call
        cached_pdf_ = pdf;
        cached_prediction_time_ = prediction_time;
        cached_move_speed_ = move_speed;
        cached_timestamp_ = current_time;

        return pdf;
    }

    OpportunityWindow& TargetBehaviorTracker::get_opportunity_window(int spell_slot) const
    {
        // Create if doesn't exist (mutable map allows modification in const method)
        if (opportunity_windows_.find(spell_slot) == opportunity_windows_.end())
        {
            opportunity_windows_[spell_slot] = OpportunityWindow();
            // Safety: Only set window start time if SDK is valid
            if (g_sdk && g_sdk->clock_facade)
            {
                opportunity_windows_[spell_slot].window_start_time = g_sdk->clock_facade->get_game_time();
            }
        }
        return opportunity_windows_[spell_slot];
    }

    // =========================================================================
    // OPPORTUNITY WINDOW IMPLEMENTATION
    // =========================================================================

    void OpportunityWindow::update(float current_time, float hit_chance)
    {
        // Add current sample to history
        history.push_back({ current_time, hit_chance });

        // MEMORY SAFETY: Cap deque size to prevent unbounded growth during lag spikes
        // 240Hz × 10s lag = 2400 entries without cap - limit to 200
        constexpr size_t MAX_HISTORY_SIZE = 200;
        while (history.size() > MAX_HISTORY_SIZE)
        {
            history.pop_front();
        }

        // Remove samples older than 3 seconds
        constexpr float WINDOW_DURATION = 3.0f;
        while (!history.empty() && current_time - history.front().first > WINDOW_DURATION)
        {
            history.pop_front();
        }

        // Update peak if this is better
        if (hit_chance > peak_hit_chance)
        {
            peak_hit_chance = hit_chance;
            peak_timestamp = current_time;
        }

        // Reset peak if too old (more than 2 seconds ago)
        if (current_time - peak_timestamp > 2.0f)
        {
            // Recalculate peak from current window
            peak_hit_chance = 0.f;
            for (const auto& sample : history)
            {
                if (sample.second > peak_hit_chance)
                {
                    peak_hit_chance = sample.second;
                    peak_timestamp = sample.first;
                }
            }
        }
    }

    bool OpportunityWindow::is_peak_opportunity(float current_time, float hit_chance, float adaptive_threshold, float elapsed_time, float patience_window) const
    {
        // SAFEGUARD 1: Adaptive Patience Window
        // Don't flag peaks too early - wait patience_window seconds
        // Patience adapts to spell cooldown: short CD (3s) = 1.5s, long CD (90s+) = 3.0s
        // This prevents casting on first tiny peak when better opportunity might come
        if (elapsed_time < patience_window)
            return false;

        if (history.size() < 6)  // Need at least 6 samples to detect trend
            return false;

        // SAFEGUARD 2: Minimum Quality
        // Peak must meet the full adaptive_threshold to be worth taking
        // No compromise - if threshold is 65%, need at least 65%
        if (hit_chance < adaptive_threshold)
            return false;

        // Check if this is a local maximum
        // Compare current hit_chance to recent average (last 1 second)
        float recent_sum = 0.f;
        int recent_count = 0;
        for (auto it = history.rbegin(); it != history.rend() && current_time - it->first < 1.0f; ++it)
        {
            recent_sum += it->second;
            recent_count++;
        }

        if (recent_count < 3)
            return false;

        float recent_avg = recent_sum / recent_count;

        // Current hit_chance must be above recent average (we're at a peak)
        if (hit_chance < recent_avg * 1.05f)  // 5% margin
            return false;

        // SAFEGUARD 3: Sustained Decline
        // Check if declining for 4+ consecutive samples (not just 2-3)
        // This prevents casting on random noise/blips
        if (history.size() >= 5)
        {
            float sample_5_ago = history[history.size() - 5].second;
            float sample_4_ago = history[history.size() - 4].second;
            float sample_3_ago = history[history.size() - 3].second;
            float sample_2_ago = history[history.size() - 2].second;
            float sample_1_ago = history[history.size() - 1].second;

            // SUSTAINED declining trend: 4+ consecutive drops
            bool is_sustained_decline = (sample_1_ago < sample_2_ago) &&
                (sample_2_ago < sample_3_ago) &&
                (sample_3_ago < sample_4_ago) &&
                (sample_4_ago < sample_5_ago);

            if (is_sustained_decline)
                return true;  // Safe to cast - sustained decline confirmed!
        }

        return false;
    }

    float OpportunityWindow::get_adaptive_threshold(float base_threshold, float elapsed_time) const
    {
        // Adaptive threshold decay: lower standards over time (CONSERVATIVE)
        // 0-5s: Full threshold (no decay) - be patient
        // 5-12s: Linear decay to 85% of threshold
        // 12s+: Minimum 85% of original threshold

        if (elapsed_time < 5.0f)
            return base_threshold;  // No decay yet - be patient

        if (elapsed_time < 12.0f)
        {
            // Linear interpolation: 100% → 85% over 7 seconds
            float decay_factor = 1.0f - ((elapsed_time - 5.0f) / 7.0f) * 0.15f;
            return base_threshold * decay_factor;
        }

        // Minimum: 85% of original (don't go too low)
        return base_threshold * 0.85f;
    }

    void HybridFusionEngine::update_opportunity_signals(
        HybridPredictionResult& result,
        game_object* source,
        game_object* target,
        const pred_sdk::spell_data& spell,
        TargetBehaviorTracker& tracker)
    {
        // Safety: Validate SDK before accessing
        if (!g_sdk || !g_sdk->clock_facade)
        {
            // Fallback: No opportunistic signals
            result.is_peak_opportunity = false;
            result.opportunity_score = 0.f;
            result.adaptive_threshold = 0.65f;  // Default high threshold
            return;
        }

        float current_time = g_sdk->clock_facade->get_game_time();
        int spell_slot = spell.spell_slot;

        // SMART TRIGGER: Animation Lock Check
        // Only bypass patience if the lock lasts long enough for the spell to land
        // Otherwise, the target will be free to dodge by the time our spell arrives
        if (tracker.is_animation_locked() && source && target)
        {
            // Calculate how long until the spell lands
            float arrival_time = PhysicsPredictor::compute_arrival_time(
                source->get_position(),
                target->get_position(),
                spell.projectile_speed,
                spell.delay
            );

            // Get how long the target will remain locked
            float remaining_lock = get_remaining_lock_time(target);

            // Only bypass patience if lock lasts until (or past) spell arrival
            // This prevents firing at targets who will be free to dodge
            if (remaining_lock >= arrival_time)
            {
                result.is_peak_opportunity = true;
                result.opportunity_score = 1.0f;
                result.adaptive_threshold = 0.50f;  // Lower threshold to ensure we take the shot

                // Still update window history so we resume tracking correctly after lock ends
                OpportunityWindow& window = tracker.get_opportunity_window(spell.spell_slot);
                window.update(current_time, result.hit_chance);
                return;
            }
            // If lock ends before spell lands, fall through to normal patience logic
            // The get_effective_move_speed() will handle partial freedom calculation
        }

        // ADAPTIVE PATIENCE: Calculate patience window based on spell cooldown
        // LOW-CD FIX: Old range (2.0-4.0s) was too long for Ezreal Q, Zeri Q, etc.
        // New range (0.25-1.0s) allows low-CD spells to fire quickly while still
        // waiting for peak opportunities on high-CD ults
        float spell_cooldown = 10.0f;  // Default fallback
        if (source && source->is_valid() && spell_slot >= 0 && spell_slot <= 3)
        {
            spell_entry* spell_entry_ptr = source->get_spell(spell_slot);
            if (spell_entry_ptr)
            {
                spell_cooldown = spell_entry_ptr->get_cooldown();
            }
        }
        float patience_window = std::clamp(spell_cooldown * 0.3f, 0.25f, 1.0f);

        // Safety: Validate spell slot (should be 0-3 for Q/W/E/R, or special slots)
        if (spell_slot < -1 || spell_slot > 10)
        {
            // Invalid spell slot - use slot 0 as fallback
            spell_slot = 0;
        }

        // Get or create opportunity window for this spell
        OpportunityWindow& window = tracker.get_opportunity_window(spell_slot);

        // Update window with current hit_chance
        window.update(current_time, result.hit_chance);

        // Calculate elapsed time since window started
        float elapsed_time = current_time - window.window_start_time;

        // Calculate opportunity score: how good is this moment relative to recent peak?
        // Clamped to [0, 1] range to ensure it's a valid probability
        if (window.peak_hit_chance > EPSILON)
        {
            result.opportunity_score = std::min(result.hit_chance / window.peak_hit_chance, 1.0f);
        }
        else
        {
            result.opportunity_score = 1.0f;  // First sample
        }

        // Calculate adaptive threshold FIRST (needed for peak detection)
        // Convert hitchance enum to float threshold
        float base_threshold = 0.65f;  // Default high
        if (spell.expected_hitchance == pred_sdk::hitchance::very_high)
            base_threshold = 0.80f;
        else if (spell.expected_hitchance == pred_sdk::hitchance::high)
            base_threshold = 0.65f;
        else if (spell.expected_hitchance == pred_sdk::hitchance::medium)
            base_threshold = 0.55f;
        else if (spell.expected_hitchance == pred_sdk::hitchance::low)
            base_threshold = 0.45f;

        result.adaptive_threshold = window.get_adaptive_threshold(base_threshold, elapsed_time);

        // Detect if this is a peak opportunity (uses adaptive_threshold, elapsed_time, and patience_window)
        result.is_peak_opportunity = window.is_peak_opportunity(current_time, result.hit_chance,
            result.adaptive_threshold, elapsed_time,
            patience_window);

        // OPPORTUNITY ENFORCEMENT: Gentle penalty for non-peak opportunities
        // RELAXED: Only penalize if < 80% (good shots should fire regardless of peak)
        // REDUCED: 10% penalty instead of 15% (avoid "double jeopardy" with stricter thresholds)
        // SKIP IF LOCKED: If target is currently animation-locked, don't penalize even if
        // lock ends before spell arrives. Shooting at a locked target is ALWAYS a good opportunity.
        bool target_currently_locked = tracker.is_animation_locked();
        if (!result.is_peak_opportunity && result.hit_chance < 0.80f && !target_currently_locked)
        {
            result.hit_chance *= 0.90f;  // 10% penalty for impatience
        }

        // Reset window if spell was likely cast (hit_chance suddenly drops or becomes invalid)
        // This happens when target moves out of range or dies
        // FIX: Use 20% threshold instead of 50% to prevent false resets during normal juking
        // A 50% drop can happen from normal dodging (80% -> 40%), not just casting
        // A 20% threshold (80% -> 16%) indicates actual major state change (out of range, died, flashed)
        if (result.hit_chance < window.last_hit_chance * 0.2f && elapsed_time > 1.0f)
        {
            // Significant drop - likely cast occurred, reset window
            window = OpportunityWindow();
            window.window_start_time = current_time;
        }
        window.last_hit_chance = result.hit_chance;
    }

    // =========================================================================
    // PHYSICS PREDICTOR IMPLEMENTATION
    // =========================================================================

    /**
     * Clamp position to pathable terrain
     * CRITICAL: Prevents predicting through walls
     *
     * If position is already pathable, returns it unchanged.
     * Otherwise, searches in expanding radius for closest pathable position.
     */
    static math::vector3 clamp_to_pathable(const math::vector3& pos)
    {
        // No nav mesh available - return as-is (can't validate)
        if (!g_sdk || !g_sdk->nav_mesh)
            return pos;

        // Make a copy since is_pathable takes non-const reference
        math::vector3 pos_copy = pos;

        // Already pathable - fast path
        if (g_sdk->nav_mesh->is_pathable(pos_copy))
            return pos;

        // Position is in wall - search for closest pathable position
        // Use spiral search pattern: check nearby positions in expanding rings
        constexpr float SEARCH_STEP = 10.f;  // 10 unit increments
        constexpr int MAX_RINGS = 8;         // Search up to 80 units (reduced from 15 for performance)
        constexpr float GOOD_ENOUGH_DIST = 20.f;  // Early exit if we find position within 20 units

        math::vector3 best_pos = pos;
        float best_distance = FLT_MAX;

        for (int ring = 1; ring <= MAX_RINGS; ++ring)
        {
            float radius = ring * SEARCH_STEP;
            int samples = ring * 8;  // More samples for larger rings

            for (int i = 0; i < samples; ++i)
            {
                float angle = (2.0f * PI * i) / samples;
                math::vector3 test_pos = pos;
                test_pos.x += std::cos(angle) * radius;
                test_pos.z += std::sin(angle) * radius;

                if (g_sdk->nav_mesh->is_pathable(test_pos))
                {
                    float dist = (test_pos - pos).magnitude();
                    if (dist < best_distance)
                    {
                        best_pos = test_pos;
                        best_distance = dist;

                        // PERFORMANCE: Early exit if we found a close enough position
                        if (dist < GOOD_ENOUGH_DIST)
                            return best_pos;
                    }
                }
            }

            // Found pathable position in this ring - return it
            if (best_distance < FLT_MAX)
                return best_pos;
        }

        // Couldn't find pathable position - return original (better than nothing)
        // This should be extremely rare (only if surrounded by walls on all sides)
        return pos;
    }

    /**
     * Estimate wall normal at a collision point by sampling nearby positions
     * Returns normalized vector pointing AWAY from the wall (into open space)
     */
    static math::vector3 estimate_wall_normal(const math::vector3& collision_point)
    {
        if (!g_sdk || !g_sdk->nav_mesh)
            return math::vector3(0, 0, 0);

        // Sample 8 directions around the collision point to find wall orientation
        constexpr float CHECK_DIST = 15.f;
        constexpr int NUM_SAMPLES = 8;

        math::vector3 normal(0, 0, 0);

        for (int i = 0; i < NUM_SAMPLES; ++i)
        {
            float angle = (2.0f * PI * i) / NUM_SAMPLES;
            math::vector3 offset(std::cos(angle) * CHECK_DIST, 0, std::sin(angle) * CHECK_DIST);
            math::vector3 test_pos = collision_point + offset;

            // If this direction is pathable (open), add it to normal
            // If it's a wall, subtract it
            if (g_sdk->nav_mesh->is_pathable(test_pos))
            {
                // Open space in this direction - normal points this way
                normal.x += offset.x;
                normal.z += offset.z;
            }
            else
            {
                // Wall in this direction - normal points opposite
                normal.x -= offset.x;
                normal.z -= offset.z;
            }
        }

        float mag = normal.magnitude();
        if (mag < 0.1f)
            return math::vector3(0, 0, 0);  // Surrounded or ambiguous

        return normal / mag;  // Normalized
    }

    /**
     * Calculate wall slide velocity when hitting terrain
     * Returns the component of velocity parallel to the wall surface
     */
    static math::vector3 calculate_wall_slide_velocity(
        const math::vector3& collision_point,
        const math::vector3& velocity)
    {
        math::vector3 normal = estimate_wall_normal(collision_point);

        if (normal.magnitude() < 0.1f)
            return math::vector3(0, 0, 0);  // Can't determine wall orientation

        // Project velocity onto wall surface (remove component going INTO wall)
        // Formula: V_slide = V - (V · N) * N
        float dot = velocity.x * normal.x + velocity.z * normal.z;

        // Only slide if moving INTO the wall (dot < 0 means velocity opposes normal)
        if (dot >= 0.f)
            return velocity;  // Moving away from wall, no slide needed

        math::vector3 slide_velocity;
        slide_velocity.x = velocity.x - normal.x * dot;
        slide_velocity.y = 0.f;
        slide_velocity.z = velocity.z - normal.z * dot;

        return slide_velocity;
    }

    /**
     * Find the exact collision point between start and end using binary search
     * Returns the last pathable position before hitting the wall
     */
    static math::vector3 find_wall_collision_point(
        const math::vector3& start,
        const math::vector3& end)
    {
        if (!g_sdk || !g_sdk->nav_mesh)
            return end;

        // Binary search for collision point
        math::vector3 low = start;
        math::vector3 high = end;
        constexpr int MAX_ITERATIONS = 8;  // ~1 unit precision at 256 unit range

        for (int i = 0; i < MAX_ITERATIONS; ++i)
        {
            math::vector3 mid;
            mid.x = (low.x + high.x) * 0.5f;
            mid.y = (low.y + high.y) * 0.5f;
            mid.z = (low.z + high.z) * 0.5f;

            if (g_sdk->nav_mesh->is_pathable(mid))
            {
                low = mid;  // Mid is pathable, collision is further
            }
            else
            {
                high = mid;  // Mid is wall, collision is closer
            }
        }

        return low;  // Last known pathable position
    }

    ReachableRegion PhysicsPredictor::compute_reachable_region(
        const math::vector3& current_pos,
        const math::vector3& current_velocity,
        float prediction_time,
        float move_speed,
        float turn_rate,
        float acceleration,
        float reaction_time)
    {
        ReachableRegion region;

        // =====================================================================
        // CRITICAL FIX: Account for INERTIA during reaction time
        // =====================================================================
        // Humans don't freeze during reaction time - they continue drifting
        // in their current direction due to inertia!
        //
        // Example: Target moving right at 400 MS, spell lands in 0.5s
        //   - Reaction time: 0.25s
        //   - During reaction: Drifts 100 units right (400 * 0.25)
        //   - Reachable center: current_pos + (100, 0)
        //   - NOT current_pos!
        //
        // Without this: System aims behind moving targets on fast spells
        // =====================================================================
        float non_reactive_time = std::min(prediction_time, reaction_time);

        // FIX: Only apply drift if target is ACTIVELY moving
        // If they've stopped (velocity near-zero), drift = 0 (they stay put)
        // This prevents overshooting when targets tap 'S' or reach destination
        bool is_actively_moving = current_velocity.magnitude() > 50.f;  // ~50 units/s threshold
        math::vector3 drift_offset = is_actively_moving ? (current_velocity * non_reactive_time) : math::vector3{};
        math::vector3 center_pos = current_pos + drift_offset;

        // FIX: Clamp to pathable terrain (drift might push into walls)
        region.center = clamp_to_pathable(center_pos);
        region.velocity = current_velocity;  // Store for momentum weighting

        // CRITICAL FIX: Subtract reaction time from prediction time
        // Humans cannot react instantly - they need 200-300ms to see and respond
        // This dramatically reduces reachable area for realistic predictions
        float effective_dodge_time = std::max(0.f, prediction_time - reaction_time);

        if (effective_dodge_time < EPSILON)
        {
            // No time to dodge - region is essentially current position
            region.max_radius = 0.f;
            region.area = 0.f;
            return region;
        }

        // Maximum distance considering acceleration from current velocity
        // Now uses EFFECTIVE dodge time (with reaction time subtracted)
        float current_speed = current_velocity.magnitude();
        float speed_diff = move_speed - current_speed;

        float max_distance;
        if (speed_diff > 0.f && acceleration > 0.f)
        {
            // Time to reach max speed
            float accel_time = std::min(speed_diff / acceleration, effective_dodge_time);

            // Distance during acceleration: d = v₀*t + 0.5*a*t²
            float accel_distance = current_speed * accel_time +
                0.5f * acceleration * accel_time * accel_time;

            // Distance at max speed
            float max_speed_time = effective_dodge_time - accel_time;
            float max_speed_distance = move_speed * max_speed_time;

            max_distance = accel_distance + max_speed_distance;
        }
        else
        {
            // Already at or above max speed (or instant speed)
            max_distance = move_speed * effective_dodge_time;
        }

        region.max_radius = max_distance;

        // NOTE: turn_rate parameter is currently unused because League of Legends champions
        // have instant turn rate (can change direction immediately). If implementing for games
        // with turn rate mechanics (e.g., Dota 2), reachable region should be a sector/cone
        // instead of a full circle.
        (void)turn_rate; // Suppress unused parameter warning

        // Discretize boundary (circle approximation - full 360° reachability)
        // Boundary points centered at drift-adjusted position (region.center)
        constexpr int BOUNDARY_POINTS = 32;
        for (int i = 0; i < BOUNDARY_POINTS; ++i)
        {
            float angle = (2.f * PI * i) / BOUNDARY_POINTS;
            math::vector3 boundary_point = region.center;  // Use drift-adjusted center
            boundary_point.x += max_distance * std::cos(angle);
            boundary_point.z += max_distance * std::sin(angle);
            region.boundary_points.push_back(boundary_point);
        }

        // Area = πr²
        region.area = PI * max_distance * max_distance;

        // WALL-HUGGER DETECTION: Sample terrain around reachable region
        // If target is against a wall, they can't dodge in that direction
        // This significantly increases hit probability for choke points
        region.pathable_ratio = 1.0f;  // Default: fully pathable
        if (g_sdk && g_sdk->nav_mesh && max_distance > 10.f)
        {
            int pathable_count = 0;
            constexpr int SAMPLE_DIRECTIONS = 8;  // N, NE, E, SE, S, SW, W, NW

            for (int i = 0; i < SAMPLE_DIRECTIONS; ++i)
            {
                float angle = (2.f * PI * i) / SAMPLE_DIRECTIONS;
                math::vector3 sample_point = region.center;
                // Sample at 80% of max radius to account for hitbox overlap
                sample_point.x += max_distance * 0.8f * std::cos(angle);
                sample_point.z += max_distance * 0.8f * std::sin(angle);

                if (g_sdk->nav_mesh->is_pathable(sample_point))
                {
                    pathable_count++;
                }
            }

            region.pathable_ratio = static_cast<float>(pathable_count) / SAMPLE_DIRECTIONS;

            // Clamp to minimum 0.25 (target can always try to dodge somewhere)
            region.pathable_ratio = std::max(0.25f, region.pathable_ratio);
        }

        return region;
    }

    math::vector3 PhysicsPredictor::predict_linear_position(
        const math::vector3& current_pos,
        const math::vector3& current_velocity,
        float prediction_time)
    {
        return current_pos + current_velocity * prediction_time;
    }

    math::vector3 PhysicsPredictor::predict_on_path(
        game_object* target,
        float prediction_time,
        float delay_start_time)
    {
        if (!target || !target->is_valid())
            return math::vector3{};

        // Use server position (authoritative for hit detection, avoids 30-100ms client lag)
        math::vector3 position = target->get_server_position();

        // CC CHECK: Immobilized targets can't move
        if (target->has_buff_of_type(buff_type::stun) ||
            target->has_buff_of_type(buff_type::snare) ||
            target->has_buff_of_type(buff_type::charm) ||
            target->has_buff_of_type(buff_type::fear) ||
            target->has_buff_of_type(buff_type::taunt) ||
            target->has_buff_of_type(buff_type::suppression) ||
            target->has_buff_of_type(buff_type::knockup))
        {
            return position;  // CC'd - stay at current position
        }

        auto path = target->get_path();

        // No path or stationary - return current position
        if (path.size() <= 1)
            return position;

        float move_speed = target->get_move_speed();
        if (move_speed < 1.f)
            return position;

        // ANIMATION LOCK: Stop-then-go model
        // Target stays still during lock, then moves at full speed
        float effective_movement_time = prediction_time;
        if (delay_start_time > 0.f)
        {
            effective_movement_time = std::max(0.f, prediction_time - delay_start_time);
            if (effective_movement_time <= 0.f)
                return position;  // Lock lasts longer than prediction time

            // PATH STALENESS: Long locks (>0.2s) make paths unreliable
            // After attack/cast, targets often issue new movement commands
            if (delay_start_time > 0.2f)
            {
                float staleness_factor = 1.0f - std::min((delay_start_time - 0.2f) / 0.4f, 0.5f);
                effective_movement_time *= staleness_factor;
            }
        }

        // INTELLIGENT HEURISTICS BASED ON PATH STATE

        // Get path age from tracker if available
        float path_age = 0.f;
        if (g_sdk && g_sdk->object_manager)
        {
            auto tracker_map = HybridPrediction::HybridFusionEngine::get_tracker_map();
            auto it = tracker_map.find(target->get_network_id());
            if (it != tracker_map.end())
            {
                float current_time = g_sdk->clock_facade ? g_sdk->clock_facade->get_game_time() : 0.f;
                path_age = current_time - it->second.get_last_path_update_time();
            }
        }

        // HEURISTIC 1: Start-of-Path Dampening
        // Just clicked (< 0.1s ago) → assume still accelerating, reduce effective speed
        float speed_multiplier = 1.0f;
        if (path_age < 0.1f)
        {
            speed_multiplier = 0.85f;  // 15% dampening for acceleration phase
        }

        float distance_to_travel = move_speed * effective_movement_time * speed_multiplier;

        // HEURISTIC 2: End-of-Path Clamping
        // Calculate total path distance to detect approaching destination
        float total_path_distance = 0.f;
        for (size_t i = 1; i < path.size(); ++i)
        {
            total_path_distance += (path[i] - path[i-1]).magnitude();
        }

        // Distance from current position to path start
        float dist_to_path_start = (position - path[0]).magnitude();
        float remaining_path = total_path_distance - dist_to_path_start;

        // Clamp travel distance to not overshoot destination
        if (distance_to_travel > remaining_path)
        {
            distance_to_travel = remaining_path;
        }

        // SIMPLE LINEAR PATH ITERATION
        // Start from current position, iterate through waypoints at constant velocity
        float distance_traveled = 0.f;

        for (size_t i = 1; i < path.size(); ++i)
        {
            math::vector3 segment_start = (i == 1) ? position : path[i - 1];
            math::vector3 segment_end = path[i];

            math::vector3 segment_vec = segment_end - segment_start;
            float segment_length = segment_vec.magnitude();

            if (segment_length < 0.001f)
                continue;  // Skip zero-length segments

            // Check if prediction endpoint is on this segment
            if (distance_traveled + segment_length >= distance_to_travel)
            {
                // Found the segment - linear interpolate
                float distance_into_segment = distance_to_travel - distance_traveled;
                math::vector3 direction = segment_vec / segment_length;
                math::vector3 predicted_pos = segment_start + direction * distance_into_segment;

                // WALL COLLISION CHECK: Ensure predicted position is pathable
                if (g_sdk && g_sdk->nav_mesh && !g_sdk->nav_mesh->is_pathable(predicted_pos))
                {
                    // Position is in wall - clamp to segment start (safe position)
                    return segment_start;
                }

                return predicted_pos;
            }

            distance_traveled += segment_length;
        }

        // Traveled entire path - return final waypoint
        return path.back();
    }

    float PhysicsPredictor::compute_physics_hit_probability(
        const math::vector3& cast_position,
        float projectile_radius,
        const ReachableRegion& reachable_region)
    {
        // CRITICAL FIX: Check for zero area - target cannot dodge (CC'd/instant spell)
        // Hit probability is BINARY: 1.0 if cast position is within spell radius, 0.0 otherwise
        // BUG FIX: Previously returned 1.0 for ANY cast position, even miles away!
        if (reachable_region.area < EPSILON)
        {
            float dist = distance_2d(cast_position, reachable_region.center);  // 2D for ground plane
            return (dist <= projectile_radius) ? 1.0f : 0.0f;
        }

        // Distance from cast position to predicted target center (2D ground plane)
        math::vector3 to_cast = cast_position - reachable_region.center;
        float distance = distance_2d(cast_position, reachable_region.center);

        // Gaussian kernel: target most likely at predicted center
        // σ = max_radius / 2.5 (so ~95% within max_radius)
        float sigma = reachable_region.max_radius / 2.5f;
        if (sigma < 1.0f)
            sigma = 1.0f;  // Minimum sigma to avoid numerical issues

        // MOMENTUM WEIGHTING: Penalize aiming behind moving targets
        // Players are more likely to continue their current direction than turn 180°
        float vel_speed = reachable_region.velocity.magnitude();
        if (vel_speed > 50.f && distance > 1.0f)
        {
            // Calculate alignment: 1.0 = forward, -1.0 = backward
            math::vector3 vel_dir = reachable_region.velocity / vel_speed;
            // FIX: Must use 2D direction for normalization (cannot divide 3D vector by 2D distance!)
            math::vector3 to_cast_2d = flatten_2d(to_cast);
            math::vector3 to_cast_dir = to_cast_2d / distance;
            float alignment = to_cast_dir.x * vel_dir.x + to_cast_dir.y * vel_dir.y + to_cast_dir.z * vel_dir.z;

            // If cast position is behind the target's movement direction, tighten sigma
            // This makes it harder to predict they'll turn around
            if (alignment < 0.f)
            {
                // Scale penalty: fully backward (-1.0) = 30% tighter sigma
                // Perpendicular (0) = no penalty
                float penalty = 1.0f + (alignment * 0.3f);  // Range: 0.7 to 1.0
                sigma *= penalty;
            }
        }

        // Gaussian weight (clamped to avoid exp overflow)
        float exponent = -(distance * distance) / (2.0f * sigma * sigma);
        exponent = std::max(-50.0f, exponent);  // Clamp to prevent underflow
        float gaussian_weight = std::exp(exponent);

        // Compute intersection area
        float intersection_area = circle_circle_intersection_area(
            cast_position, projectile_radius,
            reachable_region.center, reachable_region.max_radius
        );

        // Weight area probability by Gaussian
        // WALL-HUGGER BOOST: If target can't dodge in all directions, reduce effective area
        // pathable_ratio of 0.5 means half the circle is blocked by walls → higher hit chance
        float effective_area = reachable_region.area * reachable_region.pathable_ratio;
        float safe_area = std::max(effective_area, 1.0f);
        float area_probability = intersection_area / safe_area;
        float weighted_probability = gaussian_weight * area_probability;

        // Bonus if predicted center is inside projectile
        if (distance < projectile_radius)
        {
            weighted_probability = std::max(weighted_probability, gaussian_weight * 0.85f);
        }

        return std::min(1.f, weighted_probability);
    }

    float PhysicsPredictor::compute_time_to_dodge_probability(
        const math::vector3& target_position,
        const math::vector3& cast_position,
        float projectile_radius,
        float target_move_speed,
        float arrival_time,
        float reaction_time)
    {
        /**
         * Time-to-Dodge Physics Probability
         * ==================================
         *
         * Instead of area intersection, we measure: "Can the target physically escape?"
         *
         * Algorithm:
         * 1. Find distance from target to edge of spell hitbox
         * 2. Calculate time needed to run that distance
         * 3. Compare to time available (arrival - reaction)
         * 4. Return ratio (or 1.0 if impossible to dodge)
         *
         * Benefits:
         * - Returns 1.0 (guaranteed hit) when escape is physically impossible
         * - Intuitive: 90% of dodge time needed = 90% physics probability
         * - No arbitrary area ratios
         * - Accounts for human reaction time
         *
         * NOTE: Current implementation assumes instant turn rate (valid for League).
         * Future enhancement: Account for velocity direction when calculating escape time.
         * If target is moving INTO spell, they need to decelerate then accelerate out:
         *   time_needed = decel_time + accel_time + (distance / move_speed)
         * This adds ~20-35% to escape time for targets moving into the spell.
         * Trade-off: Current simplification acceptable for performance.
         */

         // Validate inputs
        if (target_move_speed < EPSILON || arrival_time < EPSILON)
            return 0.f;

        // Calculate distance from target to spell center (2D ground plane)
        float distance_to_center = distance_2d(target_position, cast_position);

        // Calculate distance to escape (distance to edge of hitbox)
        // If target is outside spell, they're already safe
        if (distance_to_center >= projectile_radius)
        {
            // Debug: Log when returning 0 due to outside radius
            if (PredictionSettings::get().enable_debug_logging && g_sdk)
            {
                char dbg[256];
                snprintf(dbg, sizeof(dbg),
                    "[Danny.Prediction] TIME_DODGE: dist=%.1f >= radius=%.1f RETURNING 0",
                    distance_to_center, projectile_radius);
                g_sdk->log_console(dbg);
            }
            return 0.f;
        }

        // Distance needed to run to safety
        float distance_to_edge = projectile_radius - distance_to_center;

        // CRITICAL: Check for zero move speed to avoid division by zero
        if (target_move_speed < EPSILON)
            return 1.0f;  // Can't move = guaranteed hit

        // Time needed to escape
        float time_needed_to_escape = distance_to_edge / target_move_speed;

        // Time available to dodge (subtract reaction time)
        float time_available = arrival_time - reaction_time;

        // If reaction time >= arrival time, target has no time to react
        if (time_available <= 0.f)
            return 1.0f;  // Guaranteed hit

        // If they cannot physically escape in time, guaranteed hit
        if (time_needed_to_escape >= time_available)
            return 1.0f;

        // =====================================================================
        // TERRAIN BLOCKING DETECTION (Choke Point / Wall Trap Detection)
        // =====================================================================
        // Check if target is trapped against walls/terrain by testing escape paths
        // perpendicular to the spell's aim direction. This significantly improves
        // accuracy in common scenarios:
        //   - Jungle fights (narrow paths between walls)
        //   - Tower dives (wall behind target)
        //   - River choke points
        //   - Lane edge positioning
        //
        // Logic:
        //   - Both sides blocked → 1.0 (trapped, guaranteed hit)
        //   - One side blocked → 1.5x probability (predictable dodge direction)
        //   - Both sides open → normal probability
        // =====================================================================

        // Calculate aim direction (from spell center to target)
        math::vector3 aim_dir = (target_position - cast_position);
        float aim_magnitude = aim_dir.magnitude();

        // CRITICAL: Check nav_mesh is available before using it
        if (aim_magnitude > EPSILON && g_sdk && g_sdk->nav_mesh)
        {
            aim_dir = aim_dir / aim_magnitude;  // Normalize

            // Calculate perpendicular escape directions (left and right)
            // For 2D plane (y is up in League), perpendicular is rotation in xz plane
            math::vector3 escape_dir_left(-aim_dir.z, 0.f, aim_dir.x);   // 90° left
            math::vector3 escape_dir_right(aim_dir.z, 0.f, -aim_dir.x);  // 90° right

            // Calculate escape points (slightly beyond spell edge for safety margin)
            constexpr float SAFETY_MARGIN = 20.f;  // Extra distance for safe dodging
            float escape_distance = distance_to_edge + SAFETY_MARGIN;

            math::vector3 escape_point_left = target_position + escape_dir_left * escape_distance;
            math::vector3 escape_point_right = target_position + escape_dir_right * escape_distance;

            // Check if escape paths are walkable using nav mesh
            bool can_dodge_left = true;
            bool can_dodge_right = true;
            if (g_sdk && g_sdk->nav_mesh)
            {
                can_dodge_left = g_sdk->nav_mesh->is_pathable(escape_point_left);
                can_dodge_right = g_sdk->nav_mesh->is_pathable(escape_point_right);
            }

            // Trapped against wall on both sides = guaranteed hit
            if (!can_dodge_left && !can_dodge_right)
            {
                return 1.0f;  // TRAPPED!
            }

            // One side blocked = predictable dodge direction
            // Target MUST dodge to the open side, increasing hit probability
            if (!can_dodge_left || !can_dodge_right)
            {
                float probability = time_needed_to_escape / time_available;
                return std::clamp(probability * 1.5f, 0.f, 1.f);  // 50% boost
            }
        }

        // Sigmoid function for realistic dodge difficulty curve
        // Linear ratio doesn't capture human dodge thresholds well
        float ratio = time_needed_to_escape / time_available;

        // SIGMOID: Balanced parameters (adjusted for 60/40 physics fusion)
        // MIDPOINT=0.50 is mathematically neutral: need half the time = 50% chance
        // STEEPNESS tuned to 18 for balanced probability spread
        // Examples: 0.4s slack (0.6 ratio) → 82% | 0.2s slack (0.7 ratio) → 94%
        constexpr float SIGMOID_STEEPNESS = 18.f;   // Balanced transition (was 25, then 15)
        constexpr float SIGMOID_MIDPOINT = 0.50f;   // Neutral: 50% time = 50% chance

        // Clamp exponent to prevent overflow/underflow
        float exponent = -SIGMOID_STEEPNESS * (ratio - SIGMOID_MIDPOINT);
        exponent = std::clamp(exponent, -20.0f, 20.0f);

        float sigmoid_probability = 1.0f / (1.0f + std::exp(exponent));

        return std::clamp(sigmoid_probability, 0.f, 1.f);
    }

    float PhysicsPredictor::compute_arrival_time(
        const math::vector3& source_pos,
        const math::vector3& target_pos,
        float projectile_speed,
        float cast_delay,
        float proc_delay)
    {
        // HIGH GROUND FIX: Use 2D distance (ignore Y/height)
        // League logic is 2D - river vs mid lane height shouldn't affect projectile travel
        float distance = distance_2d(source_pos, target_pos);

        // Instant spell: No projectile travel time
        // proc_delay still applies (e.g., Syndra Q takes 0.6s to "pop" after placement)
        if (projectile_speed < EPSILON || projectile_speed >= FLT_MAX / 2.f)
        {
            return cast_delay + proc_delay;  // Windup + activation delay
        }

        // PING COMPENSATION (Zero-Ping Fallacy Fix):
        // Timeline of a spell cast:
        // - t=0 (client): We press cast button, command packet sent
        // - t=ping/2 (server): Server receives our cast command, windup starts
        // - t=ping/2+delay (server): Projectile launches
        // - t=ping/2+delay+travel (server): Projectile arrives
        // - t=ping/2+delay+travel+proc_delay (server): Spell deals damage (if proc_delay > 0)
        //
        // get_server_position() is where they are NOW on the server (t=0).
        // But our spell doesn't START until t=ping/2, by which time they've moved.
        //
        // We add ping/2 to account for the one-way packet travel time.
        // This is the time between "we decide to cast" and "server starts our cast".
        float ping_delay = 0.f;
        if (g_sdk && g_sdk->net_client)
        {
            // Ping is round-trip in ms, divide by 2 for one-way, convert to seconds
            float ping_ms = static_cast<float>(g_sdk->net_client->get_ping());
            ping_delay = ping_ms / 2000.f;

            // Clamp to reasonable range (5ms - 150ms one-way)
            // Very low ping: still some processing delay
            // Very high ping: cap to avoid wild predictions
            ping_delay = std::clamp(ping_delay, 0.005f, 0.15f);
        }

        return ping_delay + cast_delay + proc_delay + (distance / projectile_speed);
    }

    // =========================================================================
    // MINIMUM ENCLOSING CIRCLE (Welzl's Algorithm) for Multi-Target AOE
    // =========================================================================

    Circle PhysicsPredictor::make_circle_from_2_points(const math::vector3& p1, const math::vector3& p2)
    {
        math::vector3 center = (p1 + p2) * 0.5f;
        float radius = (p2 - p1).magnitude() * 0.5f;
        return Circle(center, radius);
    }

    Circle PhysicsPredictor::make_circle_from_3_points(const math::vector3& p1, const math::vector3& p2, const math::vector3& p3)
    {
        // Use circumcircle formula for 2D points (ignore Y coordinate)
        float ax = p1.x, az = p1.z;
        float bx = p2.x, bz = p2.z;
        float cx = p3.x, cz = p3.z;

        float d = 2.f * (ax * (bz - cz) + bx * (cz - az) + cx * (az - bz));

        if (std::abs(d) < 0.001f)
        {
            // Points are collinear - return circle from two farthest points
            float dist12 = (p2 - p1).magnitude();
            float dist13 = (p3 - p1).magnitude();
            float dist23 = (p3 - p2).magnitude();

            if (dist12 >= dist13 && dist12 >= dist23)
                return make_circle_from_2_points(p1, p2);
            else if (dist13 >= dist23)
                return make_circle_from_2_points(p1, p3);
            else
                return make_circle_from_2_points(p2, p3);
        }

        float ux = ((ax * ax + az * az) * (bz - cz) + (bx * bx + bz * bz) * (cz - az) + (cx * cx + cz * cz) * (az - bz)) / d;
        float uz = ((ax * ax + az * az) * (cx - bx) + (bx * bx + bz * bz) * (ax - cx) + (cx * cx + cz * cz) * (bx - ax)) / d;

        math::vector3 center(ux, p1.y, uz);  // Use Y from first point
        float radius = (p1 - center).magnitude();

        return Circle(center, radius);
    }

    Circle PhysicsPredictor::welzl_recursive(
        std::vector<math::vector3>& points,
        std::vector<math::vector3> boundary,
        size_t n)
    {
        // Base cases
        if (n == 0 || boundary.size() == 3)
        {
            if (boundary.size() == 0)
                return Circle();
            else if (boundary.size() == 1)
                return Circle(boundary[0], 0.f);
            else if (boundary.size() == 2)
                return make_circle_from_2_points(boundary[0], boundary[1]);
            else
                return make_circle_from_3_points(boundary[0], boundary[1], boundary[2]);
        }

        // Pick a random point
        size_t idx = n - 1;
        math::vector3 p = points[idx];

        // Recursive call without p
        Circle circle = welzl_recursive(points, boundary, n - 1);

        // If p is inside circle, return it
        if (circle.contains(p))
            return circle;

        // Otherwise, p must be on the boundary
        boundary.push_back(p);
        return welzl_recursive(points, boundary, n - 1);
    }

    Circle PhysicsPredictor::compute_minimum_enclosing_circle(const std::vector<math::vector3>& points)
    {
        if (points.empty())
            return Circle();

        if (points.size() == 1)
            return Circle(points[0], 0.f);

        // Make a mutable copy for Welzl's algorithm
        std::vector<math::vector3> points_copy = points;

        // Randomize for expected O(n) time (avoid worst case)
        // Use static RNG to avoid reinitialization overhead
        static std::random_device rd;
        static std::mt19937 rng(rd());
        std::shuffle(points_copy.begin(), points_copy.end(), rng);

        return welzl_recursive(points_copy, std::vector<math::vector3>(), points_copy.size());
    }

    float PhysicsPredictor::circle_circle_intersection_area(
        const math::vector3& c1, float r1,
        const math::vector3& c2, float r2)
    {
        // NUMERIC STABILITY: Guard against zero/negative radii
        constexpr float MIN_RADIUS = 1e-6f;
        if (r1 < MIN_RADIUS || r2 < MIN_RADIUS)
            return 0.f;

        float d = (c2 - c1).magnitude();

        // No intersection
        if (d >= r1 + r2)
            return 0.f;

        // Complete containment
        if (d <= std::abs(r1 - r2))
        {
            float smaller_r = std::min(r1, r2);
            return PI * smaller_r * smaller_r;
        }

        // NUMERIC STABILITY: Guard against division by zero when circles are at same position
        if (d < MIN_RADIUS)
        {
            float smaller_r = std::min(r1, r2);
            return PI * smaller_r * smaller_r;
        }

        // Partial intersection - use lens formula
        // A = r1²*arccos((d²+r1²-r2²)/(2*d*r1)) + r2²*arccos((d²+r2²-r1²)/(2*d*r2))
        //     - 0.5*sqrt((r1+r2-d)*(r1-r2+d)*(-r1+r2+d)*(r1+r2+d))

        float d2 = d * d;
        float r1_2 = r1 * r1;
        float r2_2 = r2 * r2;

        float alpha = std::acos(std::clamp((d2 + r1_2 - r2_2) / (2.f * d * r1), -1.f, 1.f));
        float beta = std::acos(std::clamp((d2 + r2_2 - r1_2) / (2.f * d * r2), -1.f, 1.f));

        float area = r1_2 * alpha + r2_2 * beta;

        // NUMERIC STABILITY: Guard against negative sqrt due to floating point errors
        float sqrt_term = (r1 + r2 - d) * (r1 - r2 + d) * (-r1 + r2 + d) * (r1 + r2 + d);
        if (sqrt_term > 0.f)
            area -= 0.5f * std::sqrt(sqrt_term);

        return area;
    }

    // =========================================================================
    // BEHAVIOR PREDICTOR IMPLEMENTATION
    // =========================================================================

    BehaviorPDF BehaviorPredictor::build_pdf_from_history(
        const TargetBehaviorTracker& tracker,
        float prediction_time,
        float move_speed)
    {
        BehaviorPDF pdf = tracker.build_behavior_pdf(prediction_time, move_speed);
        return pdf;
    }

    float BehaviorPredictor::compute_behavior_hit_probability(
        const math::vector3& cast_position,
        float projectile_radius,
        const BehaviorPDF& pdf)
    {
        // If PDF has no data (not normalized or empty), return neutral 1.0
        // This allows fallback to physics-only prediction when behavior data is insufficient
        if (pdf.total_probability < EPSILON)
            return 1.0f;

        // Direct grid summation (more accurate than Monte Carlo sampling)
        // Sum probability mass of all cells whose centers fall inside the hit circle
        float radius_sq = projectile_radius * projectile_radius;
        float prob = 0.f;

        // PERFORMANCE OPTIMIZATION: Calculate bounding box of circle in grid coordinates
        // Only iterate cells that could possibly be inside the circle
        // This reduces checks from 1024 (32×32) down to ~20-80 cells (10-50x speedup!)

        // Find min/max world coordinates of circle bounding box
        float min_wx = cast_position.x - projectile_radius;
        float max_wx = cast_position.x + projectile_radius;
        float min_wz = cast_position.z - projectile_radius;
        float max_wz = cast_position.z + projectile_radius;

        // Convert world coordinates to grid coordinates
        int grid_center = BehaviorPDF::GRID_SIZE / 2;

        // CRASH PROTECTION: Check cell_size before division
        if (pdf.cell_size < 0.1f)
            return 1.0f;  // Fallback neutral probability

        int min_x = static_cast<int>((min_wx - pdf.origin.x) / pdf.cell_size) + grid_center;
        int max_x = static_cast<int>((max_wx - pdf.origin.x) / pdf.cell_size) + grid_center + 1;
        int min_z = static_cast<int>((min_wz - pdf.origin.z) / pdf.cell_size) + grid_center;
        int max_z = static_cast<int>((max_wz - pdf.origin.z) / pdf.cell_size) + grid_center + 1;

        // Clamp to grid bounds [0, GRID_SIZE)
        min_x = std::max(0, min_x);
        max_x = std::min(BehaviorPDF::GRID_SIZE, max_x);
        min_z = std::max(0, min_z);
        max_z = std::min(BehaviorPDF::GRID_SIZE, max_z);

        // Iterate ONLY cells within bounding box (much faster!)
        for (int x = min_x; x < max_x; ++x)
        {
            for (int z = min_z; z < max_z; ++z)
            {
                // World position of cell center
                float wx = pdf.origin.x + (x - grid_center + 0.5f) * pdf.cell_size;
                float wz = pdf.origin.z + (z - grid_center + 0.5f) * pdf.cell_size;

                // Check if cell center is inside hit circle
                float dx = wx - cast_position.x;
                float dz = wz - cast_position.z;

                if (dx * dx + dz * dz <= radius_sq)
                {
                    prob += pdf.pdf_grid[x][z];
                }
            }
        }

        // PDF is normalized (sums to 1), so this sum is the exact hit probability
        return std::clamp(prob, 0.f, 1.f);
    }

    math::vector3 BehaviorPredictor::predict_from_behavior(
        const TargetBehaviorTracker& tracker,
        float prediction_time)
    {
        const auto& history = tracker.get_history();
        if (history.empty())
            return math::vector3{};

        const auto& latest = history.back();

        // Weighted average of predicted positions
        math::vector3 predicted_pos{};
        float total_weight = 0.f;

        // ADAPTIVE DECAY RATE: Adjust based on target mobility
        float decay_rate = get_adaptive_decay_rate(latest.velocity.magnitude());
        float current_weight = 1.0f;  // decay_rate^0 = 1.0

        for (size_t i = 0; i < std::min(history.size(), size_t(20)); ++i)
        {
            size_t idx = history.size() - 1 - i;
            const auto& snapshot = history[idx];

            // Use multiplicative accumulation instead of std::pow for O(1) per iteration
            float weight = current_weight;
            current_weight *= decay_rate;  // Prepare next weight

            predicted_pos = predicted_pos + (snapshot.position + snapshot.velocity * prediction_time) * weight;
            total_weight += weight;
        }

        if (total_weight > EPSILON)
            predicted_pos = predicted_pos / total_weight;

        return predicted_pos;
    }

    void BehaviorPredictor::apply_contextual_factors(
        BehaviorPDF& pdf,
        const TargetBehaviorTracker& tracker,
        game_object* target)
    {
        (void)target;  // Unused - reserved for HP pressure and minion CS detection

        // Apply HP pressure - low HP targets tend to retreat
        // Apply CS patterns - targets approach low HP minions
        // Apply animation locks - targets are stationary during AA/spells

        // This is a simplified version - full implementation would:
        // 1. Check nearby low HP minions and bias PDF toward them
        // 2. Check target HP and bias PDF away from threats
        // 3. Boost probability at current position if animation locked

        if (tracker.is_animation_locked())
        {
            const auto& history = tracker.get_history();
            if (!history.empty())
            {
                // Strong bias toward current position
                pdf.add_weighted_sample(history.back().position, 2.0f);
                pdf.normalize();
            }
        }
    }

    // =========================================================================
    // HYBRID FUSION ENGINE IMPLEMENTATION
    // =========================================================================

    HybridPredictionResult HybridFusionEngine::compute_hybrid_prediction(
        game_object* source,
        game_object* target,
        const pred_sdk::spell_data& spell,
        TargetBehaviorTracker& tracker)
    {
        HybridPredictionResult result;

        // FIX: Add dead checks - is_valid() doesn't return false for dead targets
        if (!source || !target || !source->is_valid() || !target->is_valid() ||
            source->is_dead() || target->is_dead())
        {
            result.is_valid = false;
            return result;
        }

        // Validate SDK is initialized
        if (!g_sdk || !g_sdk->clock_facade)
        {
            result.is_valid = false;
            result.reasoning = "SDK not initialized";
            return result;
        }

        // =============================================================================
        // EDGE CASE DETECTION AND HANDLING
        // =============================================================================

        EdgeCases::EdgeCaseAnalysis edge_cases = EdgeCases::analyze_target(target, source);

        // Filter out invalid targets
        if (edge_cases.is_clone)
        {
            result.is_valid = false;
            result.reasoning = "Target is a clone (Shaco/Wukong/LeBlanc/Neeko)";
            return result;
        }

        if (edge_cases.blocked_by_windwall)
        {
            result.is_valid = false;
            result.reasoning = "Projectile will be blocked by windwall (Yasuo/Samira/Braum)";
            return result;
        }

        // Handle stasis (Zhonya's, GA, Bard R) - PERFECT TIMING
        if (edge_cases.stasis.is_in_stasis)
        {
            if (!g_sdk || !g_sdk->clock_facade)
            {
                result.is_valid = false;
                return result;
            }
            float current_time = g_sdk->clock_facade->get_game_time();
            float spell_travel_time = PhysicsPredictor::compute_arrival_time(
                source->get_position(),
                target->get_position(),
                spell.projectile_speed,
                spell.delay
            );

            float cast_delay = EdgeCases::calculate_stasis_cast_timing(
                edge_cases.stasis,
                spell_travel_time,
                current_time
            );

            if (cast_delay < 0.f)
            {
                // Can't time it properly
                result.is_valid = false;
                result.reasoning = "Stasis timing impossible - travel time too long";
                return result;
            }

            if (cast_delay > 0.f)
            {
                // Need to wait before casting
                result.is_valid = false;
                result.reasoning = "Wait " + std::to_string(cast_delay) + "s for stasis exit timing";
                return result;
            }

            // Perfect timing! Cast now for guaranteed hit on stasis exit
            // But first check if in range (2D - High Ground Fix)
            math::vector3 to_stasis = edge_cases.stasis.exit_position - source->get_position();
            if (magnitude_2d(to_stasis) > spell.range)
            {
                result.is_valid = false;
                result.reasoning = "Stasis target out of range";
                return result;
            }

            result.cast_position = edge_cases.stasis.exit_position;
            result.hit_chance = 1.0f;  // 100% guaranteed
            result.physics_contribution = 1.0f;
            result.behavior_contribution = 1.0f;
            result.confidence_score = 1.0f;
            result.is_valid = true;
            result.reasoning = "STASIS EXIT PREDICTION - Spell will hit exactly when " +
                edge_cases.stasis.stasis_type + " ends. GUARANTEED HIT!";
            return result;
        }

        // Handle channeling/recall - HIGH PRIORITY STATIONARY TARGET
        if (edge_cases.channel.is_channeling || edge_cases.channel.is_recalling)
        {
            float spell_travel_time = PhysicsPredictor::compute_arrival_time(
                source->get_position(),
                target->get_position(),
                spell.projectile_speed,
                spell.delay
            );

            bool can_interrupt = EdgeCases::can_interrupt_channel(
                edge_cases.channel,
                spell_travel_time
            );

            if (!can_interrupt)
            {
                result.is_valid = false;
                result.reasoning = "Channel will finish before spell arrives";
                return result;
            }

            // Check if in range (2D - High Ground Fix)
            math::vector3 to_channel = edge_cases.channel.position - source->get_position();
            if (magnitude_2d(to_channel) > spell.range)
            {
                result.is_valid = false;
                result.reasoning = "Channeling target out of range";
                return result;
            }

            // Stationary target - 100% hit chance
            result.cast_position = edge_cases.channel.position;
            result.hit_chance = 1.0f;
            result.physics_contribution = 1.0f;
            result.behavior_contribution = 1.0f;
            result.confidence_score = 1.0f;
            result.is_valid = true;

            std::string action = edge_cases.channel.is_recalling ? "RECALL" : "CHANNEL";
            result.reasoning = action + " INTERRUPT - Target is stationary. GUARANTEED HIT!";
            return result;
        }

        // Handle dash prediction - ENDPOINT WITH TIMING VALIDATION (if enabled)
        if (edge_cases.dash.is_dashing && PredictionSettings::get().enable_dash_prediction)
        {
            float spell_travel_time = PhysicsPredictor::compute_arrival_time(
                source->get_position(),
                edge_cases.dash.dash_end_position,
                spell.projectile_speed,
                spell.delay
            );

            float current_time = 0.f;
            if (g_sdk && g_sdk->clock_facade)
                current_time = g_sdk->clock_facade->get_game_time();

            // Calculate time relationship between spell arrival and dash end
            float time_after_dash = spell_travel_time - edge_cases.dash.dash_arrival_time;
            float move_speed = target->get_move_speed();

            math::vector3 cast_position;
            float confidence;
            std::string reasoning;

            // BLINK DETECTION: Very fast "dashes" are actually blinks (Ezreal E, Kassadin R, Flash)
            // Blinks have no travel path to intercept - they teleport instantly
            // Detect by: dash_arrival_time <= 0.1s (default for speed=0) or very high speed
            bool is_blink = edge_cases.dash.dash_arrival_time <= 0.1f ||
                           edge_cases.dash.dash_speed > 3000.f;

            if (time_after_dash < 0.f && !is_blink)
            {
                // Spell arrives MID-DASH - this is a GREAT opportunity!
                // They're locked in forced movement, can't dodge or change direction
                // Calculate intercept position along dash path

                // FIX: Use server position for accurate dash start (client lags behind)
                math::vector3 dash_start = target->get_server_position();
                math::vector3 dash_end = edge_cases.dash.dash_end_position;
                math::vector3 dash_vector = dash_end - dash_start;
                float dash_length = dash_vector.magnitude();

                if (dash_length < 1.f)
                {
                    // Dash too short, use current position
                    cast_position = dash_start;
                    confidence = 0.85f;
                    reasoning = "MID-DASH - Very short dash, high confidence";
                }
                else
                {
                    // ITERATIVE INTERCEPT: Solve for true geometric intercept point
                    // Find progress p where: spell_arrival_time(p) == target_arrival_time(p)
                    // This requires iteration because spell distance depends on intercept position
                    //
                    // OLD BUG: Used ratio spell_travel_time / dash_arrival_time, but these have
                    // different starting points (my pos vs target pos), making ratio meaningless
                    // NEW: Proper iterative solver that converges to true intercept geometry

                    float progress = 0.5f;  // Start with midpoint guess

                    // Iterate to find true intercept (3 iterations sufficient for convergence)
                    for (int iter = 0; iter < 3; ++iter)
                    {
                        // Calculate intercept position at current progress estimate
                        math::vector3 intercept_pos = dash_start + dash_vector * progress;

                        // Time for spell to reach this intercept position (2D - dashes have height arcs)
                        float dist_to_intercept = distance_2d(intercept_pos, source->get_position());
                        float spell_time = spell.delay;
                        if (spell.projectile_speed > 0.f)
                            spell_time += dist_to_intercept / spell.projectile_speed;

                        // Time for target to reach this intercept position
                        float target_time = progress * edge_cases.dash.dash_arrival_time;

                        // Adjust progress based on time difference
                        // If spell arrives late (spell_time > target_time), aim earlier (decrease progress)
                        // If spell arrives early (spell_time < target_time), aim later (increase progress)
                        float time_error = spell_time - target_time;
                        if (edge_cases.dash.dash_arrival_time > 0.f)
                        {
                            float adjustment = -time_error / edge_cases.dash.dash_arrival_time;
                            progress = progress + adjustment * 0.5f;  // Damped for stability
                            progress = std::clamp(progress, 0.1f, 0.95f);  // Stay within dash
                        }
                    }

                    cast_position = dash_start + dash_vector * progress;

                    // HIGH confidence - forced movement is very predictable
                    confidence = 0.9f * edge_cases.dash.confidence_multiplier;

                    int progress_pct = static_cast<int>(progress * 100);
                    reasoning = "MID-DASH INTERCEPT - Hitting at " + std::to_string(progress_pct) +
                        "% through dash (locked in forced movement)";
                }
            }
            else if (time_after_dash < 0.f && is_blink)
            {
                // BLINK: No mid-point exists, aim at endpoint
                // They teleport instantly - aim where they'll appear
                cast_position = edge_cases.dash.dash_end_position;
                confidence = 0.85f * edge_cases.dash.confidence_multiplier;
                reasoning = "BLINK ENDPOINT - Instant teleport, aiming at destination";
            }
            else
            {
                // Spell arrives AFTER dash ends - aim at endpoint
                // Confidence based on how much time they have to move after landing
                cast_position = edge_cases.dash.dash_end_position;

                float potential_dodge_distance = move_speed * time_after_dash;

                // Confidence decreases as potential dodge distance increases
                float dodge_threshold = spell.radius * 2.0f;
                float time_confidence = 1.0f;
                if (potential_dodge_distance > 0.f && dodge_threshold > 0.f)
                {
                    time_confidence = 1.0f - std::min(potential_dodge_distance / dodge_threshold, 1.0f) * 0.5f;
                }

                // Additional penalty for long delay (>reaction time to respond)
                if (time_after_dash > HUMAN_REACTION_TIME)
                {
                    time_confidence *= 0.8f;
                }

                confidence = time_confidence * edge_cases.dash.confidence_multiplier;

                int ms_after = static_cast<int>(time_after_dash * 1000);
                if (time_after_dash < MIN_REACTION_TIME)
                    reasoning = "DASH ENDPOINT - Spell arrives " + std::to_string(ms_after) + "ms after landing (excellent)";
                else if (time_after_dash < HUMAN_REACTION_TIME)
                    reasoning = "DASH ENDPOINT - Spell arrives " + std::to_string(ms_after) + "ms after landing (good)";
                else
                    reasoning = "DASH ENDPOINT - Spell arrives " + std::to_string(ms_after) + "ms after landing (they may dodge)";
            }

            // Check if cast position is in range (2D - High Ground Fix)
            math::vector3 to_cast = cast_position - source->get_position();
            if (magnitude_2d(to_cast) > spell.range)
            {
                result.is_valid = false;
                result.reasoning = "Dash target position out of range";
                return result;
            }

            result.cast_position = cast_position;
            result.hit_chance = confidence;
            result.physics_contribution = 1.0f;
            result.behavior_contribution = confidence;
            result.confidence_score = confidence;
            result.is_valid = true;
            result.reasoning = reasoning;

            update_opportunity_signals(result, source, target, spell, tracker);

            return result;
        }

        // Dispatch to spell-type specific implementation based on pred_sdk type
        // NOTE: Automatic cone detection removed - SDK data has cone_angle for many non-cone spells
        // (Viktor E, Mel Q, Pyke Q, etc.) causing false positives. Use explicit spell_type instead.
        HybridPredictionResult spell_result;

        switch (spell.spell_type)
        {
        case pred_sdk::spell_type::linear:
            spell_result = compute_linear_prediction(source, target, spell, tracker, edge_cases);
            break;

        case pred_sdk::spell_type::circular:
            spell_result = compute_circular_prediction(source, target, spell, tracker, edge_cases);
            break;

        case pred_sdk::spell_type::targetted:
            spell_result = compute_targeted_prediction(source, target, spell, tracker, edge_cases);
            break;

        case pred_sdk::spell_type::vector:
            // Vector spells (Viktor E, Rumble R, Irelia E) - two-position optimization
            spell_result = compute_vector_prediction(source, target, spell, tracker, edge_cases);
            break;

        default:
            // Fallback to circular for unknown types
            spell_result = compute_circular_prediction(source, target, spell, tracker, edge_cases);
            break;
        }

        // Apply edge case adjustments to final result
        if (spell_result.is_valid)
        {
            // Apply confidence multipliers from edge case analysis
            spell_result.confidence_score *= edge_cases.confidence_multiplier;
            spell_result.hit_chance *= edge_cases.confidence_multiplier;

            // Clamp to valid range
            spell_result.confidence_score = std::clamp(spell_result.confidence_score, 0.f, 1.f);
            spell_result.hit_chance = std::clamp(spell_result.hit_chance, 0.f, 1.f);

            // Add edge case info to reasoning
            if (edge_cases.is_slowed)
                spell_result.reasoning += "\n[SLOWED: +15% confidence]";

            if (edge_cases.has_shield)
                spell_result.reasoning += "\n[WARNING: Spell shield active - will be blocked!]";

            if (edge_cases.dash.is_dashing && !math::is_zero(edge_cases.dash.dash_end_position))
                spell_result.reasoning += "\n[DASH PREDICTION: Aiming at dash endpoint]";
        }

        return spell_result;
    }

    HybridPredictionResult HybridFusionEngine::compute_circular_prediction(
        game_object* source,
        game_object* target,
        const pred_sdk::spell_data& spell,
        TargetBehaviorTracker& tracker,
        const EdgeCases::EdgeCaseAnalysis& edge_cases)
    {
        HybridPredictionResult result;

        // FIX: Add dead checks - is_valid() doesn't return false for dead targets
        if (!source || !target || !source->is_valid() || !target->is_valid() ||
            source->is_dead() || target->is_dead())
        {
            result.is_valid = false;
            result.reasoning = "Invalid source or target";
            return result;
        }

        // Step 1: Compute arrival time with iterative intercept refinement
        // CRITICAL FIX: Arrival time must account for target movement during flight
        // Problem: Initial calculation uses current distance, but target moves → changes distance
        // Solution: Iterate to converge on true intercept time (where projectile meets target)
        math::vector3 source_pos = source->get_position();
        math::vector3 target_client_pos = target->get_position();
        // FIX: Use server position for target to avoid latency lag (30-100ms behind)
        math::vector3 target_pos = target->get_server_position();
        float initial_distance = distance_2d(target_pos, source_pos);  // 2D distance (ignore height)

        float arrival_time = PhysicsPredictor::compute_arrival_time(
            source_pos,
            target_pos,
            spell.projectile_speed,
            spell.delay,
            spell.proc_delay  // PROC_DELAY FIX: Include activation delay (e.g., Syndra Q pop time)
        );
        float initial_arrival_time = arrival_time;

        // ANIMATION LOCK DELAY (Stop-then-Go):
        // Calculate remaining lock time for accurate path prediction
        // Target stays at current position during lock, then moves at full speed
        // ADAPTIVE: Use measured cancel delay for this specific player
        float animation_lock_delay = 0.f;
        if (!target->is_moving() && (is_auto_attacking(target) || is_casting_spell(target) || is_channeling(target)))
        {
            // Use adaptive reaction buffer if we have enough samples, otherwise default
            float reaction_buffer = tracker.get_adaptive_reaction_buffer();
            animation_lock_delay = get_remaining_lock_time_adaptive(target, reaction_buffer);
        }

        // Track refinement convergence
        int refinement_iterations = 0;
        bool arrival_converged = false;
        math::vector3 final_predicted_pos = target_pos;

        // Refine arrival time iteratively (converges in 2-3 iterations)
        // PERFORMANCE: Early exit if converged (< 1ms change)
        for (int iteration = 0; iteration < 3; ++iteration)
        {
            refinement_iterations = iteration + 1;
            float prev_arrival = arrival_time;
            math::vector3 predicted_pos = PhysicsPredictor::predict_on_path(target, arrival_time, animation_lock_delay);
            final_predicted_pos = predicted_pos;

            arrival_time = PhysicsPredictor::compute_arrival_time(
                source_pos,
                predicted_pos,
                spell.projectile_speed,
                spell.delay,
                spell.proc_delay
            );

            // Early exit if converged (change < 1ms)
            if (std::abs(arrival_time - prev_arrival) < 0.001f)
            {
                arrival_converged = true;
                break;
            }
        }

        float final_arrival_time = arrival_time;
        float predicted_distance = distance_2d(final_predicted_pos, source_pos);  // 2D

        // Step 2: Build reachable region (physics)
        // FIX: Use path-following prediction for initial center position
        math::vector3 path_predicted_pos = PhysicsPredictor::predict_on_path(target, arrival_time, animation_lock_delay);

        // POINT-BLANK DETECTION: At very close range, different rules apply
        // Use server position for accurate distance (client position lags)
        // HIGH GROUND FIX: 2D distance (ignore height)
        float current_distance = distance_2d(target->get_server_position(), source_pos);
        bool is_point_blank = current_distance < 200.f;  // Within 200 units

        float move_speed = target->get_move_speed();
        float effective_move_speed = get_effective_move_speed(target, arrival_time);

        // USE OBSERVED JUKE MAGNITUDE for reachable region
        const DodgePattern& dodge_pattern = tracker.get_dodge_pattern();
        float observed_magnitude = dodge_pattern.get_juke_magnitude(move_speed);

        // Calculate effective reaction time accounting for FOW and cast delay
        // FOW FIX: If we're hidden, enemies can't see our animation - they get full reaction time
        // VISIBLE: Enemies react DURING cast animation, consuming reaction time
        float effective_reaction_time = get_effective_reaction_time(source, target, spell.delay);

        float max_dodge_time = arrival_time - effective_reaction_time;
        float dodge_time = 0.f;
        if (max_dodge_time > 0.f && effective_move_speed > EPSILON)
        {
            float observed_dodge_time = observed_magnitude / effective_move_speed;
            dodge_time = std::min(observed_dodge_time, max_dodge_time);

            // CLOSE-RANGE FIX: Dramatically reduce dodge time for fast spells
            // At close range (arrival < 0.5s), targets can't execute full jukes
            // They barely finish reacting before spell lands
            // GUARD: Skip if effective_reaction_time >= 0.5s to avoid division by zero
            if (arrival_time < 0.5f && effective_reaction_time < 0.5f - EPSILON)
            {
                // Scale dodge time: 0.3s arrival → 40% dodge, 0.5s → 100% dodge
                float close_range_scale = (arrival_time - effective_reaction_time) / (0.5f - effective_reaction_time);
                close_range_scale = std::clamp(close_range_scale, 0.f, 1.f);
                dodge_time *= close_range_scale;

                // POINT-BLANK OVERRIDE: At <200 units, minimal dodge time for narrow skillshots
                // Narrow skillshots (Pyke Q: 140 width, Thresh Q: 140 width) need pinpoint accuracy
                // At 400 MS: 0.01s = 4 unit radius (vs 0.05s = 20 unit radius)
                float min_dodge = is_point_blank ? 0.01f : 0.05f;
                dodge_time = std::max(dodge_time, min_dodge);
            }
        }

        // Use dynamic acceleration if tracker has measured it
        float dynamic_accel = tracker.has_measured_physics() ?
            tracker.get_measured_acceleration() : DEFAULT_ACCELERATION;

        // SPLIT-PATH APPROACH: Calculate position at reaction time
        // During reaction time, target follows their clicked path (not dodging yet)
        // This handles curved paths correctly (linear velocity extrapolation doesn't)
        math::vector3 pos_at_reaction = PhysicsPredictor::predict_on_path(target, effective_reaction_time, animation_lock_delay);

        // Build reachable region FROM the reaction position
        // Zero velocity because path-following during reaction time is already handled above
        // Only need to calculate dodge possibilities from the reaction point
        ReachableRegion reachable_region = PhysicsPredictor::compute_reachable_region(
            pos_at_reaction,  // Start at reaction position (after path following)
            math::vector3(0, 0, 0),  // Zero velocity (drift already accounted for)
            arrival_time - effective_reaction_time,  // Remaining time to dodge
            effective_move_speed,
            DEFAULT_TURN_RATE,
            dynamic_accel,
            0.0f  // No more reaction drift needed (already at reaction point)
        );

        result.reachable_region = reachable_region;

        // Step 3: Build behavior PDF and align with path prediction
        BehaviorPDF behavior_pdf = BehaviorPredictor::build_pdf_from_history(tracker, arrival_time, move_speed);
        BehaviorPredictor::apply_contextual_factors(behavior_pdf, tracker, target);

        // REACTION-GATE OVERRIDE: If undodgeable, force PDF to pure physics
        // When arrival_time < reaction_time, behavior patterns are irrelevant.
        // They physically cannot execute any dodge - trust linear extrapolation 100%.
        bool is_undodgeable = arrival_time < effective_reaction_time;
        if (is_undodgeable)
        {
            // Clear PDF grid and concentrate all probability at arrival position
            // This removes any lateral/forward variance from behavior prediction
            for (int i = 0; i < BehaviorPDF::GRID_SIZE; ++i)
                for (int j = 0; j < BehaviorPDF::GRID_SIZE; ++j)
                    behavior_pdf.pdf_grid[i][j] = 0.f;
            behavior_pdf.total_probability = 0.f;
        }

        // Center PDF on path prediction at full arrival time
        behavior_pdf.origin = path_predicted_pos;

        // For undodgeable case, add single concentrated sample at origin
        if (is_undodgeable)
        {
            behavior_pdf.add_weighted_sample(path_predicted_pos, 1.0f);
        }
        result.behavior_pdf = behavior_pdf;

        // Step 4: Compute confidence score
        float confidence = compute_confidence_score(source, target, spell, tracker, edge_cases);
        result.confidence_score = confidence;

        // Step 5: Find optimal cast position using GRADIENT ASCENT
        // FIX: Replaced "Disagreement Logic" with rigorous optimization
        // This ensures the cast position is mathematically optimal for the fused probability,
        // and physically contained within the reachable region.
        // It prevents "wild" PDFs from pulling the aim point into impossible territory.
        float effective_radius = spell.radius + target->get_bounding_radius();

        math::vector3 optimal_cast_pos = find_optimal_cast_position(
            reachable_region,
            behavior_pdf,
            source->get_position(),
            effective_radius,
            confidence
        );

        // NAVMESH CLAMPING: Ensure cast position is on pathable terrain
        if (g_sdk && g_sdk->nav_mesh)
        {
            if (!g_sdk->nav_mesh->is_pathable(optimal_cast_pos))
            {
                // Search in a small radius for pathable position
                // PERFORMANCE: Reduced from 150 to 75 units (predictions rarely off-path by >75)
                constexpr float SEARCH_STEP = 25.f;
                constexpr int SEARCH_DIRECTIONS = 8;
                constexpr float MAX_SEARCH_DIST = 75.f;
                float best_distance = FLT_MAX;
                math::vector3 best_pos = optimal_cast_pos;

                for (int i = 0; i < SEARCH_DIRECTIONS; ++i)
                {
                    float angle = (2.f * PI * i) / SEARCH_DIRECTIONS;
                    for (float dist = SEARCH_STEP; dist <= MAX_SEARCH_DIST; dist += SEARCH_STEP)
                    {
                        math::vector3 test_pos = optimal_cast_pos;
                        test_pos.x += std::cos(angle) * dist;
                        test_pos.z += std::sin(angle) * dist;

                        if (g_sdk->nav_mesh->is_pathable(test_pos))
                        {
                            if (dist < best_distance)
                            {
                                best_distance = dist;
                                best_pos = test_pos;
                            }
                            break;
                        }
                    }
                }
                optimal_cast_pos = best_pos;
            }
        }

        // RANGE CLAMPING: Ensure cast position is within spell range (2D - High Ground Fix)
        math::vector3 to_cast = optimal_cast_pos - source_pos;
        float distance_to_cast = magnitude_2d(to_cast);
        if (distance_to_cast > spell.range && distance_to_cast > 0.01f)
        {
            // Clamp to max range
            // FIX: Must use 2D direction for normalization (cannot divide 3D vector by 2D magnitude!)
            math::vector3 to_cast_2d = flatten_2d(to_cast);
            math::vector3 direction_2d = to_cast_2d / distance_to_cast;  // Now both are 2D
            float y_offset = optimal_cast_pos.y - source_pos.y;  // Preserve Y difference
            optimal_cast_pos = source_pos + direction_2d * spell.range;
            optimal_cast_pos.y = source_pos.y + y_offset;  // Restore Y coordinate
        }

        result.cast_position = optimal_cast_pos;

        // Step 6: Evaluate final hit chance
        float physics_prob = PhysicsPredictor::compute_physics_hit_probability(
            optimal_cast_pos,
            effective_radius,
            reachable_region
        );

        float behavior_prob = BehaviorPredictor::compute_behavior_hit_probability(
            optimal_cast_pos,
            effective_radius,
            behavior_pdf
        );

        result.physics_contribution = physics_prob;
        result.behavior_contribution = behavior_prob;

        // Weighted fusion
        size_t sample_count = tracker.get_history().size();
        float current_time = 0.f;
        if (g_sdk && g_sdk->clock_facade)
            current_time = g_sdk->clock_facade->get_game_time();
        float time_since_update = current_time - tracker.get_last_update_time();

        result.hit_chance = fuse_probabilities(
            physics_prob, behavior_prob, confidence,
            sample_count, time_since_update, move_speed, current_distance
        );

        // Debug logging for 0 hit chance analysis
        if (result.hit_chance < 0.01f && g_sdk && PredictionSettings::get().enable_debug_logging)
        {
            char debug_msg[512];
            snprintf(debug_msg, sizeof(debug_msg),
                "[Danny.Prediction] CIRCULAR FAIL: arr=%.2f phys=%.2f behav=%.2f conf=%.2f samp=%zu R=%.0f",
                arrival_time, physics_prob, behavior_prob, confidence, sample_count, reachable_region.max_radius);
            g_sdk->log_console(debug_msg);
        }

        result.hit_chance = std::clamp(result.hit_chance, 0.f, 1.f);

#if HYBRID_PRED_ENABLE_REASONING
        std::ostringstream reasoning;
        reasoning << "Hybrid Prediction Analysis:\n";
        reasoning << "  Arrival Time: " << arrival_time << "s\n";
        reasoning << "  Reachable Radius: " << reachable_region.max_radius << " units\n";
        reasoning << "  Physics Hit Prob: " << (physics_prob * 100.f) << "%\n";
        reasoning << "  Behavior Hit Prob: " << (behavior_prob * 100.f) << "%\n";
        reasoning << "  Confidence: " << (confidence * 100.f) << "%\n";
        reasoning << "  Final HitChance: " << (result.hit_chance * 100.f) << "%\n";
        reasoning << "  Cast Position: (" << optimal_cast_pos.x << ", " << optimal_cast_pos.z << ")\n";
        result.reasoning = reasoning.str();
#else
        result.reasoning = "";
#endif

        result.is_valid = true;

        update_opportunity_signals(result, source, target, spell, tracker);

        // ===================================================================
        // POPULATE DETAILED TELEMETRY DEBUG DATA
        // ===================================================================
        result.telemetry_data.source_pos_x = source_pos.x;
        result.telemetry_data.source_pos_z = source_pos.z;
        result.telemetry_data.target_client_pos_x = target_client_pos.x;
        result.telemetry_data.target_client_pos_z = target_client_pos.z;
        result.telemetry_data.target_server_pos_x = target_pos.x;
        result.telemetry_data.target_server_pos_z = target_pos.z;
        result.telemetry_data.predicted_pos_x = path_predicted_pos.x;
        result.telemetry_data.predicted_pos_z = path_predicted_pos.z;
        result.telemetry_data.cast_pos_x = result.cast_position.x;
        result.telemetry_data.cast_pos_z = result.cast_position.z;

        // Arrival time data
        result.telemetry_data.initial_distance = initial_distance;
        result.telemetry_data.initial_arrival_time = initial_arrival_time;
        result.telemetry_data.refinement_iterations = refinement_iterations;
        result.telemetry_data.final_arrival_time = final_arrival_time;
        result.telemetry_data.arrival_time_change = final_arrival_time - initial_arrival_time;
        result.telemetry_data.arrival_converged = arrival_converged;
        result.telemetry_data.predicted_distance = predicted_distance;

        // Path prediction data
        auto target_path = target->get_path();
        result.telemetry_data.path_segment_count = static_cast<int>(target_path.size());
        result.telemetry_data.path_segment_used = 0;  // Circular doesn't expose this easily
        result.telemetry_data.path_distance_traveled = 0.f;  // Would need to track in predict_on_path
        result.telemetry_data.path_distance_total = 0.f;
        result.telemetry_data.path_segment_progress = 0.f;
        result.telemetry_data.distance_from_path = (target_pos - (target_path.size() > 0 ? target_path[0] : target_pos)).magnitude();

        // Dodge & reachable region data
        result.telemetry_data.dodge_time = dodge_time;
        result.telemetry_data.effective_reaction_time = effective_reaction_time;
        result.telemetry_data.reachable_radius = reachable_region.max_radius;
        result.telemetry_data.reachable_center_x = reachable_region.center.x;
        result.telemetry_data.reachable_center_z = reachable_region.center.z;
        result.telemetry_data.effective_move_speed = effective_move_speed;

        // Outcome tracking (will be filled in later if we track outcomes)
        result.telemetry_data.outcome_recorded = false;
        result.telemetry_data.was_hit = false;
        result.telemetry_data.actual_pos_x = 0.f;
        result.telemetry_data.actual_pos_z = 0.f;
        result.telemetry_data.prediction_error = 0.f;
        result.telemetry_data.time_to_outcome = 0.f;

        return result;
    }

    /**
     * Check if target is in an "obvious hit" state where we should bypass normal confidence penalties
     *
     * Obvious hit states:
     * - CC'd (stunned, rooted, knocked up, etc.)
     * - Channeling (stationary abilities like Malz R, Velkoz R, etc.)
     * - Recalling (100% stationary and predictable)
     * - Walking perfectly straight (stable velocity with no direction changes)
     */
    bool is_obvious_hit(game_object* target, const TargetBehaviorTracker& tracker,
        const EdgeCases::EdgeCaseAnalysis& edge_cases)
    {
        if (!target || !target->is_valid())
            return false;

        // Check animation lock states (CC, casting, auto-attacking)
        if (tracker.is_animation_locked())
            return true;

        // Channeling or recalling = stationary = obvious hit
        if (edge_cases.channel.is_channeling || edge_cases.channel.is_recalling)
            return true;

        // NOTE: Straight-line walking is NOT an obvious hit
        // Targets can still react to our cast animation and dodge
        // Let the normal physics/behavior predictions handle this case
        // which will naturally give high probability for straight-line movement

        return false;
    }

    float HybridFusionEngine::compute_confidence_score(
        game_object* source,
        game_object* target,
        const pred_sdk::spell_data& spell,
        const TargetBehaviorTracker& tracker,
        const EdgeCases::EdgeCaseAnalysis& edge_cases)
    {
        // OBVIOUS HIT OVERRIDE: Bypass all penalties for clearly hittable targets
        // This prevents confidence from strangling predictions on CC'd/channeling/straight-line targets
        if (is_obvious_hit(target, tracker, edge_cases))
        {
            return 0.95f;  // Near-perfect confidence for obvious hits
        }

        // NOTE: We intentionally do NOT boost confidence based on expected_hit_types here.
        // The is_obvious_hit() check above already handles CC'd/locked targets.
        // If spell expects undodgeable/cc but target ISN'T in that state, we should use
        // normal confidence calculation - boosting would cause us to fire at dodgeable targets.

        float confidence = 1.0f;

        // Distance factor - further = less confident
        // HIGH GROUND FIX: 2D distance (ignore height)
        // FIX: Use server position for accurate distance (client lags behind)
        float distance = distance_2d(target->get_server_position(), source->get_position());
        confidence *= std::exp(-distance * CONFIDENCE_DISTANCE_DECAY);

        // Latency factor (ping in seconds)
        // CRASH FIX: Check g_sdk AND net_client before accessing
        float ping = 0.f;
        if (g_sdk && g_sdk->net_client)
            ping = static_cast<float>(g_sdk->net_client->get_ping()) * 0.001f;
        confidence *= std::exp(-ping * CONFIDENCE_LATENCY_FACTOR);

        // Spell-specific adjustments
        if (spell.projectile_speed >= FLT_MAX / 2.f)
        {
            // Instant spell - higher confidence (less time for target to react)
            confidence *= 1.2f;
        }
        else if (spell.projectile_speed < 1000.f)
        {
            // Slow projectile - lower confidence (more time to dodge)
            confidence *= 0.9f;
        }

        // Mobility factor - high mobility champions are harder to predict
        float move_speed = target->get_move_speed();
        float mobility_penalty = std::clamp(move_speed / 500.f, 0.5f, 1.5f);

        // Safe to divide - clamp guarantees [0.5, 1.5] range (never zero)
        confidence /= mobility_penalty;

        // Sample size factor - more data = more confidence
        // FAST-TRACK: If target made a recent sharp turn (>45°), bypass most of the penalty
        // Sharp turns give us high-quality data about their behavior even with few samples
        const auto& history = tracker.get_history();
        if (history.size() < MIN_SAMPLES_FOR_BEHAVIOR)
        {
            if (tracker.has_recent_hard_juke())
            {
                // Sharp turn detected - minimal penalty (95% confidence with early data)
                // We know they're actively dodging, which is valuable information
                confidence *= 0.95f;
            }
            else
            {
                // Standard ramp-up: need more samples to trust behavior
                confidence *= static_cast<float>(history.size()) / MIN_SAMPLES_FOR_BEHAVIOR;
            }
        }

        // Animation lock boost - target is locked in animation
        // Multiplicative (not additive) to preserve relative confidence levels
        if (tracker.is_animation_locked())
        {
            confidence *= (1.0f + ANIMATION_LOCK_CONFIDENCE_BOOST);  // 1.3x multiplier
        }

        // Fog of war advantage - if we're hidden from enemy, they can't react to cast animation
        // They only see the spell once it's in flight, reducing reaction time
        // Check if our position is in fog of war for the ENEMY team
        if (g_sdk && g_sdk->nav_mesh)
        {
            int enemy_team = target->get_team_id();
            math::vector3 source_pos = source->get_position();

            // Check if we're in fog of war for the enemy team
            // This covers: bush (unwarded), behind walls, any fog condition
            if (g_sdk->nav_mesh->is_in_fow_for_team(source_pos, enemy_team))
            {
                // Enemy can't see us - significant confidence boost
                // They lose ~0.25s of reaction time (cast animation)
                confidence *= 1.25f;
            }
        }

        // AVERAGE TURN ANGLE - Intelligent juke analysis
        // Now context-aware: distinguishes predictable patterns from random dancing
        float avg_turn_angle = tracker.get_average_turn_angle();
        const auto& dodge_pattern = tracker.get_dodge_pattern();

        // PATTERN-AWARE CONFIDENCE: Don't penalize patterned movement
        if (avg_turn_angle < 15.f)
        {
            // Running nearly straight - boost confidence
            // < 15° = very predictable movement (kiting, fleeing, chasing)
            confidence *= 1.12f;
        }
        else if (avg_turn_angle < 30.f)
        {
            // Slightly curved but still predictable
            confidence *= 1.05f;
        }
        else if (avg_turn_angle > 60.f)
        {
            // HIGH turn angle - but check if it's PATTERNED or RANDOM
            if (dodge_pattern.has_pattern && dodge_pattern.pattern_confidence > 0.65f)
            {
                // HIGH variance BUT we detected a strong pattern (alternating, repeating, etc.)
                // This is PREDICTABLE juking (rhythmic kiter, predictable dodger)
                // BOOST confidence instead of penalizing!
                confidence *= 1.08f;  // 8% boost for predictable pattern
            }
            else
            {
                // HIGH variance AND no pattern = truly random dancing
                // This is unpredictable - HEAVY penalty
                confidence *= 0.75f;  // 25% penalty (increased from 15%)
            }
        }
        else if (avg_turn_angle > 45.f)
        {
            // MODERATE turn angle - check for patterns
            if (dodge_pattern.has_pattern && dodge_pattern.pattern_confidence > 0.60f)
            {
                // Moderate variance but patterned = still predictable
                // Neutral or slight boost
                confidence *= 1.02f;  // 2% boost
            }
            else
            {
                // Moderate variance, no pattern = somewhat unpredictable
                confidence *= 0.90f;  // 10% penalty (slightly reduced from 8%)
            }
        }
        // 30-45° range = neutral, no modifier

        // PATH STALENESS PENALTY
        // Old paths become unreliable - target may have changed their mind or arrived at destination
        // Fresh clicks (< 0.5s) are high confidence, stale paths (> 2s) need significant penalties
        float current_time = (g_sdk && g_sdk->clock_facade)
            ? g_sdk->clock_facade->get_game_time() : 0.f;
        float path_age = current_time - tracker.get_last_path_update_time();

        if (tracker.get_last_path_size() > 0)  // Has an active path
        {
            if (path_age > 2.0f)
            {
                // Path is very stale (> 2s) - significant penalty
                // They may have changed their mind, arrived, or are about to re-path
                confidence *= 0.70f;  // 30% penalty
            }
            else if (path_age > 1.0f)
            {
                // Path is somewhat stale (1-2s) - moderate penalty
                confidence *= 0.85f;  // 15% penalty
            }
            else if (path_age > 0.5f)
            {
                // Path is slightly stale (0.5-1s) - minor penalty
                confidence *= 0.95f;  // 5% penalty
            }
            // Path < 0.5s = fresh, no penalty (full confidence)

            // PATH COMPLETION BOOST
            // If target is approaching destination (final waypoint, < 100 units away), they're more predictable
            // They're committed to the path and unlikely to change direction
            if (tracker.is_approaching_destination(target))
            {
                confidence *= 1.15f;  // 15% boost for path completion
            }

            // LONG PATH PENALTY
            // Long paths (> 5 waypoints) give more opportunity to change direction or re-path
            size_t path_size = tracker.get_last_path_size();
            if (path_size > 5)
            {
                // Very long path - reduce confidence
                float path_penalty = std::min((path_size - 5) * 0.05f, 0.25f);
                confidence *= (1.0f - path_penalty);  // Up to 25% penalty for 10+ waypoints
            }
        }

        return std::clamp(confidence, 0.1f, 1.0f);
    }

    math::vector3 HybridFusionEngine::find_optimal_cast_position(
        const ReachableRegion& reachable_region,
        const BehaviorPDF& behavior_pdf,
        const math::vector3& source_pos,
        float projectile_radius,
        float confidence)
    {
        // Grid search over reachable region
        // FIX: Use configurable resolution instead of hardcoded value
        // Default: 8x8 = 64 samples (balanced), User can increase to 16x16 = 256 (high quality)
        int grid_size = PredictionSettings::get().grid_search_resolution;
        float best_score = -1.f;
        math::vector3 best_position = reachable_region.center;

        float search_radius = reachable_region.max_radius;
        float step = search_radius * 2.f / grid_size;

        for (int i = 0; i < grid_size; ++i)
        {
            for (int j = 0; j < grid_size; ++j)
            {
                math::vector3 test_pos = reachable_region.center;
                test_pos.x += (i - grid_size / 2) * step;
                test_pos.z += (j - grid_size / 2) * step;

                float score = evaluate_hit_chance_at_point(
                    test_pos,
                    reachable_region,
                    behavior_pdf,
                    projectile_radius,
                    confidence
                );

                if (score > best_score)
                {
                    best_score = score;
                    best_position = test_pos;
                }
            }
        }

        // Gradient ascent refinement (2 iterations)
        for (int iter = 0; iter < 2; ++iter)
        {
            constexpr float GRADIENT_STEP = 10.f;
            constexpr int GRADIENT_SAMPLES = 8;

            math::vector3 gradient{};

            for (int i = 0; i < GRADIENT_SAMPLES; ++i)
            {
                float angle = (2.f * PI * i) / GRADIENT_SAMPLES;
                math::vector3 test_pos = best_position;
                test_pos.x += GRADIENT_STEP * std::cos(angle);
                test_pos.z += GRADIENT_STEP * std::sin(angle);

                float score = evaluate_hit_chance_at_point(
                    test_pos,
                    reachable_region,
                    behavior_pdf,
                    projectile_radius,
                    confidence
                );

                float angle_weight = score - best_score;
                gradient.x += angle_weight * std::cos(angle);
                gradient.z += angle_weight * std::sin(angle);
            }

            if (gradient.magnitude() > EPSILON)
            {
                best_position = best_position + gradient.normalized() * GRADIENT_STEP * 0.5f;
                best_score = evaluate_hit_chance_at_point(
                    best_position,
                    reachable_region,
                    behavior_pdf,
                    projectile_radius,
                    confidence
                );
            }
        }

        return best_position;
    }

    math::vector3 HybridFusionEngine::find_multi_target_aoe_position(
        game_object* source,
        const std::vector<game_object*>& targets,
        const pred_sdk::spell_data& spell,
        float max_range)
    {
        if (!source || targets.empty())
        {
            return math::vector3{};
        }

        math::vector3 source_pos = source->get_position();
        float spell_radius = spell.radius;

        // Collect valid target positions (server positions for accuracy)
        std::vector<math::vector3> target_positions;
        target_positions.reserve(targets.size());

        for (game_object* target : targets)
        {
            if (!target || !target->is_valid() || target->is_dead())
                continue;

            // Use server position for accuracy
            math::vector3 target_pos = target->get_server_position();
            float distance = distance_2d(target_pos, source_pos);  // 2D

            // Only consider targets within range
            if (distance <= max_range)
            {
                // Flatten to 2D for MEC calculation (High Ground Fix - prevent floating center)
                target_positions.push_back(flatten_2d(target_pos));
            }
        }

        // Need at least 1 target
        if (target_positions.empty())
        {
            return math::vector3{};
        }

        // For single target, just return their position
        if (target_positions.size() == 1)
        {
            return target_positions[0];
        }

        // Use MEC algorithm to find optimal AOE center (inputs already flattened to 2D)
        Circle mec = PhysicsPredictor::compute_minimum_enclosing_circle(target_positions);

        // Verify the MEC center is within cast range (2D distance)
        float center_distance = distance_2d(mec.center, source_pos);

        if (center_distance <= max_range)
        {
            // Check if all targets are within the spell radius (2D - High Ground Fix)
            bool all_hit = true;
            for (const auto& pos : target_positions)
            {
                float dist_from_center = distance_2d(pos, mec.center);
                if (dist_from_center > spell_radius)
                {
                    all_hit = false;
                    break;
                }
            }

            // If MEC works perfectly, use it
            if (all_hit)
            {
                return mec.center;
            }
        }

        // Fallback: If MEC center is out of range or doesn't hit all targets,
        // find the position that maximizes hit count within range
        math::vector3 best_position = target_positions[0];
        int best_hit_count = 0;

        for (const auto& candidate_pos : target_positions)
        {
            // Count how many targets would be hit if we cast at this position (2D - High Ground Fix)
            int hit_count = 0;
            for (const auto& target_pos : target_positions)
            {
                float dist = distance_2d(target_pos, candidate_pos);
                if (dist <= spell_radius)
                {
                    hit_count++;
                }
            }

            if (hit_count > best_hit_count)
            {
                best_hit_count = hit_count;
                best_position = candidate_pos;
            }
        }

        return best_position;
    }

    float HybridFusionEngine::evaluate_hit_chance_at_point(
        const math::vector3& point,
        const ReachableRegion& reachable_region,
        const BehaviorPDF& behavior_pdf,
        float projectile_radius,
        float confidence)
    {
        float physics_prob = PhysicsPredictor::compute_physics_hit_probability(
            point,
            projectile_radius,
            reachable_region
        );

        float behavior_prob = BehaviorPredictor::compute_behavior_hit_probability(
            point,
            projectile_radius,
            behavior_pdf
        );

        // FIX: Use weighted fusion instead of multiplication to prevent behavior dominance
        // Old: physics_prob * behavior_prob * confidence (no physics cap!)
        // New: fuse_probabilities with 80% physics cap to prevent wide misses
        // When behavior is strongly biased to a side but wrong, physics cap prevents
        // aiming too far off-center
        float distance = (point - reachable_region.center).magnitude();
        return fuse_probabilities(
            physics_prob,
            behavior_prob,
            confidence,
            32,  // Assume reasonable sample count for evaluation
            0.f, // No staleness in optimization
            350.f, // Average move speed
            distance  // Distance from center (for close-range boost)
        );
    }

    // =========================================================================
    // SPELL-TYPE SPECIFIC IMPLEMENTATIONS
    // =========================================================================

    HybridPredictionResult HybridFusionEngine::compute_linear_prediction(
        game_object* source,
        game_object* target,
        const pred_sdk::spell_data& spell,
        TargetBehaviorTracker& tracker,
        const EdgeCases::EdgeCaseAnalysis& edge_cases)
    {
        HybridPredictionResult result;

        // FIX: Add dead checks - is_valid() doesn't return false for dead targets
        if (!source || !target || !source->is_valid() || !target->is_valid() ||
            source->is_dead() || target->is_dead())
        {
            result.is_valid = false;
            return result;
        }

        // Validate SDK is initialized
        if (!g_sdk || !g_sdk->clock_facade)
        {
            result.is_valid = false;
            result.reasoning = "SDK not initialized";
            return result;
        }

        // Step 1: Compute arrival time with iterative intercept refinement
        // CRITICAL FIX: Arrival time must account for target movement during flight
        math::vector3 source_pos = source->get_position();
        math::vector3 target_client_pos = target->get_position();
        // FIX: Use server position for target to avoid latency lag (30-100ms behind)
        math::vector3 target_pos = target->get_server_position();
        float initial_distance = distance_2d(target_pos, source_pos);  // 2D distance (ignore height)

        float arrival_time = PhysicsPredictor::compute_arrival_time(
            source_pos,
            target_pos,
            spell.projectile_speed,
            spell.delay,
            spell.proc_delay  // PROC_DELAY FIX: Include activation delay (e.g., Syndra Q pop time)
        );
        float initial_arrival_time = arrival_time;

        // ANIMATION LOCK DELAY (Stop-then-Go):
        // Calculate remaining lock time for accurate path prediction
        // Target stays at current position during lock, then moves at full speed
        // ADAPTIVE: Use measured cancel delay for this specific player
        float animation_lock_delay = 0.f;
        if (!target->is_moving() && (is_auto_attacking(target) || is_casting_spell(target) || is_channeling(target)))
        {
            // Use adaptive reaction buffer if we have enough samples, otherwise default
            float reaction_buffer = tracker.get_adaptive_reaction_buffer();
            animation_lock_delay = get_remaining_lock_time_adaptive(target, reaction_buffer);
        }

        // Track refinement convergence
        int refinement_iterations = 0;
        bool arrival_converged = false;
        math::vector3 final_predicted_pos = target_pos;

        // Refine arrival time iteratively (converges in 2-3 iterations)
        // PERFORMANCE: Early exit if converged (< 1ms change)
        for (int iteration = 0; iteration < 3; ++iteration)
        {
            refinement_iterations = iteration + 1;
            float prev_arrival = arrival_time;
            math::vector3 predicted_pos = PhysicsPredictor::predict_on_path(target, arrival_time, animation_lock_delay);
            final_predicted_pos = predicted_pos;

            arrival_time = PhysicsPredictor::compute_arrival_time(
                source_pos,
                predicted_pos,
                spell.projectile_speed,
                spell.delay,
                spell.proc_delay
            );

            // Early exit if converged (change < 1ms)
            if (std::abs(arrival_time - prev_arrival) < 0.001f)
            {
                arrival_converged = true;
                break;
            }
        }

        float final_arrival_time = arrival_time;
        float predicted_distance = distance_2d(final_predicted_pos, source_pos);  // 2D

        // Step 2: Build reachable region (physics)
        // FIX: Use path-following prediction for better accuracy
        math::vector3 path_predicted_pos = PhysicsPredictor::predict_on_path(target, arrival_time, animation_lock_delay);

        // POINT-BLANK DETECTION: At very close range, different rules apply
        // Use server position for accurate distance (client position lags)
        // HIGH GROUND FIX: 2D distance (ignore height)
        float current_distance = distance_2d(target->get_server_position(), source_pos);
        bool is_point_blank = current_distance < 200.f;  // Within 200 units

        math::vector3 target_velocity = tracker.get_current_velocity();
        float move_speed = target->get_move_speed();  // Stat value for historical lookups
        float effective_move_speed = get_effective_move_speed(target, arrival_time);  // Scaled by free time

        // USE OBSERVED JUKE MAGNITUDE for reachable region
        const DodgePattern& dodge_pattern = tracker.get_dodge_pattern();
        float observed_magnitude = dodge_pattern.get_juke_magnitude(move_speed);

        // Calculate effective reaction time accounting for FOW and cast delay
        // FOW FIX: If we're hidden, enemies can't see our animation - they get full reaction time
        // VISIBLE: Enemies react DURING cast animation, consuming reaction time
        float effective_reaction_time = get_effective_reaction_time(source, target, spell.delay);

        float max_dodge_time = arrival_time - effective_reaction_time;
        float dodge_time = 0.f;
        if (max_dodge_time > 0.f && effective_move_speed > EPSILON)
        {
            float observed_dodge_time = observed_magnitude / effective_move_speed;
            dodge_time = std::min(observed_dodge_time, max_dodge_time);

            // CLOSE-RANGE FIX: Dramatically reduce dodge time for fast spells
            // At close range (arrival < 0.5s), targets can't execute full jukes
            // They barely finish reacting before spell lands
            // GUARD: Skip if effective_reaction_time >= 0.5s to avoid division by zero
            if (arrival_time < 0.5f && effective_reaction_time < 0.5f - EPSILON)
            {
                // Scale dodge time: 0.3s arrival → 40% dodge, 0.5s → 100% dodge
                float close_range_scale = (arrival_time - effective_reaction_time) / (0.5f - effective_reaction_time);
                close_range_scale = std::clamp(close_range_scale, 0.f, 1.f);
                dodge_time *= close_range_scale;

                // POINT-BLANK OVERRIDE: At <200 units, minimal dodge time for narrow skillshots
                // Narrow skillshots (Pyke Q: 140 width, Thresh Q: 140 width) need pinpoint accuracy
                // At 400 MS: 0.01s = 4 unit radius (vs 0.05s = 20 unit radius)
                float min_dodge = is_point_blank ? 0.01f : 0.05f;
                dodge_time = std::max(dodge_time, min_dodge);
            }
        }

        // Use dynamic acceleration if tracker has measured it
        float dynamic_accel = tracker.has_measured_physics() ?
            tracker.get_measured_acceleration() : DEFAULT_ACCELERATION;

        // SPLIT-PATH APPROACH: Calculate position at reaction time
        // During reaction time, target follows their clicked path (not dodging yet)
        // This handles curved paths correctly (linear velocity extrapolation doesn't)
        math::vector3 pos_at_reaction = PhysicsPredictor::predict_on_path(target, effective_reaction_time, animation_lock_delay);

        // Build reachable region FROM the reaction position
        // Zero velocity because path-following during reaction time is already handled above
        // Only need to calculate dodge possibilities from the reaction point
        ReachableRegion reachable_region = PhysicsPredictor::compute_reachable_region(
            pos_at_reaction,  // Start at reaction position (after path following)
            math::vector3(0, 0, 0),  // Zero velocity (drift already accounted for)
            arrival_time - effective_reaction_time,  // Remaining time to dodge
            effective_move_speed,
            DEFAULT_TURN_RATE,
            dynamic_accel,
            0.0f  // No more reaction drift needed (already at reaction point)
        );

        result.reachable_region = reachable_region;

        // Step 3: Build behavior PDF
        BehaviorPDF behavior_pdf = BehaviorPredictor::build_pdf_from_history(tracker, arrival_time, move_speed);
        BehaviorPredictor::apply_contextual_factors(behavior_pdf, tracker, target);

        // REACTION-GATE OVERRIDE: If undodgeable, force PDF to pure physics
        bool is_undodgeable = arrival_time < effective_reaction_time;
        if (is_undodgeable)
        {
            for (int i = 0; i < BehaviorPDF::GRID_SIZE; ++i)
                for (int j = 0; j < BehaviorPDF::GRID_SIZE; ++j)
                    behavior_pdf.pdf_grid[i][j] = 0.f;
            behavior_pdf.total_probability = 0.f;
        }
        behavior_pdf.origin = path_predicted_pos;
        if (is_undodgeable)
            behavior_pdf.add_weighted_sample(path_predicted_pos, 1.0f);

        result.behavior_pdf = behavior_pdf;

        // Step 4: Compute confidence score
        float confidence = compute_confidence_score(source, target, spell, tracker, edge_cases);
        result.confidence_score = confidence;

        // Compute staleness for fusion
        float current_time = 0.f;
        if (g_sdk && g_sdk->clock_facade)
            current_time = g_sdk->clock_facade->get_game_time();
        float time_since_update = current_time - tracker.get_last_update_time();
        size_t sample_count = tracker.get_history().size();

        // Note: Linear spells use angular optimization (±10°) which already handles
        // lateral variance through the PDF evaluation. No fusion needed here -
        // it would double-compensate and overshoot jukes.

        // Step 5: Compute capsule parameters
        // Linear spell = capsule from source toward target
        // Use server position for accurate aim direction (client position lags)
        math::vector3 to_target = target->get_server_position() - source->get_position();
        float dist_to_target = to_target.magnitude();

        constexpr float MIN_SAFE_DISTANCE = 1.0f;  // Minimum safe distance for normalization
        if (dist_to_target < MIN_SAFE_DISTANCE)
        {
            result.is_valid = false;
            result.reasoning = "Target too close - zero distance";
            return result;
        }

        math::vector3 direction = to_target / dist_to_target;  // Safe manual normalize
        math::vector3 capsule_start = source->get_position();
        float capsule_length = spell.range;

        // Use exact bounding radius - trust the geometry
        // Shrinking by 10% would reduce hit area by ~19%, causing misses on valid edge hits
        float capsule_radius = spell.radius + target->get_bounding_radius();

        // For linear spells, find optimal direction using angular search
        // Test multiple angles around the predicted center to maximize hit probability
        math::vector3 to_center = reachable_region.center - source->get_position();
        float dist_to_center = to_center.magnitude();

        math::vector3 base_direction;
        if (dist_to_center > MIN_SAFE_DISTANCE)
        {
            base_direction = to_center / dist_to_center;  // Safe manual normalize
        }
        else
        {
            base_direction = direction;  // Fallback to target direction
        }

        // ADAPTIVE ANGULAR SEARCH: Width scales with target's juke behavior
        // Runners (low juke) -> ±10° precision
        // Jukers (high juke) -> ±30° coverage
        const auto& pattern = tracker.get_dodge_pattern();

        // Juke factor = sum of side dodge frequencies (0.0 to 1.0)
        float juke_factor = pattern.left_dodge_frequency + pattern.right_dodge_frequency;

        // Add variance penalty: erratic timing increases search width
        if (pattern.juke_interval_variance > 0.1f)
            juke_factor += 0.2f;

        juke_factor = std::clamp(juke_factor, 0.f, 1.f);

        // Base 10° + up to 20° extra based on behavior (10° to 30° range)
        float deviation_degrees = 10.f + (juke_factor * 20.f);
        float max_angle_deviation = deviation_degrees * (PI / 180.f);

        // Scale sample count with width to maintain angular density
        // 10° -> 7 samples, 30° -> 15 samples
        // At 15 samples over 60°, step size ≈ 4.3°, gap at 1000 range ≈ 75 units
        // This keeps gaps smaller than typical hitbox diameter (130 units)
        int num_angle_tests = 7 + static_cast<int>(juke_factor * 8.f);

        // Ensure odd number so there's always a center ray at angle 0
        // Even numbers would straddle the center, missing the direct line to target
        if (num_angle_tests % 2 == 0)
            num_angle_tests++;

        float best_hit_chance = 0.f;
        math::vector3 optimal_direction = base_direction;
        float best_physics_prob = 0.f;
        float best_behavior_prob = 0.f;

        for (int i = 0; i < num_angle_tests; ++i)
        {
            // Calculate angle offset from -max_angle_deviation to +max_angle_deviation
            float angle_offset = (i - num_angle_tests / 2) * (2.f * max_angle_deviation / (num_angle_tests - 1));

            // Rotate base_direction around Y axis by angle_offset
            // Using 2D rotation in XZ plane (Y is up in League)
            float cos_angle = std::cos(angle_offset);
            float sin_angle = std::sin(angle_offset);
            math::vector3 test_direction(
                base_direction.x * cos_angle - base_direction.z * sin_angle,
                base_direction.y,
                base_direction.x * sin_angle + base_direction.z * cos_angle
            );

            // Compute hit probability for this angle using time-to-dodge method
            // For linear spells, check if predicted position is within capsule
            // Project predicted position onto spell line to find closest point
            math::vector3 to_predicted = reachable_region.center - capsule_start;

            // Use full 3D dot product for correct projection onto spell line
            // 2D projection causes Y errors that scale with distance
            float projection = to_predicted.x * test_direction.x +
                              to_predicted.y * test_direction.y +
                              to_predicted.z * test_direction.z;

            // Clamp projection to spell range [0, capsule_length]
            projection = std::max(0.f, std::min(projection, capsule_length));
            math::vector3 closest_point = capsule_start + test_direction * projection;

            // Calculate perpendicular distance to spell line
            float perp_dist = (reachable_region.center - closest_point).magnitude();

            // Debug: Log for center angle
            if (i == num_angle_tests / 2 && PredictionSettings::get().enable_debug_logging && g_sdk)
            {
                char dbg[512];
                snprintf(dbg, sizeof(dbg),
                    "[Danny.Prediction] LINEAR PROJ: proj=%.1f dist_center=%.1f perp=%.1f radius=%.1f",
                    projection, dist_to_center, perp_dist, capsule_radius);
                g_sdk->log_console(dbg);
            }

            float test_physics_prob = PhysicsPredictor::compute_time_to_dodge_probability(
                reachable_region.center,  // Predicted target position
                closest_point,            // Closest point on spell line
                capsule_radius,           // Spell hitbox radius
                move_speed,               // Target move speed
                arrival_time,             // Time until spell arrives
                effective_reaction_time   // Adjusted for cast animation visibility
            );

            // FIX: Boost physics for stationary targets (with duration check)
            // Physics assumes targets WILL dodge if they CAN, penalizing stationary targets at range
            // Only boost if target has been stationary for a meaningful duration (not just a brief pause)
            float current_speed = target_velocity.magnitude();
            if (current_speed < 50.f)
            {
                // Check how long they've been stationary by examining movement history
                const auto& history = tracker.get_history();
                int stationary_samples = 0;
                constexpr float STATIONARY_THRESHOLD = 50.f;

                // Count consecutive stationary samples from most recent
                for (auto it = history.rbegin(); it != history.rend(); ++it)
                {
                    if (it->velocity.magnitude() < STATIONARY_THRESHOLD)
                        stationary_samples++;
                    else
                        break;  // Movement detected, stop counting
                }

                // Require at least 0.3s of stationary (6 samples at 50ms rate)
                // Scale boost based on duration: 0.3s = 70%, 0.5s = 80%, 1.0s+ = 85%
                constexpr int MIN_SAMPLES_FOR_BOOST = 6;   // ~0.3s
                constexpr int MED_SAMPLES_FOR_BOOST = 10;  // ~0.5s
                constexpr int MAX_SAMPLES_FOR_BOOST = 20;  // ~1.0s

                if (stationary_samples >= MAX_SAMPLES_FOR_BOOST)
                {
                    // Very confident they're AFK or not reacting
                    test_physics_prob = std::max(test_physics_prob, 0.85f);
                }
                else if (stationary_samples >= MED_SAMPLES_FOR_BOOST)
                {
                    // Confident they're stationary
                    test_physics_prob = std::max(test_physics_prob, 0.80f);
                }
                else if (stationary_samples >= MIN_SAMPLES_FOR_BOOST)
                {
                    // Likely stationary but could be brief pause
                    test_physics_prob = std::max(test_physics_prob, 0.70f);
                }
                // If < MIN_SAMPLES, don't boost - they might be about to move

                // PATH VALIDATION: Don't apply full stationary boost if they have a fresh path
                // Target with recent path click is about to start moving, not truly stationary
                auto path = target->get_path();
                if (path.size() > 1)
                {
                    float current_time = (g_sdk && g_sdk->clock_facade)
                        ? g_sdk->clock_facade->get_game_time() : 0.f;
                    float path_age = current_time - tracker.get_last_path_update_time();

                    if (path_age < 0.5f)  // Fresh path within last 0.5s
                    {
                        // They have an active path - they're about to move, not stationary
                        // Cap physics boost to moderate level (don't assume they're AFK)
                        test_physics_prob = std::min(test_physics_prob, 0.60f);
                    }
                }
            }
            else if (current_speed < 100.f)
            {
                // Very slow movement: modest boost floor to 40%
                test_physics_prob = std::max(test_physics_prob, 0.4f);
            }

            float test_behavior_prob = compute_capsule_behavior_probability(
                capsule_start,
                test_direction,
                capsule_length,
                capsule_radius,
                behavior_pdf
            );

            // Fuse probabilities to get overall hit chance
            float test_hit_chance = fuse_probabilities(
                test_physics_prob,
                test_behavior_prob,
                confidence,
                sample_count,
                time_since_update,
                move_speed,
                current_distance  // Pass distance for close-range physics boost
            );

            // Track best configuration
            if (test_hit_chance > best_hit_chance)
            {
                best_hit_chance = test_hit_chance;
                optimal_direction = test_direction;
                best_physics_prob = test_physics_prob;
                best_behavior_prob = test_behavior_prob;

                // Debug: Log when we update best
                if (PredictionSettings::get().enable_debug_logging && g_sdk)
                {
                    char dbg[256];
                    snprintf(dbg, sizeof(dbg),
                        "[Danny.Prediction] BEST UPDATE: angle=%d phys=%.3f behav=%.3f hit=%.3f",
                        i, test_physics_prob, test_behavior_prob, test_hit_chance);
                    g_sdk->log_console(dbg);
                }
            }
        }

        // LINEAR AIM FIX: Aim at predicted center position, not along angled direction
        // The angular optimization finds the best ANGLE for hit detection, but we should
        // still aim at the CENTER of where they'll be (reachable_region.center).
        // Aiming along optimal_direction can cause 500+ unit lateral misses at range!
        //
        // OLD BUG: result.cast_position = capsule_start + optimal_direction * project_distance
        // This aimed along an angled line, causing massive wide misses when behavior was wrong
        //
        // NEW: Aim directly at the predicted center position
        result.cast_position = reachable_region.center;

        // Use best probabilities found
        float physics_prob = best_physics_prob;
        float behavior_prob = best_behavior_prob;

        result.physics_contribution = physics_prob;
        result.behavior_contribution = behavior_prob;

        // Use best hit chance from angular optimization
        result.hit_chance = std::clamp(best_hit_chance, 0.f, 1.f);

        // Debug logging for 0 hit chance
        if (result.hit_chance < 0.01f && g_sdk)
        {
            char debug_msg[512];
            snprintf(debug_msg, sizeof(debug_msg),
                "[Danny.Prediction] LINEAR DEBUG: arrival=%.3f phys=%.3f behav=%.3f conf=%.3f samples=%zu move_speed=%.1f",
                arrival_time, physics_prob, behavior_prob, confidence, sample_count, move_speed);
            g_sdk->log_console(debug_msg);
        }

#if HYBRID_PRED_ENABLE_REASONING
        // Generate mathematical reasoning
        std::ostringstream reasoning;
        reasoning << "Hybrid Prediction Analysis (LINEAR):\n";
        reasoning << "  Arrival Time: " << arrival_time << "s\n";
        reasoning << "  Reachable Radius: " << reachable_region.max_radius << " units\n";
        reasoning << "  Capsule Length: " << capsule_length << " units\n";
        reasoning << "  Capsule Width: " << (capsule_radius * 2.f) << " units\n";
        reasoning << "  Physics Hit Prob: " << (physics_prob * 100.f) << "%\n";
        reasoning << "  Behavior Hit Prob: " << (behavior_prob * 100.f) << "%\n";
        reasoning << "  Confidence: " << (confidence * 100.f) << "%\n";
        reasoning << "  Final HitChance: " << (result.hit_chance * 100.f) << "%\n";
        result.reasoning = reasoning.str();
#else
        result.reasoning = "";
#endif

        result.is_valid = true;

        // Update opportunistic casting signals
        update_opportunity_signals(result, source, target, spell, tracker);

        // ===================================================================
        // POPULATE DETAILED TELEMETRY DEBUG DATA
        // ===================================================================
        result.telemetry_data.source_pos_x = source_pos.x;
        result.telemetry_data.source_pos_z = source_pos.z;
        result.telemetry_data.target_client_pos_x = target_client_pos.x;
        result.telemetry_data.target_client_pos_z = target_client_pos.z;
        result.telemetry_data.target_server_pos_x = target_pos.x;
        result.telemetry_data.target_server_pos_z = target_pos.z;
        result.telemetry_data.predicted_pos_x = path_predicted_pos.x;
        result.telemetry_data.predicted_pos_z = path_predicted_pos.z;
        result.telemetry_data.cast_pos_x = result.cast_position.x;
        result.telemetry_data.cast_pos_z = result.cast_position.z;

        // Arrival time data
        result.telemetry_data.initial_distance = initial_distance;
        result.telemetry_data.initial_arrival_time = initial_arrival_time;
        result.telemetry_data.refinement_iterations = refinement_iterations;
        result.telemetry_data.final_arrival_time = final_arrival_time;
        result.telemetry_data.arrival_time_change = final_arrival_time - initial_arrival_time;
        result.telemetry_data.arrival_converged = arrival_converged;
        result.telemetry_data.predicted_distance = predicted_distance;

        // Path prediction data
        auto target_path = target->get_path();
        result.telemetry_data.path_segment_count = static_cast<int>(target_path.size());
        result.telemetry_data.path_segment_used = 0;
        result.telemetry_data.path_distance_traveled = 0.f;
        result.telemetry_data.path_distance_total = 0.f;
        result.telemetry_data.path_segment_progress = 0.f;
        result.telemetry_data.distance_from_path = (target_pos - (target_path.size() > 0 ? target_path[0] : target_pos)).magnitude();

        // Dodge & reachable region data
        result.telemetry_data.dodge_time = dodge_time;
        result.telemetry_data.effective_reaction_time = effective_reaction_time;
        result.telemetry_data.reachable_radius = reachable_region.max_radius;
        result.telemetry_data.reachable_center_x = reachable_region.center.x;
        result.telemetry_data.reachable_center_z = reachable_region.center.z;
        result.telemetry_data.effective_move_speed = effective_move_speed;

        // Outcome tracking (will be filled in later if we track outcomes)
        result.telemetry_data.outcome_recorded = false;
        result.telemetry_data.was_hit = false;
        result.telemetry_data.actual_pos_x = 0.f;
        result.telemetry_data.actual_pos_z = 0.f;
        result.telemetry_data.prediction_error = 0.f;
        result.telemetry_data.time_to_outcome = 0.f;

        return result;
    }

    HybridPredictionResult HybridFusionEngine::compute_targeted_prediction(
        game_object* source,
        game_object* target,
        const pred_sdk::spell_data& spell,
        TargetBehaviorTracker& tracker,
        const EdgeCases::EdgeCaseAnalysis& edge_cases)
    {
        (void)spell;  // Targeted spells ignore most spell parameters

        HybridPredictionResult result;

        // FIX: Add dead checks - is_valid() doesn't return false for dead targets
        if (!source || !target || !source->is_valid() || !target->is_valid() ||
            source->is_dead() || target->is_dead())
        {
            result.is_valid = false;
            return result;
        }

        // Validate SDK is initialized
        if (!g_sdk || !g_sdk->clock_facade)
        {
            result.is_valid = false;
            result.reasoning = "SDK not initialized";
            return result;
        }

        // Targeted spells can't miss (unless target becomes untargetable)
        // FIX: Use server position to avoid casting at stale client position (30-100ms lag)
        result.cast_position = target->get_server_position();
        result.hit_chance = 1.0f;
        result.physics_contribution = 1.0f;
        result.behavior_contribution = 1.0f;
        result.confidence_score = compute_confidence_score(source, target, spell, tracker, edge_cases);
        result.is_valid = true;
        result.reasoning = "Targeted spell - guaranteed hit (unless target becomes untargetable)";

        return result;
    }

    HybridPredictionResult HybridFusionEngine::compute_vector_prediction(
        game_object* source,
        game_object* target,
        const pred_sdk::spell_data& spell,
        TargetBehaviorTracker& tracker,
        const EdgeCases::EdgeCaseAnalysis& edge_cases)
    {
        HybridPredictionResult result;

        // FIX: Add dead checks - is_valid() doesn't return false for dead targets
        if (!source || !target || !source->is_valid() || !target->is_valid() ||
            source->is_dead() || target->is_dead())
        {
            result.is_valid = false;
            return result;
        }

        // Validate SDK is initialized
        if (!g_sdk || !g_sdk->clock_facade)
        {
            result.is_valid = false;
            result.reasoning = "SDK not initialized";
            return result;
        }

        // Step 1: Compute arrival time with iterative intercept refinement
        // CRITICAL FIX: Arrival time must account for target movement during flight
        math::vector3 source_pos = source->get_position();
        math::vector3 target_client_pos = target->get_position();
        // FIX: Use server position for target to avoid latency lag (30-100ms behind)
        math::vector3 target_pos = target->get_server_position();
        float initial_distance = distance_2d(target_pos, source_pos);  // 2D distance (ignore height)

        float arrival_time = PhysicsPredictor::compute_arrival_time(
            source_pos,
            target_pos,
            spell.projectile_speed,
            spell.delay,
            spell.proc_delay  // PROC_DELAY FIX: Include activation delay (e.g., Syndra Q pop time)
        );
        float initial_arrival_time = arrival_time;

        // ANIMATION LOCK DELAY (Stop-then-Go):
        // Calculate remaining lock time for accurate path prediction
        // Target stays at current position during lock, then moves at full speed
        // ADAPTIVE: Use measured cancel delay for this specific player
        float animation_lock_delay = 0.f;
        if (!target->is_moving() && (is_auto_attacking(target) || is_casting_spell(target) || is_channeling(target)))
        {
            // Use adaptive reaction buffer if we have enough samples, otherwise default
            float reaction_buffer = tracker.get_adaptive_reaction_buffer();
            animation_lock_delay = get_remaining_lock_time_adaptive(target, reaction_buffer);
        }

        // Track refinement convergence
        int refinement_iterations = 0;
        bool arrival_converged = false;
        math::vector3 final_predicted_pos = target_pos;

        // Refine arrival time iteratively (converges in 2-3 iterations)
        // PERFORMANCE: Early exit if converged (< 1ms change)
        for (int iteration = 0; iteration < 3; ++iteration)
        {
            refinement_iterations = iteration + 1;
            float prev_arrival = arrival_time;
            math::vector3 predicted_pos = PhysicsPredictor::predict_on_path(target, arrival_time, animation_lock_delay);
            final_predicted_pos = predicted_pos;

            arrival_time = PhysicsPredictor::compute_arrival_time(
                source_pos,
                predicted_pos,
                spell.projectile_speed,
                spell.delay,
                spell.proc_delay
            );

            // Early exit if converged (change < 1ms)
            if (std::abs(arrival_time - prev_arrival) < 0.001f)
            {
                arrival_converged = true;
                break;
            }
        }

        float final_arrival_time = arrival_time;
        float predicted_distance = distance_2d(final_predicted_pos, source_pos);  // 2D

        // Step 2: Build reachable region (physics)
        // FIX: Use path-following prediction for better accuracy
        math::vector3 path_predicted_pos = PhysicsPredictor::predict_on_path(target, arrival_time, animation_lock_delay);

        // POINT-BLANK DETECTION: At very close range, different rules apply
        // Use server position for accurate distance (client position lags)
        // HIGH GROUND FIX: 2D distance (ignore height)
        float current_distance = distance_2d(target->get_server_position(), source_pos);
        bool is_point_blank = current_distance < 200.f;  // Within 200 units

        math::vector3 target_velocity = tracker.get_current_velocity();
        float move_speed = target->get_move_speed();  // Stat value for historical lookups
        float effective_move_speed = get_effective_move_speed(target, arrival_time);  // Scaled by free time

        // USE OBSERVED JUKE MAGNITUDE for reachable region
        const DodgePattern& dodge_pattern = tracker.get_dodge_pattern();
        float observed_magnitude = dodge_pattern.get_juke_magnitude(move_speed);

        // Calculate effective reaction time accounting for FOW and cast delay
        // FOW FIX: If we're hidden, enemies can't see our animation - they get full reaction time
        // VISIBLE: Enemies react DURING cast animation, consuming reaction time
        float effective_reaction_time = get_effective_reaction_time(source, target, spell.delay);

        float max_dodge_time = arrival_time - effective_reaction_time;
        float dodge_time = 0.f;
        if (max_dodge_time > 0.f && effective_move_speed > EPSILON)
        {
            float observed_dodge_time = observed_magnitude / effective_move_speed;
            dodge_time = std::min(observed_dodge_time, max_dodge_time);

            // CLOSE-RANGE FIX: Dramatically reduce dodge time for fast spells
            // At close range (arrival < 0.5s), targets can't execute full jukes
            // They barely finish reacting before spell lands
            // GUARD: Skip if effective_reaction_time >= 0.5s to avoid division by zero
            if (arrival_time < 0.5f && effective_reaction_time < 0.5f - EPSILON)
            {
                // Scale dodge time: 0.3s arrival → 40% dodge, 0.5s → 100% dodge
                float close_range_scale = (arrival_time - effective_reaction_time) / (0.5f - effective_reaction_time);
                close_range_scale = std::clamp(close_range_scale, 0.f, 1.f);
                dodge_time *= close_range_scale;

                // POINT-BLANK OVERRIDE: At <200 units, minimal dodge time for narrow skillshots
                // Narrow skillshots (Pyke Q: 140 width, Thresh Q: 140 width) need pinpoint accuracy
                // At 400 MS: 0.01s = 4 unit radius (vs 0.05s = 20 unit radius)
                float min_dodge = is_point_blank ? 0.01f : 0.05f;
                dodge_time = std::max(dodge_time, min_dodge);
            }
        }

        // Use dynamic acceleration if tracker has measured it
        float dynamic_accel = tracker.has_measured_physics() ?
            tracker.get_measured_acceleration() : DEFAULT_ACCELERATION;

        // SPLIT-PATH APPROACH: Calculate position at reaction time
        // During reaction time, target follows their clicked path (not dodging yet)
        // This handles curved paths correctly (linear velocity extrapolation doesn't)
        math::vector3 pos_at_reaction = PhysicsPredictor::predict_on_path(target, effective_reaction_time, animation_lock_delay);

        // Build reachable region FROM the reaction position
        // Zero velocity because path-following during reaction time is already handled above
        // Only need to calculate dodge possibilities from the reaction point
        ReachableRegion reachable_region = PhysicsPredictor::compute_reachable_region(
            pos_at_reaction,  // Start at reaction position (after path following)
            math::vector3(0, 0, 0),  // Zero velocity (drift already accounted for)
            arrival_time - effective_reaction_time,  // Remaining time to dodge
            effective_move_speed,
            DEFAULT_TURN_RATE,
            dynamic_accel,
            0.0f  // No more reaction drift needed (already at reaction point)
        );

        result.reachable_region = reachable_region;

        // Step 3: Build behavior PDF
        BehaviorPDF behavior_pdf = BehaviorPredictor::build_pdf_from_history(tracker, arrival_time, move_speed);
        BehaviorPredictor::apply_contextual_factors(behavior_pdf, tracker, target);

        // REACTION-GATE OVERRIDE: If undodgeable, force PDF to pure physics
        bool is_undodgeable = arrival_time < effective_reaction_time;
        if (is_undodgeable)
        {
            for (int i = 0; i < BehaviorPDF::GRID_SIZE; ++i)
                for (int j = 0; j < BehaviorPDF::GRID_SIZE; ++j)
                    behavior_pdf.pdf_grid[i][j] = 0.f;
            behavior_pdf.total_probability = 0.f;
        }
        behavior_pdf.origin = path_predicted_pos;
        if (is_undodgeable)
            behavior_pdf.add_weighted_sample(path_predicted_pos, 1.0f);

        result.behavior_pdf = behavior_pdf;

        // Step 4: Compute confidence score
        float confidence = compute_confidence_score(source, target, spell, tracker, edge_cases);
        result.confidence_score = confidence;

        // Note: Vector spells use orientation optimization which handles lateral
        // variance through PDF evaluation. No fusion needed - would double-compensate.

        // Step 5: Optimize vector orientation
        // Test multiple orientations to find best two-position configuration
        size_t sample_count = tracker.get_history().size();
        float current_time = 0.f;
        if (g_sdk && g_sdk->clock_facade)
            current_time = g_sdk->clock_facade->get_game_time();
        float time_since_update = current_time - tracker.get_last_update_time();
        VectorConfiguration best_config = optimize_vector_orientation(
            source,
            target,
            reachable_region.center,
            reachable_region,
            behavior_pdf,
            spell,
            confidence,
            sample_count,
            time_since_update
        );

        // Step 6: Set result from best configuration
        result.first_cast_position = best_config.first_cast_position;
        result.cast_position = best_config.cast_position;
        result.physics_contribution = best_config.physics_prob;
        result.behavior_contribution = best_config.behavior_prob;
        result.hit_chance = best_config.hit_chance;

#if HYBRID_PRED_ENABLE_REASONING
        // Generate mathematical reasoning
        std::ostringstream reasoning;
        reasoning << "Hybrid Prediction Analysis (VECTOR):\n";
        reasoning << "  Arrival Time: " << arrival_time << "s\n";
        reasoning << "  Reachable Radius: " << reachable_region.max_radius << " units\n";
        reasoning << "  Vector Length: " << spell.range << " units\n";
        reasoning << "  Vector Width: " << (spell.radius * 2.f) << " units\n";
        reasoning << "  First Cast: (" << best_config.first_cast_position.x << ", "
            << best_config.first_cast_position.z << ")\n";
        reasoning << "  Second Cast: (" << best_config.cast_position.x << ", "
            << best_config.cast_position.z << ")\n";
        reasoning << "  Physics Hit Prob: " << (best_config.physics_prob * 100.f) << "%\n";
        reasoning << "  Behavior Hit Prob: " << (best_config.behavior_prob * 100.f) << "%\n";
        reasoning << "  Confidence: " << (confidence * 100.f) << "%\n";
        reasoning << "  Final HitChance: " << (result.hit_chance * 100.f) << "%\n";
        result.reasoning = reasoning.str();
#else
        result.reasoning = "";
#endif

        result.is_valid = true;

        // Update opportunistic casting signals
        update_opportunity_signals(result, source, target, spell, tracker);

        // ===================================================================
        // POPULATE DETAILED TELEMETRY DEBUG DATA
        // ===================================================================
        result.telemetry_data.source_pos_x = source_pos.x;
        result.telemetry_data.source_pos_z = source_pos.z;
        result.telemetry_data.target_client_pos_x = target_client_pos.x;
        result.telemetry_data.target_client_pos_z = target_client_pos.z;
        result.telemetry_data.target_server_pos_x = target_pos.x;
        result.telemetry_data.target_server_pos_z = target_pos.z;
        result.telemetry_data.predicted_pos_x = path_predicted_pos.x;
        result.telemetry_data.predicted_pos_z = path_predicted_pos.z;
        result.telemetry_data.cast_pos_x = result.cast_position.x;
        result.telemetry_data.cast_pos_z = result.cast_position.z;

        // Arrival time data
        result.telemetry_data.initial_distance = initial_distance;
        result.telemetry_data.initial_arrival_time = initial_arrival_time;
        result.telemetry_data.refinement_iterations = refinement_iterations;
        result.telemetry_data.final_arrival_time = final_arrival_time;
        result.telemetry_data.arrival_time_change = final_arrival_time - initial_arrival_time;
        result.telemetry_data.arrival_converged = arrival_converged;
        result.telemetry_data.predicted_distance = predicted_distance;

        // Path prediction data
        auto target_path = target->get_path();
        result.telemetry_data.path_segment_count = static_cast<int>(target_path.size());
        result.telemetry_data.path_segment_used = 0;
        result.telemetry_data.path_distance_traveled = 0.f;
        result.telemetry_data.path_distance_total = 0.f;
        result.telemetry_data.path_segment_progress = 0.f;
        result.telemetry_data.distance_from_path = (target_pos - (target_path.size() > 0 ? target_path[0] : target_pos)).magnitude();

        // Dodge & reachable region data
        result.telemetry_data.dodge_time = dodge_time;
        result.telemetry_data.effective_reaction_time = effective_reaction_time;
        result.telemetry_data.reachable_radius = reachable_region.max_radius;
        result.telemetry_data.reachable_center_x = reachable_region.center.x;
        result.telemetry_data.reachable_center_z = reachable_region.center.z;
        result.telemetry_data.effective_move_speed = effective_move_speed;

        // Outcome tracking (will be filled in later if we track outcomes)
        result.telemetry_data.outcome_recorded = false;
        result.telemetry_data.was_hit = false;
        result.telemetry_data.actual_pos_x = 0.f;
        result.telemetry_data.actual_pos_z = 0.f;
        result.telemetry_data.prediction_error = 0.f;
        result.telemetry_data.time_to_outcome = 0.f;

        return result;
    }

    HybridPredictionResult HybridFusionEngine::compute_cone_prediction(
        game_object* source,
        game_object* target,
        const pred_sdk::spell_data& spell,
        TargetBehaviorTracker& tracker,
        const EdgeCases::EdgeCaseAnalysis& edge_cases,
        float cone_angle_override)
    {
        HybridPredictionResult result;

        // FIX: Add dead checks - is_valid() doesn't return false for dead targets
        if (!source || !target || !source->is_valid() || !target->is_valid() ||
            source->is_dead() || target->is_dead())
        {
            result.is_valid = false;
            return result;
        }

        // Validate SDK is initialized
        if (!g_sdk || !g_sdk->clock_facade)
        {
            result.is_valid = false;
            result.reasoning = "SDK not initialized";
            return result;
        }

        // Step 1: Compute arrival time with iterative intercept refinement
        // CRITICAL FIX: Arrival time must account for target movement during flight
        math::vector3 source_pos = source->get_position();
        math::vector3 target_client_pos = target->get_position();
        // FIX: Use server position for target to avoid latency lag (30-100ms behind)
        math::vector3 target_pos = target->get_server_position();
        float initial_distance = distance_2d(target_pos, source_pos);  // 2D distance (ignore height)

        float arrival_time = PhysicsPredictor::compute_arrival_time(
            source_pos,
            target_pos,
            spell.projectile_speed,
            spell.delay,
            spell.proc_delay  // PROC_DELAY FIX: Include activation delay (e.g., Syndra Q pop time)
        );
        float initial_arrival_time = arrival_time;

        // ANIMATION LOCK DELAY (Stop-then-Go):
        // Calculate remaining lock time for accurate path prediction
        // Target stays at current position during lock, then moves at full speed
        // ADAPTIVE: Use measured cancel delay for this specific player
        float animation_lock_delay = 0.f;
        if (!target->is_moving() && (is_auto_attacking(target) || is_casting_spell(target) || is_channeling(target)))
        {
            // Use adaptive reaction buffer if we have enough samples, otherwise default
            float reaction_buffer = tracker.get_adaptive_reaction_buffer();
            animation_lock_delay = get_remaining_lock_time_adaptive(target, reaction_buffer);
        }

        // Track refinement convergence
        int refinement_iterations = 0;
        bool arrival_converged = false;
        math::vector3 final_predicted_pos = target_pos;

        // Refine arrival time iteratively (converges in 2-3 iterations)
        // PERFORMANCE: Early exit if converged (< 1ms change)
        for (int iteration = 0; iteration < 3; ++iteration)
        {
            refinement_iterations = iteration + 1;
            float prev_arrival = arrival_time;
            math::vector3 predicted_pos = PhysicsPredictor::predict_on_path(target, arrival_time, animation_lock_delay);
            final_predicted_pos = predicted_pos;

            arrival_time = PhysicsPredictor::compute_arrival_time(
                source_pos,
                predicted_pos,
                spell.projectile_speed,
                spell.delay,
                spell.proc_delay
            );

            // Early exit if converged (change < 1ms)
            if (std::abs(arrival_time - prev_arrival) < 0.001f)
            {
                arrival_converged = true;
                break;
            }
        }

        float final_arrival_time = arrival_time;
        float predicted_distance = distance_2d(final_predicted_pos, source_pos);  // 2D

        // Step 2: Build reachable region (physics)
        // FIX: Use path-following prediction like circular/linear for consistency
        math::vector3 path_predicted_pos = PhysicsPredictor::predict_on_path(target, arrival_time, animation_lock_delay);

        // POINT-BLANK DETECTION: At very close range, different rules apply
        // Use server position for accurate distance (client position lags)
        // HIGH GROUND FIX: 2D distance (ignore height)
        float current_distance = distance_2d(target->get_server_position(), source_pos);
        bool is_point_blank = current_distance < 200.f;  // Within 200 units

        float move_speed = target->get_move_speed();  // Stat value for historical lookups
        float effective_move_speed = get_effective_move_speed(target, arrival_time);  // Scaled by free time

        // Calculate dodge_time (missing from original cone implementation!)
        math::vector3 target_velocity = tracker.get_current_velocity();
        const DodgePattern& dodge_pattern = tracker.get_dodge_pattern();
        float observed_magnitude = dodge_pattern.get_juke_magnitude(move_speed);

        // Calculate effective reaction time accounting for FOW and cast delay
        // FOW FIX: If we're hidden, enemies can't see our animation - they get full reaction time
        // VISIBLE: Enemies react DURING cast animation, consuming reaction time
        float effective_reaction_time = get_effective_reaction_time(source, target, spell.delay);
        float max_dodge_time = arrival_time - effective_reaction_time;
        float dodge_time = 0.f;
        if (max_dodge_time > 0.f && effective_move_speed > EPSILON)
        {
            float observed_dodge_time = observed_magnitude / effective_move_speed;
            dodge_time = std::min(observed_dodge_time, max_dodge_time);

            // CLOSE-RANGE FIX: Dramatically reduce dodge time for fast spells
            // GUARD: Skip if effective_reaction_time >= 0.5s to avoid division by zero
            if (arrival_time < 0.5f && effective_reaction_time < 0.5f - EPSILON)
            {
                float close_range_scale = (arrival_time - effective_reaction_time) / (0.5f - effective_reaction_time);
                close_range_scale = std::clamp(close_range_scale, 0.f, 1.f);
                dodge_time *= close_range_scale;

                // POINT-BLANK OVERRIDE: At <200 units, minimal dodge time
                float min_dodge = is_point_blank ? 0.01f : 0.05f;
                dodge_time = std::max(dodge_time, min_dodge);
            }
        }

        // Use dynamic acceleration if tracker has measured it, otherwise fall back to defaults
        float dynamic_accel = tracker.has_measured_physics() ?
            tracker.get_measured_acceleration() : DEFAULT_ACCELERATION;

        // SPLIT-PATH APPROACH: Calculate position at reaction time
        // During reaction time, target follows their clicked path (not dodging yet)
        // This handles curved paths correctly (linear velocity extrapolation doesn't)
        math::vector3 pos_at_reaction = PhysicsPredictor::predict_on_path(target, effective_reaction_time, animation_lock_delay);

        // Build reachable region FROM the reaction position
        // Zero velocity because path-following during reaction time is already handled above
        // Only need to calculate dodge possibilities from the reaction point
        ReachableRegion reachable_region = PhysicsPredictor::compute_reachable_region(
            pos_at_reaction,  // Start at reaction position (after path following)
            math::vector3(0, 0, 0),  // Zero velocity (drift already accounted for)
            arrival_time - effective_reaction_time,  // Remaining time to dodge
            effective_move_speed,
            DEFAULT_TURN_RATE,
            dynamic_accel,
            0.0f  // No more reaction drift needed (already at reaction point)
        );

        result.reachable_region = reachable_region;

        // Step 3: Build behavior PDF and align with path prediction
        BehaviorPDF behavior_pdf = BehaviorPredictor::build_pdf_from_history(tracker, arrival_time, move_speed);
        BehaviorPredictor::apply_contextual_factors(behavior_pdf, tracker, target);

        // REACTION-GATE OVERRIDE: If undodgeable, force PDF to pure physics
        bool is_undodgeable = arrival_time < effective_reaction_time;
        if (is_undodgeable)
        {
            for (int i = 0; i < BehaviorPDF::GRID_SIZE; ++i)
                for (int j = 0; j < BehaviorPDF::GRID_SIZE; ++j)
                    behavior_pdf.pdf_grid[i][j] = 0.f;
            behavior_pdf.total_probability = 0.f;
        }

        // Center PDF on path prediction at full arrival time
        behavior_pdf.origin = path_predicted_pos;
        if (is_undodgeable)
            behavior_pdf.add_weighted_sample(path_predicted_pos, 1.0f);
        result.behavior_pdf = behavior_pdf;

        // Step 4: Compute confidence score
        float confidence = compute_confidence_score(source, target, spell, tracker, edge_cases);
        result.confidence_score = confidence;

        // Step 5: Compute cone parameters
        float cone_half_angle;
        float cone_range = spell.range;

        if (cone_angle_override > 0.f)
        {
            // Use the angle from get_cast_cone_angle() - this is the TOTAL angle in degrees
            // Convert to half-angle in radians
            cone_half_angle = (cone_angle_override * 0.5f) * (PI / 180.f);
        }
        else
        {
            // Fallback: calculate from radius (assuming radius = width at range)
            // This is less accurate but works when cone angle isn't available
            cone_half_angle = std::atan2(spell.radius, spell.range);
        }

        // Optimal direction toward predicted target position
        math::vector3 to_center = reachable_region.center - source->get_position();
        float dist_to_center = to_center.magnitude();

        constexpr float MIN_SAFE_DISTANCE = 1.0f;  // Minimum safe distance for normalization
        if (dist_to_center < MIN_SAFE_DISTANCE)
        {
            result.is_valid = false;
            result.reasoning = "Target too close - zero distance";
            return result;
        }

        math::vector3 direction = to_center / dist_to_center;  // Safe manual normalize
        result.cast_position = source->get_position() + direction * cone_range;

        // Step 6: Compute hit probabilities for cone
        // Use effective range = cone range + target bounding radius
        float effective_range = cone_range + target->get_bounding_radius();

        float physics_prob = compute_cone_reachability_overlap(
            source->get_position(),
            direction,
            cone_half_angle,
            effective_range,
            reachable_region
        );

        float behavior_prob = compute_cone_behavior_probability(
            source->get_position(),
            direction,
            cone_half_angle,
            effective_range,
            behavior_pdf
        );

        result.physics_contribution = physics_prob;
        result.behavior_contribution = behavior_prob;

        // Weighted geometric fusion (trust physics more when behavior samples are sparse)
        size_t sample_count = tracker.get_history().size();
        float current_time = 0.f;
        if (g_sdk && g_sdk->clock_facade)
            current_time = g_sdk->clock_facade->get_game_time();
        float time_since_update = current_time - tracker.get_last_update_time();
        result.hit_chance = fuse_probabilities(physics_prob, behavior_prob, confidence, sample_count, time_since_update, move_speed, current_distance);
        result.hit_chance = std::clamp(result.hit_chance, 0.f, 1.f);

#if HYBRID_PRED_ENABLE_REASONING
        // Generate mathematical reasoning
        std::ostringstream reasoning;
        reasoning << "Hybrid Prediction Analysis (CONE):\n";
        reasoning << "  Arrival Time: " << arrival_time << "s\n";
        reasoning << "  Reachable Radius: " << reachable_region.max_radius << " units\n";
        reasoning << "  Cone Range: " << cone_range << " units\n";
        reasoning << "  Cone Half-Angle: " << (cone_half_angle * 180.f / PI) << " degrees\n";
        reasoning << "  Physics Hit Prob: " << (physics_prob * 100.f) << "%\n";
        reasoning << "  Behavior Hit Prob: " << (behavior_prob * 100.f) << "%\n";
        reasoning << "  Confidence: " << (confidence * 100.f) << "%\n";
        reasoning << "  Final HitChance: " << (result.hit_chance * 100.f) << "%\n";
        result.reasoning = reasoning.str();
#else
        result.reasoning = "";
#endif

        result.is_valid = true;

        // Update opportunistic casting signals
        update_opportunity_signals(result, source, target, spell, tracker);

        // ===================================================================
        // POPULATE DETAILED TELEMETRY DEBUG DATA
        // ===================================================================
        result.telemetry_data.source_pos_x = source_pos.x;
        result.telemetry_data.source_pos_z = source_pos.z;
        result.telemetry_data.target_client_pos_x = target_client_pos.x;
        result.telemetry_data.target_client_pos_z = target_client_pos.z;
        result.telemetry_data.target_server_pos_x = target_pos.x;
        result.telemetry_data.target_server_pos_z = target_pos.z;
        result.telemetry_data.predicted_pos_x = path_predicted_pos.x;
        result.telemetry_data.predicted_pos_z = path_predicted_pos.z;
        result.telemetry_data.cast_pos_x = result.cast_position.x;
        result.telemetry_data.cast_pos_z = result.cast_position.z;

        // Arrival time data
        result.telemetry_data.initial_distance = initial_distance;
        result.telemetry_data.initial_arrival_time = initial_arrival_time;
        result.telemetry_data.refinement_iterations = refinement_iterations;
        result.telemetry_data.final_arrival_time = final_arrival_time;
        result.telemetry_data.arrival_time_change = final_arrival_time - initial_arrival_time;
        result.telemetry_data.arrival_converged = arrival_converged;
        result.telemetry_data.predicted_distance = predicted_distance;

        // Path prediction data
        auto target_path = target->get_path();
        result.telemetry_data.path_segment_count = static_cast<int>(target_path.size());
        result.telemetry_data.path_segment_used = 0;
        result.telemetry_data.path_distance_traveled = 0.f;
        result.telemetry_data.path_distance_total = 0.f;
        result.telemetry_data.path_segment_progress = 0.f;
        result.telemetry_data.distance_from_path = (target_pos - (target_path.size() > 0 ? target_path[0] : target_pos)).magnitude();

        // Dodge & reachable region data
        result.telemetry_data.dodge_time = dodge_time;
        result.telemetry_data.effective_reaction_time = effective_reaction_time;
        result.telemetry_data.reachable_radius = reachable_region.max_radius;
        result.telemetry_data.reachable_center_x = reachable_region.center.x;
        result.telemetry_data.reachable_center_z = reachable_region.center.z;
        result.telemetry_data.effective_move_speed = effective_move_speed;

        // Outcome tracking (will be filled in later if we track outcomes)
        result.telemetry_data.outcome_recorded = false;
        result.telemetry_data.was_hit = false;
        result.telemetry_data.actual_pos_x = 0.f;
        result.telemetry_data.actual_pos_z = 0.f;
        result.telemetry_data.prediction_error = 0.f;
        result.telemetry_data.time_to_outcome = 0.f;

        return result;
    }

    // =========================================================================
    // GEOMETRY HELPERS FOR SPELL SHAPES
    // =========================================================================

    bool HybridFusionEngine::point_in_capsule(
        const math::vector3& point,
        const math::vector3& capsule_start,
        const math::vector3& capsule_end,
        float capsule_radius)
    {
        // Capsule = line segment + radius
        // Point is inside if distance to line segment ≤ radius

        math::vector3 segment = capsule_end - capsule_start;
        math::vector3 to_point = point - capsule_start;

        float segment_length_sq = segment.x * segment.x + segment.z * segment.z;

        if (segment_length_sq < EPSILON)
        {
            // Degenerate case: start == end (treat as sphere)
            float dist_sq = to_point.x * to_point.x + to_point.z * to_point.z;
            return dist_sq <= capsule_radius * capsule_radius;
        }

        // Project point onto line segment (clamped to [0, 1])
        float t = (to_point.x * segment.x + to_point.z * segment.z) / segment_length_sq;
        t = std::clamp(t, 0.f, 1.f);

        // Closest point on segment
        math::vector3 closest = capsule_start + segment * t;

        // Distance from point to closest point on segment
        float dx = point.x - closest.x;
        float dz = point.z - closest.z;
        float dist_sq = dx * dx + dz * dz;

        return dist_sq <= capsule_radius * capsule_radius;
    }

    float HybridFusionEngine::compute_capsule_reachability_overlap(
        const math::vector3& capsule_start,
        const math::vector3& capsule_direction,
        float capsule_length,
        float capsule_radius,
        const ReachableRegion& reachable_region)
    {
        // Quasi-Monte Carlo integration: Sample points in reachable circle using Fermat spiral,
        // count how many fall inside capsule (deterministic low-discrepancy sampling)

        if (reachable_region.area < EPSILON)
            return 0.f;

        math::vector3 capsule_end = capsule_start + capsule_direction * capsule_length;

        constexpr int SAMPLES = 64;  // Reduced from 128 - sufficient precision, 2x performance gain
        constexpr float SPIRAL_FACTOR = 7.f;  // Coprime with SAMPLES for uniform coverage
        int hits = 0;

        for (int i = 0; i < SAMPLES; ++i)
        {
            // Fermat spiral: uniform area distribution in reachable disk
            float r = reachable_region.max_radius * std::sqrt(static_cast<float>(i) / SAMPLES);
            float theta = (2.f * PI * i) / SAMPLES * SPIRAL_FACTOR;  // 7 is coprime with 64

            math::vector3 sample_point = reachable_region.center;
            sample_point.x += r * std::cos(theta);
            sample_point.z += r * std::sin(theta);

            if (point_in_capsule(sample_point, capsule_start, capsule_end, capsule_radius))
            {
                ++hits;
            }
        }

        return static_cast<float>(hits) / SAMPLES;
    }

    float HybridFusionEngine::compute_capsule_behavior_probability(
        const math::vector3& capsule_start,
        const math::vector3& capsule_direction,
        float capsule_length,
        float capsule_radius,
        const BehaviorPDF& pdf)
    {
        if (pdf.total_probability < EPSILON)
            return 1.0f;  // Neutral fallback

        // Direct grid summation (more accurate than sampling)
        // Sum probability mass of all cells whose centers fall inside the capsule
        math::vector3 capsule_end = capsule_start + capsule_direction * capsule_length;
        float prob = 0.f;

        // PERFORMANCE OPTIMIZATION: Calculate bounding box of capsule in grid coordinates
        // Only iterate cells that could possibly intersect with the capsule
        // This reduces checks from 1024 (32×32) down to ~50-100 cells (10-20x speedup!)

        // Find min/max world coordinates of capsule bounding box
        float min_wx = std::min(capsule_start.x, capsule_end.x) - capsule_radius;
        float max_wx = std::max(capsule_start.x, capsule_end.x) + capsule_radius;
        float min_wz = std::min(capsule_start.z, capsule_end.z) - capsule_radius;
        float max_wz = std::max(capsule_start.z, capsule_end.z) + capsule_radius;

        // Convert world coordinates to grid coordinates
        // Grid formula: grid_coord = (world_coord - origin) / cell_size + GRID_SIZE/2
        int grid_center = BehaviorPDF::GRID_SIZE / 2;

        int min_x = static_cast<int>((min_wx - pdf.origin.x) / pdf.cell_size) + grid_center;
        int max_x = static_cast<int>((max_wx - pdf.origin.x) / pdf.cell_size) + grid_center + 1;
        int min_z = static_cast<int>((min_wz - pdf.origin.z) / pdf.cell_size) + grid_center;
        int max_z = static_cast<int>((max_wz - pdf.origin.z) / pdf.cell_size) + grid_center + 1;

        // Clamp to grid bounds [0, GRID_SIZE)
        min_x = std::max(0, min_x);
        max_x = std::min(BehaviorPDF::GRID_SIZE, max_x);
        min_z = std::max(0, min_z);
        max_z = std::min(BehaviorPDF::GRID_SIZE, max_z);

        // Iterate ONLY cells within bounding box (much faster!)
        for (int x = min_x; x < max_x; ++x)
        {
            for (int z = min_z; z < max_z; ++z)
            {
                // World position of cell center
                float wx = pdf.origin.x + (x - grid_center + 0.5f) * pdf.cell_size;
                float wz = pdf.origin.z + (z - grid_center + 0.5f) * pdf.cell_size;
                math::vector3 cell_center(wx, pdf.origin.y, wz);

                // Check if cell center is inside capsule
                if (point_in_capsule(cell_center, capsule_start, capsule_end, capsule_radius))
                {
                    prob += pdf.pdf_grid[x][z];
                }
            }
        }

        // PDF is normalized (sums to 1), so this sum is the exact hit probability
        return std::clamp(prob, 0.f, 1.f);
    }

    // =========================================================================
    // CONE GEOMETRY HELPERS
    // =========================================================================

    bool HybridFusionEngine::point_in_cone(
        const math::vector3& point,
        const math::vector3& cone_origin,
        const math::vector3& cone_direction,
        float cone_half_angle,
        float cone_range)
    {
        // Cone = circular sector in 3D
        // Point is inside if:
        // 1. Distance from origin ≤ range
        // 2. Angle from cone axis ≤ half_angle

        math::vector3 to_point = point - cone_origin;
        float distance_sq = to_point.x * to_point.x + to_point.z * to_point.z;

        // Check range
        if (distance_sq > cone_range * cone_range)
            return false;

        float distance = std::sqrt(distance_sq);
        // CRASH PROTECTION: Use larger safety margin for division
        constexpr float MIN_SAFE_DISTANCE = 0.01f;
        if (distance < MIN_SAFE_DISTANCE)
            return true;  // At origin or too close

        // Check angle
        // cos(angle) = dot(to_point, cone_direction) / |to_point|
        float dot_product = to_point.x * cone_direction.x + to_point.z * cone_direction.z;
        float cos_angle = dot_product / distance;

        // cos decreases as angle increases, so:
        // angle ≤ half_angle  ⇔  cos(angle) ≥ cos(half_angle)
        float cos_half_angle = std::cos(cone_half_angle);

        return cos_angle >= cos_half_angle;
    }

    float HybridFusionEngine::compute_cone_reachability_overlap(
        const math::vector3& cone_origin,
        const math::vector3& cone_direction,
        float cone_half_angle,
        float cone_range,
        const ReachableRegion& reachable_region)
    {
        // Quasi-Monte Carlo integration: Sample points in reachable circle using Fermat spiral,
        // count how many fall inside cone (deterministic low-discrepancy sampling)

        if (reachable_region.area < EPSILON)
            return 0.f;

        constexpr int SAMPLES = 64;  // Reduced from 128 - sufficient precision, 2x performance gain
        constexpr float SPIRAL_FACTOR = 7.f;  // Coprime with SAMPLES for uniform coverage
        int hits = 0;

        for (int i = 0; i < SAMPLES; ++i)
        {
            // Fermat spiral: uniform area distribution in reachable disk
            float r = reachable_region.max_radius * std::sqrt(static_cast<float>(i) / SAMPLES);
            float theta = (2.f * PI * i) / SAMPLES * SPIRAL_FACTOR;  // 7 is coprime with 64

            math::vector3 sample_point = reachable_region.center;
            sample_point.x += r * std::cos(theta);
            sample_point.z += r * std::sin(theta);

            if (point_in_cone(sample_point, cone_origin, cone_direction, cone_half_angle, cone_range))
            {
                ++hits;
            }
        }

        return static_cast<float>(hits) / SAMPLES;
    }

    float HybridFusionEngine::compute_cone_behavior_probability(
        const math::vector3& cone_origin,
        const math::vector3& cone_direction,
        float cone_half_angle,
        float cone_range,
        const BehaviorPDF& pdf)
    {
        if (pdf.total_probability < EPSILON)
            return 1.0f;  // Neutral fallback

        // Direct grid summation (more accurate than sampling)
        // Sum probability mass of all cells whose centers fall inside the cone
        float prob = 0.f;

        // PERFORMANCE OPTIMIZATION: Calculate bounding box of cone in grid coordinates
        // Only iterate cells that could possibly be inside the cone
        // Cone bounding box: origin to (origin + direction * range) ± lateral_extent

        // Calculate cone endpoint
        math::vector3 cone_end = cone_origin + cone_direction * cone_range;

        // Calculate maximum lateral extent at the cone's end (perpendicular to direction)
        float lateral_extent = cone_range * std::tan(cone_half_angle);

        // Find min/max world coordinates of cone bounding box
        float min_wx = std::min(cone_origin.x, cone_end.x) - lateral_extent;
        float max_wx = std::max(cone_origin.x, cone_end.x) + lateral_extent;
        float min_wz = std::min(cone_origin.z, cone_end.z) - lateral_extent;
        float max_wz = std::max(cone_origin.z, cone_end.z) + lateral_extent;

        // Convert world coordinates to grid coordinates
        int grid_center = BehaviorPDF::GRID_SIZE / 2;

        // CRASH PROTECTION: Check cell_size before division
        if (pdf.cell_size < 0.1f)
            return 1.0f;  // Fallback neutral probability

        int min_x = static_cast<int>((min_wx - pdf.origin.x) / pdf.cell_size) + grid_center;
        int max_x = static_cast<int>((max_wx - pdf.origin.x) / pdf.cell_size) + grid_center + 1;
        int min_z = static_cast<int>((min_wz - pdf.origin.z) / pdf.cell_size) + grid_center;
        int max_z = static_cast<int>((max_wz - pdf.origin.z) / pdf.cell_size) + grid_center + 1;

        // Clamp to grid bounds [0, GRID_SIZE)
        min_x = std::max(0, min_x);
        max_x = std::min(BehaviorPDF::GRID_SIZE, max_x);
        min_z = std::max(0, min_z);
        max_z = std::min(BehaviorPDF::GRID_SIZE, max_z);

        // Iterate ONLY cells within bounding box (much faster!)
        for (int x = min_x; x < max_x; ++x)
        {
            for (int z = min_z; z < max_z; ++z)
            {
                // World position of cell center
                float wx = pdf.origin.x + (x - grid_center + 0.5f) * pdf.cell_size;
                float wz = pdf.origin.z + (z - grid_center + 0.5f) * pdf.cell_size;
                math::vector3 cell_center(wx, pdf.origin.y, wz);

                // Check if cell center is inside cone
                if (point_in_cone(cell_center, cone_origin, cone_direction, cone_half_angle, cone_range))
                {
                    prob += pdf.pdf_grid[x][z];
                }
            }
        }

        // PDF is normalized (sums to 1), so this sum is the exact hit probability
        return std::clamp(prob, 0.f, 1.f);
    }

    // =========================================================================
    // VECTOR SPELL OPTIMIZATION HELPERS
    // =========================================================================

    HybridFusionEngine::VectorConfiguration HybridFusionEngine::optimize_vector_orientation(
        game_object* source,
        game_object* target,
        const math::vector3& predicted_target_pos,
        const ReachableRegion& reachable_region,
        const BehaviorPDF& behavior_pdf,
        const pred_sdk::spell_data& spell,
        float confidence,
        size_t sample_count,
        float time_since_update)
    {
        /**
         * Vector Spell Optimization Algorithm (Viktor E, Rumble R, etc.)
         * ===============================================================
         *
         * Vector spells are NOT like linear skillshots. You want to:
         * 1. Cast ALONG their movement path (to burn them as they run)
         * 2. Cast PERPENDICULAR to catch strafing/dodging
         * 3. NOT just aim straight at them like Ezreal Q
         *
         * Method:
         * 1. Test smart angles: parallel and perpendicular to velocity
         * 2. Test sweep angles: 16 angles for full 360° coverage
         * 3. Center vector on predicted target position
         * 4. Clamp first_cast to within cast_range
         * 5. Return configuration with highest hit_chance
         */

        VectorConfiguration best_config;
        best_config.hit_chance = 0.f;

        math::vector3 source_pos = source->get_position();
        float vector_length = spell.range;
        float vector_width = spell.radius + target->get_bounding_radius();
        float max_first_cast_range = spell.cast_range;

        // If cast_range is 0, use range as default (some spells don't set cast_range)
        if (max_first_cast_range < EPSILON)
            max_first_cast_range = spell.range;

        // Get target velocity direction for smart alignment
        math::vector3 target_vel = reachable_region.velocity;
        float target_speed = target_vel.magnitude();
        math::vector3 move_dir = (target_speed > 10.f) ? target_vel / target_speed : math::vector3(0, 0, 0);

        // Build list of angles to test
        std::vector<float> test_angles;

        // 1. SMART ANGLES: Align with and against movement (highest priority)
        if (target_speed > 10.f)
        {
            float move_angle = std::atan2(move_dir.z, move_dir.x);
            test_angles.push_back(move_angle);              // Parallel (Chase/Run) - burn them as they flee
            test_angles.push_back(move_angle + PI);         // Reverse (Kiting) - burn them as they chase
            test_angles.push_back(move_angle + PI / 2);     // Perpendicular (Catch strafe left)
            test_angles.push_back(move_angle - PI / 2);     // Perpendicular (Catch strafe right)
        }

        // 2. SWEEP ANGLES: Full 360° coverage for stationary targets or weird angles
        constexpr int NUM_SWEEP = 16;
        for (int i = 0; i < NUM_SWEEP; ++i)
        {
            test_angles.push_back((2.f * PI * i) / NUM_SWEEP);
        }

        // Test each angle
        for (float angle : test_angles)
        {
            math::vector3 dir(std::cos(angle), 0.f, std::sin(angle));

            // Strategy: Center the vector on the predicted target position
            // Start = Center - (Dir * Length / 2)
            // End   = Center + (Dir * Length / 2)
            math::vector3 potential_first = predicted_target_pos - dir * (vector_length * 0.5f);
            math::vector3 potential_second = predicted_target_pos + dir * (vector_length * 0.5f);

            // Constraint: First cast must be within cast_range of source (2D - High Ground Fix)
            float dist_to_start = distance_2d(potential_first, source_pos);

            // If start is out of range, slide the vector closer while keeping direction
            if (dist_to_start > max_first_cast_range)
            {
                math::vector3 to_start = potential_first - source_pos;
                float to_start_mag = magnitude_2d(to_start);
                if (to_start_mag > EPSILON)
                {
                    // FIX: Must use 2D direction for normalization (cannot divide 3D vector by 2D magnitude!)
                    math::vector3 to_start_2d = flatten_2d(to_start);
                    math::vector3 direction_2d = to_start_2d / to_start_mag;  // Now both are 2D
                    float y_offset = potential_first.y - source_pos.y;  // Preserve Y difference
                    potential_first = source_pos + direction_2d * max_first_cast_range;
                    potential_first.y = source_pos.y + y_offset;  // Restore Y coordinate
                    potential_second = potential_first + dir * vector_length;
                }
            }

            // Calculate Hit Chance for this orientation
            float physics_prob = compute_capsule_reachability_overlap(
                potential_first, dir, vector_length, vector_width, reachable_region
            );

            float behavior_prob = compute_capsule_behavior_probability(
                potential_first, dir, vector_length, vector_width, behavior_pdf
            );

            // Compute Hit Chance (2D distance for High Ground Fix)
            float distance_to_target = distance_2d(potential_first, source_pos);
            float test_hit_chance = fuse_probabilities(
                physics_prob, behavior_prob, confidence,
                sample_count, time_since_update, target_speed, distance_to_target
            );

            // Bonus: If aligned with movement, slight boost (caught in stride)
            if (target_speed > 10.f)
            {
                float alignment = std::abs(dir.dot(move_dir));
                if (alignment > 0.9f)
                    test_hit_chance *= 1.05f; // 5% bonus for parallel alignment
            }

            if (test_hit_chance > best_config.hit_chance)
            {
                best_config.hit_chance = test_hit_chance;
                best_config.first_cast_position = potential_first;
                best_config.cast_position = potential_second;
                best_config.physics_prob = physics_prob;
                best_config.behavior_prob = behavior_prob;
            }
        }

        // Fallback: If no valid angle found (e.g. target out of range), aim directly at them
        if (best_config.hit_chance < EPSILON)
        {
            math::vector3 to_target = predicted_target_pos - source_pos;
            float dist = magnitude_2d(to_target);  // 2D - High Ground Fix
            // FIX: Must use 2D direction for normalization (cannot divide 3D vector by 2D magnitude!)
            math::vector3 to_target_2d = flatten_2d(to_target);
            math::vector3 dir = (dist > EPSILON) ? to_target_2d / dist : math::vector3(1, 0, 0);

            best_config.first_cast_position = source_pos + dir * std::min(max_first_cast_range, dist);
            best_config.first_cast_position.y = predicted_target_pos.y;  // Preserve target Y coordinate
            best_config.cast_position = best_config.first_cast_position + dir * vector_length;
            best_config.hit_chance = 0.05f; // Low confidence fallback
            best_config.physics_prob = 0.05f;
            best_config.behavior_prob = 1.0f;
        }

        return best_config;
    }

    // =========================================================================
    // PREDICTION MANAGER IMPLEMENTATION
    // =========================================================================

    void PredictionManager::update()
    {
        // MASTER TRY-CATCH: Prevent any crash from update
        try
        {
            // CRASH FIX: Add null checks for g_sdk subsystems
            if (!g_sdk || !g_sdk->clock_facade || !g_sdk->object_manager)
                return;

            float current_time = g_sdk->clock_facade->get_game_time();

            // Update and clean up trackers in single pass
            // CRITICAL: Get fresh pointer from object_manager each frame to prevent dangling pointer
            for (auto it = trackers_.begin(); it != trackers_.end(); )
            {
                bool should_remove = false;

                if (!it->second)
                {
                    // Null tracker pointer - immediate removal
                    should_remove = true;
                }
                else
                {
                    // Get FRESH pointer from object manager using stored network_id
                    game_object* target = g_sdk->object_manager->get_object_by_network_id(it->first);

                    if (!target || !target->is_valid())
                    {
                        // Target no longer exists or invalid (died, recalled, DC'd)
                        // Keep tracker for TRACKER_TIMEOUT in case they respawn/reconnect
                        const auto& history = it->second->get_history();
                        if (history.empty())
                        {
                            should_remove = true;  // Never collected data
                        }
                        else
                        {
                            float time_since_last = current_time - history.back().timestamp;
                            if (time_since_last > TRACKER_TIMEOUT)
                            {
                                should_remove = true;
                            }
                        }
                    }
                    else
                    {
                        // Target is valid - pass FRESH pointer to tracker update
                        // This prevents using dangling pointer stored in tracker
                        it->second->update(target);
                        should_remove = false;
                    }
                }

                if (should_remove)
                {
                    it = trackers_.erase(it);
                }
                else
                {
                    ++it;
                }
            }

            last_update_time_ = current_time;

            // Self physics measurement removed - calibration phase complete
        } // End master try
        catch (...) { /* Prevent any crash from update */ }
    }

    void PredictionManager::measure_self_physics()
    {
        // SELF-MEASUREMENT: Track local player to test your own acceleration/deceleration
        // Move from standstill, then stop suddenly to generate measurements
        if (!g_sdk || !g_sdk->object_manager || !g_sdk->clock_facade)
            return;

        game_object* local_player = g_sdk->object_manager->get_local_player();
        if (!local_player || !local_player->is_valid())
            return;

        float current_time = g_sdk->clock_facade->get_game_time();
        math::vector3 current_pos = local_player->get_position();

        // Calculate velocity from position delta
        float dt = current_time - self_last_time_;
        if (dt > 0.01f && dt < 0.2f && self_last_time_ > 0.f)
        {
            math::vector3 delta = current_pos - self_last_pos_;
            float current_speed = delta.magnitude() / dt;

            float speed_change = current_speed - self_last_speed_;

            // Detect acceleration (speed increasing significantly)
            if (speed_change > 50.f && current_speed > 50.f)
            {
                float accel = speed_change / dt;
                char log_msg[256];
                snprintf(log_msg, sizeof(log_msg),
                    "[SELF PHYSICS] ACCEL: %.0f -> %.0f in %.3fs = %.0f units/s^2",
                    self_last_speed_, current_speed, dt, accel);
                g_sdk->log_console(log_msg);
            }

            // Detect deceleration (speed decreasing significantly)
            if (speed_change < -50.f && self_last_speed_ > 50.f)
            {
                float decel = -speed_change / dt;
                char log_msg[256];
                snprintf(log_msg, sizeof(log_msg),
                    "[SELF PHYSICS] DECEL: %.0f -> %.0f in %.3fs = %.0f units/s^2",
                    self_last_speed_, current_speed, dt, decel);
                g_sdk->log_console(log_msg);
            }

            self_last_speed_ = current_speed;
        }

        self_last_pos_ = current_pos;
        self_last_time_ = current_time;
    }

    TargetBehaviorTracker* PredictionManager::get_tracker(game_object* target)
    {
        if (!target || !target->is_valid())
            return nullptr;

        uint32_t network_id = target->get_network_id();

        auto it = trackers_.find(network_id);
        if (it != trackers_.end())
        {
            return it->second.get();
        }

        // Create new tracker
        auto tracker = std::make_unique<TargetBehaviorTracker>(target);
        auto* tracker_ptr = tracker.get();
        trackers_[network_id] = std::move(tracker);

        return tracker_ptr;
    }

    HybridPredictionResult PredictionManager::predict(
        game_object* source,
        game_object* target,
        const pred_sdk::spell_data& spell)
    {
        try
        {
            auto* tracker = get_tracker(target);
            if (!tracker)
            {
                HybridPredictionResult result;
                result.is_valid = false;
                return result;
            }

            return HybridFusionEngine::compute_hybrid_prediction(source, target, spell, *tracker);
        }
        catch (...)
        {
            // Prevent any crash from prediction
            HybridPredictionResult result;
            result.is_valid = false;
            return result;
        }
    }

    void PredictionManager::clear()
    {
        trackers_.clear();
    }

} // namespace HybridPred