# Critical Analysis: Path Finding & Prediction

## Executive Summary

**Current Approach**: Single-path linear interpolation with heuristic adjustments
**Intelligence Level**: Medium - uses SDK data but makes arbitrary assumptions
**Behavior Consideration**: Yes (HybridPrediction) - tracks dodge patterns, movement history
**Multiple Paths**: No - only considers one path from SDK

---

## 1. CURRENT PATH PREDICTION LOGIC

### A. Core Algorithm (GeometricPrediction.h:850-933 & HybridPrediction.cpp:1957-2097)

```cpp
// WHAT IT DOES:
1. Get path from SDK: target->get_path()
2. Calculate distance to travel: move_speed * time
3. Apply heuristics:
   - Path staleness reduction (if animation lock > 200ms)
   - Start-of-path dampening (85-100% speed ramp)
   - End-of-path clamping (don't overshoot)
4. Iterate through waypoints linearly
5. Return interpolated position
```

**Strengths**:
- ✅ Uses authoritative SDK path data
- ✅ Handles multi-waypoint paths correctly
- ✅ Wall collision check (is_pathable)
- ✅ CC detection (immobilized = stay at current position)

**Weaknesses**:
- ❌ Constant velocity assumption (ignores acceleration/deceleration)
- ❌ Arbitrary heuristics (85% dampening, 200ms staleness)
- ❌ No turn rate modeling (assumes instant direction changes)
- ❌ Single path only (no alternates considered)

---

## 2. HEURISTICS ANALYSIS

### A. Start-of-Path Dampening (HybridPrediction.cpp:2030-2038)

```cpp
// HEURISTIC 1: Start-of-Path Dampening
if (path_age < 0.1f)
{
    // Linear ramp: 0ms=0.85x speed, 100ms=1.0x speed
    speed_multiplier = 0.85f + 0.15f * (path_age / 0.1f);
}
```

**CRITIQUE**:
- **Why 85%?** Arbitrary. League accelerates to max speed in ~23ms.
- **Why 100ms window?** Said to be "conservative" but no data backing this.
- **Intelligence improvement**: Use actual acceleration from SDK or measure it per champion.

**QUESTION**: Does SDK provide `get_acceleration()` or `get_actual_velocity()`?

### B. Path Staleness (GeometricPrediction.h:872-878)

```cpp
// PATH STALENESS: Long locks make paths unreliable
if (animation_lock_time > PATH_STALE_THRESHOLD)  // 200ms
{
    float staleness_factor = 1.0f - std::min(
        (animation_lock_time - PATH_STALE_THRESHOLD) / PATH_STALE_RANGE,  // 400ms
        PATH_STALE_MAX_REDUCTION);  // Max 50% reduction
    effective_movement_time *= staleness_factor;
}
```

**CRITIQUE**:
- **Assumption**: Longer animation locks = more likely to issue new movement command
- **Problem**: This is behavioral guessing, not based on actual path age
- **Better approach**: Track actual time since `get_path()` changed (HybridPrediction does this!)

**RECOMMENDATION**: Remove this arbitrary heuristic. Use `path_age` from tracker instead.

### C. End-of-Path Clamping (GeometricPrediction.h:886-896)

```cpp
// End-of-Path Clamping (don't overshoot destination)
float remaining_path = calculate_total_path_distance();
if (distance_to_travel > remaining_path)
    distance_to_travel = remaining_path;
```

**CRITIQUE**:
- ✅ **This is GOOD** - prevents predicting beyond destination
- ✅ **Mathematically sound** - no arbitrary constants
- ✅ **Keep this**

---

## 3. MULTIPLE PATHS CONSIDERATION

**CURRENT**: Only uses `target->get_path()` - single path

**QUESTION**: Should we consider multiple paths?

### Analysis:

**PRO Arguments** (for multiple paths):
1. Target might change direction at any waypoint
2. Behavioral uncertainty - they could turn around
3. Multiple destinations possible (fleeing vs chasing)

**CON Arguments** (against multiple paths):
1. SDK path IS the target's clicked path - this is their actual intent
2. League doesn't pathfind during movement - they follow clicked path
3. Path changes are behavioral (tracked separately by BehaviorPDF)
4. Multiple paths = exponential complexity with no clear benefit

**VERDICT**: ❌ Don't implement multiple geometric paths.

**BETTER APPROACH**:
- Physics (path-following) = deterministic prediction
- Behavior (PDF) = probabilistic direction changes
- This separation is **already implemented** in HybridPrediction!

---

## 4. BEHAVIOR PREDICTION ANALYSIS

### A. Current Implementation (HybridPrediction.h:669-798)

```cpp
class TargetBehaviorTracker
{
    - Tracks last 100 movement snapshots (sampled every 50ms)
    - Builds dodge pattern statistics (left/right/forward/backward frequencies)
    - Learns reaction delays, juke patterns, animation cancel timing
    - Builds 32x32 grid-based probability density function (PDF)
    - Exponential decay: recent movements weighted more (0.95^i)
}
```

**Strengths**:
- ✅ Captures actual player behavior over time
- ✅ Exponential decay is mathematically sound
- ✅ Per-target learning (different players = different patterns)
- ✅ PDF approach handles spatial uncertainty properly

**Weaknesses**:
- ❌ Requires 35+ samples (1.75s observation) before trusting patterns
- ❌ No context awareness (CS patterns commented out as TODO)
- ❌ Fixed 32x32 grid (25u cells) - could be dynamic based on distance
- ❌ No champion-specific defaults (Kalista vs Darius move very differently)

### B. Physics + Behavior Fusion (HybridPrediction.h:307-416)

```cpp
// Weighted geometric mean: fused = physics^w × behavior^(1-w)
// Default: 60% physics, 40% behavior
// Adjusts based on:
// - Sample count (fewer samples = trust physics more)
// - Move speed (fast targets = trust physics more)
// - Staleness (old data = trust physics more)
// - Distance (close range = trust physics more)
```

**CRITIQUE**:
- **Arbitrary weights**: Why 60/40? Why not 70/30 or 50/50?
- **No empirical backing**: These ratios are guesses, not measured
- **Better approach**: Learn optimal fusion weight per target
  - Track: when we predicted X, what actually happened?
  - Adjust: if physics was consistently closer, increase physics weight
  - Personalized: fast dodgers = more behavior weight

**RECOMMENDATION**: Add adaptive fusion weight learning.

---

## 5. MISSING INTELLIGENT FEATURES

### A. Turn Rate Modeling

**Problem**: Assumes instant direction changes at waypoints

```cpp
// CURRENT: Sharp corner from North to East
//   Path: [current_pos, waypoint_north, waypoint_east]
//   Prediction: Follows path exactly with instant 90° turn

// REALITY: League has turn rate limitations
//   Target will slightly overshoot the waypoint before turning
//   The sharper the turn, the more overshoot
```

**Intelligence improvement**:
- Calculate angle between path segments
- For sharp turns (>90°), add overshoot distance
- Formula: `overshoot = move_speed * turn_time` where `turn_time = angle / turn_rate`

**QUESTION**: Does SDK expose turn rate? Or can we measure it?

### B. Acceleration Modeling

**Problem**: Start-of-path dampening uses arbitrary 85% multiplier

**Better approach**:
- Use `target->get_velocity()` from SDK (actual current velocity)
- Predict: `position = current_pos + current_velocity * t + 0.5 * acceleration * t^2`
- Get acceleration from SDK or measure it (track velocity deltas)

**QUESTION**: Does `game_object` have `get_velocity()` or `get_actual_velocity()`?

### C. Context-Aware Prediction

**What's missing**:
1. **Minion wave positioning** (ADCs path toward CS opportunities)
2. **Jungle camp timers** (junglers have predictable pathing)
3. **Tower aggro** (forces movement away from tower)
4. **Terrain chokepoints** (limits possible paths in jungle)
5. **Ability cooldowns** (Flash available = higher dodge probability)
6. **HP pressure** (low HP = more defensive pathing)
7. **Game time** (early game = more predictable, late game = more dodging)

**Current status**:
- ✅ CS patterns declared but commented as TODO (line 551-554)
- ❌ Everything else not implemented

**RECOMMENDATION**:
1. **Priority 1**: Flash/dash cooldown tracking (high impact)
2. **Priority 2**: Minion wave CS prediction (ADC-specific)
3. **Priority 3**: HP pressure modifier (low HP = defensive bias)
4. **Priority 4**: Terrain-aware path filtering (jungle only)

### D. Champion-Specific Behavior Models

**Problem**: All champions treated equally in behavior tracking

**Reality**:
- Kalista: Always hopping (mini-dashes after AAs)
- Yasuo: Dashes through minions frequently
- Ezreal: Has E blink on short CD
- Darius: Slow, telegraphed movement
- Master Yi: Has Q untargetability
- Lucian: Has E dash on short CD

**Intelligence improvement**:
- Initialize `DodgePattern` with champion-specific defaults
- Kalista starts with `linear_continuation_prob = 0.3` (low, because hops)
- Darius starts with `linear_continuation_prob = 0.95` (high, predictable)
- Adjust learning rate based on champion mobility class

### E. Adaptive Learning Rate

**Problem**: Fixed exponential decay (0.95^i) for all scenarios

**Better approach**:
- Variable decay based on context:
  - Recent path change: decay faster (0.90^i) - prioritize new data
  - Stable path: decay slower (0.97^i) - smooth noise
  - Target emerging from fog: reset/ignore old samples

---

## 6. SPECIFIC ISSUES FOUND

### Issue #1: Duplicate Path Distance Calculation

**Location**: HybridPrediction.cpp:2042-2057

```cpp
// Calculated TWICE in same function:
// 1. Lines 2042-2051: Calculate remaining_path
// 2. Used to clamp distance_to_travel
// 3. Then SAME calculation happens in iteration loop (2063-2092)
```

**Fix**: Calculate once, store in variable. Minor performance waste.

### Issue #2: Wall Collision Returns Segment Start

**Location**: GeometricPrediction.h:920-923 & HybridPrediction.cpp:2082-2087

```cpp
if (!g_sdk->nav_mesh->is_pathable(predicted_pos))
{
    return segment_start;  // Clamp to safe position
}
```

**CRITIQUE**:
- **Problem**: If predicted position is in wall, returns START of segment
- **Why problematic**: Target might have traveled 90% of segment before wall
- **Better**: Binary search along segment to find last pathable point

### Issue #3: Zero-Length Segment Skip

**Location**: GeometricPrediction.h:908-909

```cpp
if (segment_length < 0.001f)
    continue;  // Skip zero-length segments
```

**CRITIQUE**:
- ✅ Good defensive check
- ❓ Why 0.001f specifically? Arbitrary constant
- **Better**: Use named constant `EPSILON` or `MIN_SEGMENT_LENGTH`

### Issue #4: Server Position vs Client Position

**Location**: HybridPrediction.cpp:1971

```cpp
// Use server position (authoritative for hit detection, avoids 30-100ms client lag)
math::vector3 position = target->get_server_position();
```

**CRITIQUE**:
- ✅ **This is CORRECT** - server position is authoritative
- ✅ Avoids client lag issues
- ✅ Keep this

---

## 7. RECOMMENDATIONS

### Priority 1: Remove Arbitrary Heuristics

**Remove**:
1. ❌ Start-of-path dampening (85% multiplier) - use actual velocity from SDK
2. ❌ Path staleness based on animation lock - use actual path age

**Keep**:
1. ✅ End-of-path clamping (mathematically sound)
2. ✅ Wall collision check
3. ✅ CC detection

### Priority 2: Use SDK Data Instead of Guessing

**Replace**:
- `move_speed * time` → `current_velocity + acceleration * time`
- `speed_multiplier = 0.85f` → Use actual `get_velocity()` magnitude

**Add**:
- Turn rate modeling (if SDK exposes it)
- Actual path age tracking (already in HybridPrediction tracker!)

### Priority 3: Improve Behavior Fusion

**Current**: Fixed 60/40 physics/behavior split
**Better**: Adaptive learning
- Track prediction accuracy per target
- Adjust fusion weights based on which component was more accurate
- Personalized per target

### Priority 4: Context-Aware Prediction

**High Impact**:
1. Flash/dash cooldown tracking → modify dodge probability
2. HP pressure → defensive/aggressive pathing bias
3. Minion wave positioning → CS prediction for ADCs

**Medium Impact**:
4. Tower aggro detection → forces movement away
5. Terrain chokepoints → limit reachable region in jungle

### Priority 5: Champion-Specific Models

- Initialize `DodgePattern` with champion-appropriate defaults
- Adjust learning rates for high-mobility champions
- Special cases: Kalista (hop mechanics), Yasuo (dash through minions)

---

## 8. QUESTIONS FOR USER

1. **Acceleration**: Does SDK provide `get_velocity()` or `get_acceleration()`?
2. **Turn rate**: Can we access turn rate data from SDK?
3. **Path age**: Should we use HybridPrediction's `path_age` tracking everywhere?
4. **Multiple paths**: Do you want "what-if" scenarios (e.g., what if they turn around)?
5. **Behavior weight**: Should we implement adaptive fusion weight learning?
6. **Context features**: Which context (CS, HP, Flash CD, etc.) should we prioritize?
7. **Champion models**: Should we hardcode champion-specific behavior defaults?

---

## 9. CODE QUALITY ISSUES

1. **Magic numbers**: 0.85, 0.001f, 200ms, 400ms - should be named constants
2. **Duplicate code**: Path distance calculated twice in predict_on_path
3. **Wall collision**: Returns segment_start instead of finding last pathable point
4. **Comments**: Start-of-path says "23ms to max speed" but uses 100ms window

---

## 10. FINAL VERDICT

**Current Implementation**: 6/10
- ✅ Solid foundation with SDK path following
- ✅ Good separation of physics vs behavior
- ✅ Server position usage is correct
- ❌ Too many arbitrary heuristics
- ❌ Not using available SDK data fully
- ❌ Missing intelligent context features

**Path Forward**:
1. **Remove arbitrary constants** - use SDK data
2. **Add acceleration/velocity** - more accurate physics
3. **Implement context features** - Flash CD, HP, CS patterns
4. **Adaptive fusion weights** - learn optimal physics/behavior balance
5. **Champion-specific models** - don't treat Kalista like Darius
