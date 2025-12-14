# Deterministic Bug Audit & Fixes

**Date**: 2025-12-14
**Scope**: PathStability.h, PredictionTelemetry.h, GeometricPrediction.h
**Method**: Static analysis, no gameplay required

---

## P0 BUGS (Data Corruption / Silent Failure)

### P0-1: Direction Vector Normalization Threshold Bug

**Location**: `PathStability.h:295-300`

**Current Code**:
```cpp
math::vector3 dir = path[i] - current_pos;
float mag = std::sqrt(dir.x * dir.x + dir.z * dir.z);
if (mag > 1.f)  // ← BUG: Wrong threshold
{
    sig.first_segment_dir = math::vector3(dir.x / mag, 0.f, dir.z / mag);
    sig.is_valid = true;
}
```

**Failure Mode**:
- Path segment with 3D distance >50 units but XZ projection <1 unit (e.g., steep ramp, vertical gap)
- Example: segment at (0.5, 100, 0.5) from origin has `seg_length = 100` but `mag = 0.707`
- Condition `mag > 1.f` fails → no normalization → intent stays invalid
- OR if we later set it from velocity, we get non-unit vector that breaks dot product math

**Fix**:
```cpp
math::vector3 dir = path[i] - current_pos;
float mag = std::sqrt(dir.x * dir.x + dir.z * dir.z);
if (mag > 0.01f)  // ← FIX: Just avoid div-by-zero, don't re-threshold
{
    sig.first_segment_dir = math::vector3(dir.x / mag, 0.f, dir.z / mag);
    sig.is_valid = true;
}
```

**Test Case**:
```cpp
// Scenario: Steep ramp segment (mostly vertical)
current_pos = (0, 0, 0)
path[0] = (0.5, 100, 0.5)  // 3D distance = 100, XZ mag = 0.707

Expected: sig.is_valid = true, first_segment_dir normalized
Actual (buggy): sig.is_valid = false (skipped due to mag < 1)
```

---

### P0-2: reference_intent Never Initialized on First Valid Intent

**Location**: `PathStability.h:215-266`

**Failure Mode**:
1. Tracker starts: all intents invalid, `time_stable = 0`
2. First update with valid intent: `current_intent.is_valid = true`, `reference_intent.is_valid = false`
3. `has_changed_meaningfully(prev, ref)` → prev invalid, returns false → no reset
4. Second update with 40° rotation (medium change): hysteresis says "no change", `reference_intent` still invalid
5. Cumulative drift check skipped (ref invalid) → stability accumulates despite rotation
6. **Bug**: reference_intent stays invalid for multiple frames, allowing early lock-in

**Current Code**:
```cpp
// Calculate current intent signature
previous_intent = current_intent;
current_intent = calculate_intent_signature(target, current_time);

// ← MISSING: Initialize reference on first valid intent

// Windup state machine
bool in_windup_now = target->is_winding_up() || target->is_channeling();
// ...

// Normal stability accumulation
if (current_intent.has_changed_meaningfully(previous_intent, reference_intent))
{
    time_since_meaningful_change = 0.f;
    if (current_intent.is_valid)
        reference_intent = current_intent;  // ← Only set here (too late)
}
```

**Fix**:
```cpp
// Calculate current intent signature
previous_intent = current_intent;
current_intent = calculate_intent_signature(target, current_time);

// ← FIX: Initialize reference on first valid intent
if (current_intent.is_valid && !reference_intent.is_valid)
{
    reference_intent = current_intent;
}

// Windup state machine
bool in_windup_now = target->is_winding_up() || target->is_channeling();
// ... rest of logic
```

**Test Case**:
```cpp
// Scenario: First valid intent, then medium rotation
Frame 0: target invalid → all intents invalid
Frame 1 (t=0.05): target valid, direction 0° → current valid, reference SHOULD BE SET
Frame 2 (t=0.10): direction 35° (medium) → hysteresis says "no change"
Frame 3 (t=0.15): direction 70° (cumulative big) → SHOULD reset via cumulative check

Expected: Reset at Frame 3 due to cumulative drift (70° > 50°)
Actual (buggy): No reset (reference invalid, cumulative check skipped)
```

---

### P0-3: register_cast() Accepts prediction_id=0 (Invalid)

**Location**: `PredictionTelemetry.h:434-465`

**Failure Mode**:
- `log_prediction()` returns 0 when telemetry disabled or error
- Champion script calls `register_cast(0, ...)`
- Loop searching for `event.prediction_id == 0` never finds it (or finds wrong event if we log with id=0)
- Pending cast queued with `prediction_id = 0`
- `finalize_outcome()` searches for id=0, fails to find event
- Orphan pending cast: evaluated but never updates telemetry

**Current Code**:
```cpp
static void register_cast(
    uint64_t prediction_id,
    game_object* target,
    // ...
)
{
    if (!enabled_ || !target || !target->is_valid())  // ← Missing prediction_id check
        return;

    // Mark event as cast (may fail silently if id=0)
    for (auto& event : events_)
    {
        if (event.prediction_id == prediction_id)  // ← Never matches if id=0
        {
            event.did_actually_cast = true;
            break;
        }
    }

    // Queue pending cast (orphan if id=0)
    PendingCastEvaluation pending;
    pending.prediction_id = prediction_id;  // ← Stores invalid id
    // ...
    pending_casts_.push_back(pending);
}
```

**Fix**:
```cpp
static void register_cast(
    uint64_t prediction_id,
    game_object* target,
    // ...
)
{
    if (!enabled_ || !target || !target->is_valid() || prediction_id == 0)  // ← FIX
        return;

    // ... rest unchanged
}
```

**Test Case**:
```cpp
// Scenario: Telemetry disabled, script still calls register_cast
TelemetryLogger::initialize("TestChamp", false);  // disabled
uint64_t id = TelemetryLogger::log_prediction(event);  // returns 0
TelemetryLogger::register_cast(id, target, ...);  // ← Should no-op

Expected: No pending cast queued, no telemetry corruption
Actual (buggy): Pending cast queued with id=0, evaluated, never resolves
```

---

## P1 BUGS (Logic Flaws / Edge Cases)

### P1-1: Velocity Direction Threshold Conflates Movement and Normalization

**Location**: `PathStability.h:309-315`

**Issue**:
- Condition `if (vel_mag > 10.f)` serves two purposes:
  1. Movement threshold (only track intent if moving significantly)
  2. Normalization guard (avoid div-by-zero)
- These should be separate

**Current Code**:
```cpp
float vel_mag = std::sqrt(current_vel.x * current_vel.x + current_vel.z * current_vel.z);
if (vel_mag > 10.f)  // ← Movement threshold AND normalization guard
{
    sig.first_segment_dir = math::vector3(current_vel.x / vel_mag, 0.f, current_vel.z / vel_mag);
    sig.is_valid = true;
}
```

**Not Actually a Bug**: The 10.f threshold is intentional - we don't want to track intent for nearly-stationary targets (reduces noise). The normalization is safe because `vel_mag > 10` guarantees safe division.

**Keep as-is**: No fix needed.

---

### P1-2: Multi-Sample Can Take All Samples at Same Tick After Lag Spike

**Location**: `PredictionTelemetry.h:614-665`

**Behavior**:
- One sample per call to `evaluate_pending_outcomes()`
- If frame rate drops (lag spike), multiple sample times can pass
- All samples taken at consecutive ticks, not spread across intended times

**Example**:
```
Register cast at t=10.0, impact=10.5
Sample times: 10.45, 10.50, 10.55

Normal (60 FPS):
  Frame t=10.45: Take sample 0
  Frame t=10.467: Take sample 1  (17ms later)
  Frame t=10.483: Take sample 2  (16ms later)

After lag spike (200ms gap):
  Frame t=10.20: No samples yet
  Frame t=10.60: All 3 sample times have passed
    - Take sample 0, advance index to 1
  Frame t=10.617: Take sample 1
  Frame t=10.633: Take sample 2
  → All samples at t=10.6+ (late), not spread across 10.45-10.55
```

**Impact**: Samples collapse to nearly same time, reducing effectiveness of multi-sample noise reduction.

**This is acceptable** - incremental design prevents frame-rate dependency, and most of the time frames are regular. Alternative would batch samples after lag, which is worse.

**No fix required** - document as known limitation.

---

### P1-3: Windup Exit Logic Doesn't Update reference_intent if Direction Unchanged

**Location**: `PathStability.h:222-234`

**Scenario**:
```cpp
if (just_exited_windup)
{
    if (current_intent.has_changed_meaningfully(pre_windup_intent, reference_intent))
    {
        time_since_meaningful_change = 0.f;
        if (current_intent.is_valid)
            reference_intent = current_intent;
    }
    // ← If unchanged, reference_intent NOT updated
}
```

**Concern**: If target enters windup with direction A, exits with direction A (no change), but `reference_intent` is still from older direction B, future comparisons are against stale reference.

**Example**:
```
t=0.0s: Direction 0°, reference_intent = 0°
t=0.5s: Rotate to 40° (medium, no reset due to hysteresis)
t=0.6s: Enter windup, reference_intent still 0°
t=1.0s: Exit windup, direction 40° (same as pre_windup_intent)
  → No reset (correct)
  → reference_intent NOT updated (still 0°)
t=1.2s: Rotate to 75°
  → Cumulative drift: 75° - 0° = 75° > 50° → RESET

Expected: Reset at t=1.2s (correct behavior)
```

**Actually correct**: The reference_intent staying at 0° is intentional - it tracks "last reset point", not "last sampled direction". If we never reset during windup, reference shouldn't change.

**No fix required**.

---

## P2 ISSUES (Incomplete Features / Non-Critical)

### P2-1: time_in_current_windup Never Tracked

**Location**: `PredictionTelemetry.h:152` (field exists), `PathStability.h` (never set)

**Issue**: Telemetry field `time_in_current_windup` exists but is always 0.

**Fix** (if desired):
Add to `TargetBehaviorTracker`:
```cpp
float time_in_current_windup = 0.f;

// In update():
if (in_windup_now)
{
    time_in_current_windup += delta_time;
    time_since_meaningful_change += (delta_time * WINDUP_DAMPING);
}
else
{
    time_in_current_windup = 0.f;
    // normal logic
}
```

Then in telemetry logging:
```cpp
if (tracker)
{
    event.time_in_current_windup = tracker->time_in_current_windup;
}
```

**Priority**: P2 - nice to have for analysis, not critical.

---

### P2-2: Outcome Eviction Not Tracked

**Location**: `PredictionTelemetry.h:721-732` (finalize_outcome loop)

**Issue**: If event evicted from deque before outcome finalized, we silently lose the outcome. No counter tracks this.

**Fix**:
```cpp
// In finalize_outcome(), after loop:
bool event_found = false;
for (auto& event : events_)
{
    if (event.prediction_id == pending.prediction_id)
    {
        // ... update event
        event_found = true;
        break;
    }
}

if (!event_found)
{
    // Event evicted - increment counter for diagnostics
    stats_.outcomes_lost_to_eviction++;
}
```

**Priority**: P2 - diagnostic only, doesn't affect correctness.

---

## RANKED FIX LIST

### Must Fix (P0)
1. **P0-1**: Normalization threshold (`mag > 1.f` → `mag > 0.01f`)
2. **P0-2**: Initialize `reference_intent` on first valid intent
3. **P0-3**: Validate `prediction_id != 0` in `register_cast()`

### Should Review (P1)
4. **P1-2**: Document multi-sample lag spike behavior (no fix needed)

### Nice to Have (P2)
5. **P2-1**: Track `time_in_current_windup` (optional telemetry enhancement)
6. **P2-2**: Count evicted outcomes (diagnostic counter)

---

