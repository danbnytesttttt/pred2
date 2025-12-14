# Path Stability Implementation - Priority 1

## What Was Implemented

**Core Philosophy**: Treat path stability as a **probability feature**, not a binary requirement. No hard gates, just intelligent confidence adjustment.

---

## Components

### 1. **PathStability.h** (New File)

**IntentSignature** - Robust path change detection
```cpp
struct IntentSignature
{
    math::vector3 first_segment_dir;   // Direction of first meaningful segment
    math::vector3 short_horizon_pos;   // Predicted position at 0.2s

    bool has_changed_meaningfully(prev);  // Hysteresis-based change detection
};
```

**Hysteresis thresholds:**
- Small change: `<25° direction, <30u drift` → Same intent
- Big change: `>50° direction, >80u drift` → New intent
- Medium change: Stay with previous decision (prevents flapping)

---

**TargetBehaviorTracker** - Per-target state machine
```cpp
class TargetBehaviorTracker
{
    IntentSignature current_intent;
    float time_since_meaningful_change;

    // Windup state machine
    bool was_in_windup;
    IntentSignature pre_windup_intent;

    void update(target, current_time);  // Main update loop
};
```

**Windup-aware damping:**
```cpp
if (in_windup_now)
{
    // Don't freeze, don't ramp aggressively
    time_stable += delta_time * 0.3f;  // 30% normal rate
}

if (just_exited_windup)
{
    // Reset only if direction changed meaningfully
    if (current_intent.has_changed_meaningfully(pre_windup_intent))
        time_stable = 0.f;
}
```

---

### 2. **Persistence Scoring** (Knee-Curve)

```cpp
float compute_persistence(float time_stable, float t_impact)
{
    // Required stability scales with t_impact
    float required = clamp(0.6 * t_impact, 0.10, 0.40);

    float ratio = clamp(time_stable / required, 0, 1);

    // Knee-curve: harsh early penalty, fast reward near stable
    return smoothstep(0.55, 0.95, ratio);
}
```

**Why knee-curve?**
- Early stability (10% of required) is almost meaningless
- Once you cross ~60% threshold, confidence rapidly increases
- Better than linear (early stability matters less)

---

### 3. **Contextual Floor** (Fast vs Slow Spells)

```cpp
float apply_calibrator(float base_hc, float persistence, float t_impact)
{
    float floor = (t_impact < 0.25f) ? 0.65f : 0.45f;
    return base_hc * (floor + (1 - floor) * persistence);
}
```

**Reasoning:**
- **Fast spells** (`<0.25s`): Path changes matter less → higher floor (0.65)
- **Slow spells** (`≥0.25s`): Path changes matter more → lower floor (0.45)

**Example:**
```
Base HC: 0.90 (90% geometric confidence)
Persistence: 0.0 (path just changed)
t_impact: 0.5s (slow spell)

Calibrated: 0.90 * (0.45 + 0.55 * 0.0) = 0.90 * 0.45 = 0.405 (40.5%)

→ Geometric says 90%, but path just changed → drop to 40.5%
→ Wait for path to stabilize before casting
```

---

### 4. **Enhanced Telemetry** (A/B Testing)

**New PredictionEvent fields:**
```cpp
// Core metrics
float baseline_hc;              // Before path stability
float persistence;              // Stability score [0, 1]
float calibrated_hc;            // After path stability

// Change detection
float time_stable;              // Time since meaningful change
float delta_theta;              // Direction change (radians)
float delta_short_horizon;      // Position drift (units)

// Decision tracking
bool baseline_would_cast;       // Old policy decision (at 75% threshold)
bool new_would_cast;            // New policy decision (at 75% threshold)
```

**Why log both decisions?**
- Enables offline A/B analysis
- Can measure: bad casts avoided, good casts missed, net improvement
- Prevents "fooling yourself" with just correlation metrics

---

## Integration Points

### **GeometricPrediction.h** (lines 1707-1742)

After computing base hit chance:
```cpp
// Base geometric prediction
result.hit_chance_float = reaction_window_to_confidence(reaction_window);

// PATH STABILITY CALIBRATION
float baseline_hc = result.hit_chance_float;
float persistence = PathStability::update_and_get_persistence(
    input.target,
    time_to_impact,
    current_time
);
float calibrated_hc = PathStability::apply_calibrator(
    baseline_hc,
    persistence,
    time_to_impact
);

// Use calibrated for decisions
result.hit_chance_float = calibrated_hc;

// Store both for telemetry
result.baseline_hc = baseline_hc;
result.persistence = persistence;
result.calibrated_hc = calibrated_hc;
```

### **Telemetry Logging** (lines 1812-1831)

```cpp
// Log path stability data
event.baseline_hc = result.baseline_hc;
event.persistence = result.persistence;
event.calibrated_hc = result.calibrated_hc;

// Get detailed metrics from tracker
auto* tracker = PathStability::get_tracker(input.target);
if (tracker)
{
    event.time_stable = tracker->time_since_meaningful_change;
    event.delta_theta = tracker->get_delta_theta();
    event.delta_short_horizon = tracker->get_delta_short_horizon();
}

// Decision tracking (A/B at 75% threshold)
event.baseline_would_cast = (result.baseline_hc >= 0.75f);
event.new_would_cast = (result.calibrated_hc >= 0.75f);
```

---

## How To Evaluate

### **Step 1: Collect Data**

Play 10+ games with telemetry enabled. System will log every prediction with both baseline and new decisions.

### **Step 2: Export Telemetry**

```cpp
// In your champion script, on game end:
PredictionTelemetry::export_to_csv("path_stability_eval.csv");
```

### **Step 3: Analyze**

```python
import pandas as pd

df = pd.read_csv("path_stability_eval.csv")

# Filter to reasonable predictions (exclude extreme cases)
df = df[(df.t_impact >= 0.2) & (df.t_impact <= 1.5)]

# Bucket by t_impact
short = df[df.t_impact < 0.3]    # Fast spells
medium = df[(df.t_impact >= 0.3) & (df.t_impact < 0.6)]
long = df[df.t_impact >= 0.6]     # Slow spells

# Analysis buckets
def analyze_bucket(bucket, name):
    only_baseline = bucket[(bucket.baseline_would_cast) & (~bucket.new_would_cast)]
    only_new = bucket[(~bucket.baseline_would_cast) & (bucket.new_would_cast)]
    both = bucket[(bucket.baseline_would_cast) & (bucket.new_would_cast)]

    print(f"\n{name} Spells:")
    print(f"  Baseline-only casts: {len(only_baseline)}")
    print(f"  Baseline-only hit rate: {only_baseline.was_hit.mean():.2%}")

    print(f"  New-only casts: {len(only_new)}")
    print(f"  New-only hit rate: {only_new.was_hit.mean():.2%}")

    print(f"  Agreement casts: {len(both)}")
    print(f"  Agreement hit rate: {both.was_hit.mean():.2%}")

    # Net calculation
    baseline_total_hits = only_baseline.was_hit.sum() + both.was_hit.sum()
    new_total_hits = only_new.was_hit.sum() + both.was_hit.sum()

    print(f"  Net change: {new_total_hits - baseline_total_hits:.1f} hits")

analyze_bucket(short, "Short")
analyze_bucket(medium, "Medium")
analyze_bucket(long, "Long")

# PATH_CHANGED reduction
path_changed = df[df.miss_reason == 'PATH_CHANGED']
baseline_path_changed = path_changed[path_changed.baseline_would_cast]
new_path_changed = path_changed[path_changed.new_would_cast]

print(f"\nPATH_CHANGED misses:")
print(f"  Baseline: {len(baseline_path_changed)}")
print(f"  New: {len(new_path_changed)}")
print(f"  Reduction: {len(baseline_path_changed) - len(new_path_changed)}")
```

### **Step 4: Decision Criteria**

✅ **Ship if:**
- Net improvement > 0 across all buckets
- Long t_impact shows >5% improvement
- Short t_impact doesn't regress
- PATH_CHANGED reduction > 10%
- Average delay added < 30ms

❌ **Revert if:**
- Net improvement ≤ 0
- Short t_impact regresses
- Good casts missed > bad casts avoided

---

## Expected Behavior

### **Scenario A: Target Pathing Straight for 1.5s**
```
time_stable: 1.5s
t_impact: 0.6s
required: 0.6 * 0.6 = 0.36s
ratio: 1.5 / 0.36 = 4.17 (clamped to 1.0)
persistence: smoothstep(0.55, 0.95, 1.0) = 1.0

baseHC: 0.80
floor: 0.45 (slow spell)
calibratedHC: 0.80 * (0.45 + 0.55 * 1.0) = 0.80 * 1.0 = 0.80

→ No penalty! Path is very stable.
```

### **Scenario B: Target Just Changed Direction**
```
time_stable: 0.05s
t_impact: 0.6s
required: 0.36s
ratio: 0.05 / 0.36 = 0.139
persistence: smoothstep(0.55, 0.95, 0.139) ≈ 0.0

baseHC: 0.80
floor: 0.45
calibratedHC: 0.80 * (0.45 + 0.55 * 0.0) = 0.80 * 0.45 = 0.36

→ Large penalty! Path just changed, don't trust it yet.
```

### **Scenario C: Fast Spell, Path Just Changed**
```
time_stable: 0.05s
t_impact: 0.2s
required: 0.6 * 0.2 = 0.12s
ratio: 0.05 / 0.12 = 0.417
persistence: smoothstep(0.55, 0.95, 0.417) ≈ 0.0

baseHC: 0.80
floor: 0.65 (fast spell - higher floor!)
calibratedHC: 0.80 * (0.65 + 0.35 * 0.0) = 0.80 * 0.65 = 0.52

→ Smaller penalty than slow spell. Fast spells are less sensitive.
```

---

## Known Limitations (Priority 1)

**What we DON'T have yet:**
1. Destination affinity (tower, fountain = higher persistence)
2. Allied pressure (enemy fleeing = constrained path)
3. Terrain constraints (corridors = fewer options)
4. Volatility tracking (recent click rate history)
5. Asymmetric oscillation detection (kiting)

**Why not included:**
- Priority 1 is minimal validation
- Need to prove core concept works before adding complexity
- Each feature adds magic numbers that need tuning

**If Priority 1 succeeds:** Add Priority 2 features with same A/B methodology.

---

## Magic Numbers to Watch

If evaluation shows good results but needs tuning:

```cpp
// Hysteresis (PathStability.h:71-74)
ANGLE_SMALL = 25°   // Increase to be more permissive
ANGLE_BIG = 50°     // Decrease to be stricter

// Short horizon (PathStability.h:182)
SHORT_HORIZON = 0.2s  // Increase for longer prediction window

// Required stability (PathStability.h:278)
required = 0.6 * t_impact  // Decrease to 0.5 for more lenient
                           // Increase to 0.7 for stricter

// Knee curve (PathStability.h:287)
smoothstep(0.55, 0.95, ratio)  // Adjust edges for different penalty curves

// Contextual floor (PathStability.h:302-303)
fast: 0.65  // Increase to be more permissive on fast spells
slow: 0.45  // Decrease to be stricter on slow spells
```

---

## Summary

**What's new:**
- Path stability tracking per target
- Windup-aware state machine
- Knee-curve persistence scoring
- Contextual floor (fast vs slow)
- A/B telemetry logging

**What's the goal:**
- Avoid casting on fresh paths that will change
- Don't miss good opportunities by being too conservative
- Measure: net improvement via A/B analysis

**Next step:**
- Collect data from real games
- Run analysis script
- If net > 0: ship it
- If net ≤ 0: revert and reconsider approach

**Files modified:**
- `PathStability.h` (new)
- `PredictionTelemetry.h` (8 new fields)
- `GeometricPrediction.h` (integration + logging)

Ready to test! 🚀
