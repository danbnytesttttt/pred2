# Skillshot Type Implementation Analysis

## Executive Summary

**Question**: Are all skillshot types (linear, circular, cone, vector) fully implemented with complete features?

**Answer**:
- ✅ **Linear (Capsule)**: FULLY IMPLEMENTED
- ✅ **Circular**: FULLY IMPLEMENTED
- ⚠️ **Cone**: PARTIALLY IMPLEMENTED - Missing escape distance calculation for single-target
- ⚠️ **Vector**: PARTIALLY IMPLEMENTED - Inconsistent integration between systems

---

## 1. LINEAR SKILLSHOTS (Capsule/Line)

### Status: ✅ **FULLY IMPLEMENTED**

### Single-Target Prediction:
- ✅ `point_in_capsule()` - GeometricPrediction.h:611
- ✅ `calculate_escape_distance_capsule()` - GeometricPrediction.h:686
- ✅ Minion collision detection - GeometricPrediction.h:434
- ✅ Main prediction routing - GeometricPrediction.h:1304-1321

### Multi-Target (AOE) Prediction:
- ✅ `predict_linear_aoe()` - CustomPredictionSDK.cpp:1290+
- ✅ Direction optimization (tests multiple angles)
- ✅ Hit target calculation with individual hit chances

### Examples:
- Morgana Q, Blitzcrank Hook, Xerath E, Lux Q, Nidalee Spear, Ezreal Q

### Implementation Quality:
**9/10** - Fully functional with all geometric calculations implemented correctly.

---

## 2. CIRCULAR SKILLSHOTS

### Status: ✅ **FULLY IMPLEMENTED**

### Single-Target Prediction:
- ✅ `calculate_escape_distance_circle()` - GeometricPrediction.h:593
- ✅ Main prediction routing - GeometricPrediction.h:1295-1302
- ✅ Circular AoE geometry

### Multi-Target (AOE) Prediction:
- ✅ `predict_aoe_cluster()` - CustomPredictionSDK.cpp:1091+
- ✅ `predict_aoe_circle()` - GeometricPrediction.h:1497+
- ✅ Centroid calculation for optimal multi-target hit position
- ✅ Priority weighting for high-value targets

### Examples:
- Annie R, Lux E, Ziggs Q/W/E/R, Orianna R, Veigar E, Cho'Gath W

### Implementation Quality:
**10/10** - Complete and well-implemented.

---

## 3. CONE SKILLSHOTS

### Status: ⚠️ **PARTIALLY IMPLEMENTED**

### What's Implemented:
- ✅ `point_in_cone()` - GeometricPrediction.h:650
  - Checks if point is within cone angle and range
  - Uses dot product for angle calculation
  - Mathematically correct implementation

- ✅ `predict_aoe_cone()` - GeometricPrediction.h:1815+
  - Tests multiple cast angles (24 samples default)
  - Checks which targets fall within cone
  - Returns optimal direction for maximum hits
  - Works correctly for multi-target

- ✅ Auto-detection in CustomPredictionSDK.cpp:1718-1730
  - Automatically detects cone spells via `get_cast_cone_angle()`
  - Routes to cone prediction

### What's MISSING:

#### ❌ **CRITICAL: Missing `calculate_escape_distance_cone()`**
**Location**: Should exist in GeometricPrediction.h Utils namespace, but doesn't

**Problem**: Single-target cone prediction has no way to calculate how far target must move to escape

**Impact**:
```cpp
// GeometricPrediction.h:1291-1321
// Distance to exit calculation (shape-dependent)
if (input.shape == SpellShape::Circle)
{
    distance_to_exit = Utils::calculate_escape_distance_circle(...);
}
else  // Capsule
{
    distance_to_exit = Utils::calculate_escape_distance_capsule(...);
}
// ❌ NO CONE CASE - Falls through to capsule calculation (WRONG!)
```

**Current behavior**: Cone spells are treated as capsules for escape distance
**Result**: Inaccurate hit chance calculation for cone spells

#### ❌ **Missing Main Prediction Routing**
**Location**: GeometricPrediction.h:1295-1321 (get_prediction function)

**Problem**: No `if (input.shape == SpellShape::Cone)` case
**Result**: Cone spells use capsule geometry instead of cone geometry

### Examples:
- Annie W, Cassiopeia Q, Rumble E, Cho'Gath W

### Required Fixes:

**Fix 1: Implement `calculate_escape_distance_cone()`**
```cpp
inline float calculate_escape_distance_cone(
    const math::vector3& target_pos,
    const math::vector3& cone_origin,
    const math::vector3& cone_direction,  // Normalized
    float cone_half_angle,                // In radians
    float cone_range,
    float target_radius)
{
    // Algorithm:
    // 1. Check if target is inside cone (point_in_cone)
    // 2. If inside, calculate shortest path OUT:
    //    - Either: Exit through cone edge (perpendicular to cone radius)
    //    - Or: Exit backward past cone origin
    //    - Or: Exit beyond cone range
    // 3. Return minimum escape distance

    // Use existing point_in_cone for inside check
    if (!point_in_cone(target_pos, cone_origin, cone_direction, cone_half_angle, cone_range))
        return 0.f;  // Already outside

    math::vector3 to_target = target_pos - cone_origin;
    float distance = magnitude_2d(to_target);

    // Option 1: Exit through side of cone
    // Distance to cone edge = distance * sin(actual_angle - cone_half_angle)
    float dot_product = to_target.dot(cone_direction);
    float cos_angle = dot_product / distance;
    float actual_angle = std::acos(cos_angle);
    float angle_diff = actual_angle - cone_half_angle;

    float side_exit = distance * std::sin(angle_diff) - target_radius;

    // Option 2: Exit backward (past cone origin)
    float backward_exit = distance - target_radius;

    // Option 3: Exit forward (beyond cone range)
    float forward_exit = (cone_range - distance) - target_radius;

    // Return shortest escape path
    return std::max(0.f, std::min({side_exit, backward_exit, forward_exit}));
}
```

**Fix 2: Add Cone Routing in get_prediction()**
```cpp
// GeometricPrediction.h:1295 - Replace:
if (input.shape == SpellShape::Circle)
{
    distance_to_exit = Utils::calculate_escape_distance_circle(...);
}
else  // Capsule
{
    distance_to_exit = Utils::calculate_escape_distance_capsule(...);
}

// With:
if (input.shape == SpellShape::Circle)
{
    distance_to_exit = Utils::calculate_escape_distance_circle(...);
}
else if (input.shape == SpellShape::Cone)
{
    math::vector3 spell_direction = (predicted_pos - input.source->get_position());
    float spell_length = Utils::magnitude_2d(spell_direction);
    if (spell_length > EPSILON)
        spell_direction = spell_direction / spell_length;
    else
        spell_direction = math::vector3(0.f, 0.f, 1.f);

    float cone_half_angle = input.spell_width / 2.f;  // spell_width stores cone angle
    distance_to_exit = Utils::calculate_escape_distance_cone(
        predicted_pos,
        input.source->get_position(),
        spell_direction,
        cone_half_angle,
        input.spell_range,
        target_radius
    );
}
else  // Capsule/Line
{
    distance_to_exit = Utils::calculate_escape_distance_capsule(...);
}
```

### Implementation Quality:
**5/10** - AOE works perfectly, but single-target prediction is broken due to missing escape distance calculation.

---

## 4. VECTOR SKILLSHOTS

### Status: ⚠️ **PARTIALLY IMPLEMENTED - INCONSISTENT**

### What's Implemented:

#### ✅ **Multi-Target AOE Prediction**
- `predict_aoe_vector()` - GeometricPrediction.h:1968+
- `predict_linear_aoe()` routing - CustomPredictionSDK.cpp:1743
- Optimizes both start and end positions
- Works correctly

#### ✅ **Range Calculation**
- CustomPredictionSDK.cpp:294-299
```cpp
// For vector spells, max hit range = cast_range + range
if (spell_data.spell_type == pred_sdk::spell_type::vector)
{
    effective_max_range = first_cast_range + spell_data.range + target_radius;
}
```

#### ✅ **HybridPrediction System**
- HybridPrediction.cpp:3004-3006 - `compute_vector_prediction()`
- Returns both `cast_position` (end point) and `first_cast_position` (start point)

### What's CONFUSING:

#### ⚠️ **Dual Type Systems**
**Problem**: Vector exists in TWO separate type systems:

**System 1: pred_sdk::spell_type** (pred_sdk.hpp:30-36)
```cpp
enum class spell_type: uint8_t
{
    linear = 0,
    targetted,
    circular,
    vector,  // ✅ Vector is here
};
```

**System 2: GeometricPred::SpellShape** (GeometricPrediction.h:77-83)
```cpp
enum class SpellShape
{
    Circle,
    Capsule,
    Line,
    Cone
    // ❌ No Vector
};
```

**Result**: Vector spells are a **spell_type** but not a **SpellShape**

#### ⚠️ **Routing Confusion**
**CustomPredictionSDK** knows about vector spells and handles them specially for:
- Range calculation
- AOE prediction routing
- Result population (first_cast_position)

**GeometricPrediction** has no concept of vector as a distinct shape for single-target prediction:
- No `SpellShape::Vector`
- Vector AOE prediction exists but is never called from single-target path
- Main `get_prediction()` function doesn't distinguish vector from linear

#### ⚠️ **Single-Target Prediction Gap**
**Question**: What happens when you call `GeometricPred::get_prediction()` with a vector spell?

**Answer**: You CAN'T specify vector shape - there's no `SpellShape::Vector`

**Current workaround**: Vector spells are treated as `SpellShape::Capsule` (linear)
- This is technically okay for hit detection (Viktor E laser is a line)
- But it doesn't optimize the two-position cast (where to place start and end)

**Actual vector optimization happens in**:
- HybridPrediction.cpp:compute_vector_prediction() - for full hybrid system
- GeometricPrediction.h:predict_aoe_vector() - for multi-target only

### Examples:
- Viktor E, Rumble R, Taliyah W, Irelia E

### Required Clarification:

**Question 1**: Should `SpellShape` include `Vector` as a 5th shape?
```cpp
enum class SpellShape
{
    Circle,
    Capsule,
    Line,
    Cone,
    Vector  // Add this?
};
```

**Question 2**: Should single-target `get_prediction()` handle vector spells?
- If YES: Need to add vector case that optimizes start+end positions
- If NO: Document that vector spells must use `predict_aoe_vector()` or HybridPrediction

**Current best practice** (based on code):
```cpp
// For vector spells, use AOE prediction even for single target
auto result = customPred.predict_aoe(spell_data, /*min_hits=*/1, /*min_hc=*/0.5);
// This will route to predict_linear_aoe which handles vector properly
```

### Implementation Quality:
**6/10** - Works, but requires understanding the dual type system. No clear single-target API.

---

## 5. SUMMARY TABLE

| Type     | Single-Target | Multi-Target (AOE) | Escape Distance | Point-in-Shape | Minion Collision | Overall |
|----------|---------------|-------------------|-----------------|----------------|------------------|---------|
| Linear   | ✅ Full       | ✅ Full           | ✅ Yes          | ✅ Yes         | ✅ Yes           | ✅ 10/10 |
| Circular | ✅ Full       | ✅ Full           | ✅ Yes          | ✅ Yes         | N/A              | ✅ 10/10 |
| Cone     | ❌ Broken     | ✅ Full           | ❌ NO           | ✅ Yes         | N/A              | ⚠️ 5/10  |
| Vector   | ⚠️ Unclear    | ✅ Full           | N/A             | N/A            | Same as Linear   | ⚠️ 6/10  |

---

## 6. CRITICAL BUGS

### Bug #1: Cone Escape Distance Calculation Missing
**Severity**: HIGH
**Impact**: Cone skillshots use capsule escape distance (WRONG geometry)
**Affected Spells**: Annie W, Cassiopeia Q, Rumble E, Cho'Gath W
**Result**: Inaccurate hit chance calculation for all cone spells

**Fix**: Implement `calculate_escape_distance_cone()` as shown above

### Bug #2: Cone Shape Not Routed in Main Prediction
**Severity**: HIGH
**Impact**: `get_prediction()` has no cone case, falls through to capsule
**Result**: Cone spells treated as linear skillshots

**Fix**: Add `if (input.shape == SpellShape::Cone)` case in get_prediction()

### Bug #3: Vector Shape Inconsistency
**Severity**: MEDIUM
**Impact**: Unclear how to use vector spells in single-target prediction
**Result**: Confusion, potential misuse

**Fix**: Either:
- Option A: Add `SpellShape::Vector` and implement single-target vector optimization
- Option B: Document that vector spells MUST use AOE prediction API

---

## 7. RECOMMENDED FIXES

### Priority 1: Fix Cone Geometry (Critical)
1. Implement `calculate_escape_distance_cone()` in GeometricPrediction.h Utils namespace
2. Add cone case in `get_prediction()` at line 1295
3. Test with cone spells (Annie W, Rumble E)

### Priority 2: Clarify Vector Usage (High)
**Option A (Full Implementation)**:
1. Add `SpellShape::Vector` to enum
2. Implement vector case in `get_prediction()`
3. Optimize both start and end positions for single-target

**Option B (Documentation)**:
1. Document in GeometricPrediction.h that vector spells should use AOE API
2. Add example usage in comments
3. Keep current dual-system approach

**Recommendation**: Choose **Option B** (documentation) because:
- Vector AOE prediction already works perfectly
- Single-target vector optimization is complex (two-position optimization)
- HybridPrediction already handles it for advanced users
- Most vector spells ARE multi-target (Viktor E, Rumble R)

### Priority 3: Add Cone Minion Collision (Medium)
Currently only capsule spells check minion collision. Cone spells should too.

---

## 8. TESTING CHECKLIST

### Linear Skillshots:
- [ ] Morgana Q - Test minion collision
- [ ] Blitzcrank Hook - Test escape distance calculation
- [ ] Xerath E - Test multi-target linear AOE

### Circular Skillshots:
- [ ] Annie R - Test escape distance for circular
- [ ] Orianna R - Test multi-target cluster optimization
- [ ] Veigar E - Test edge case (ring shape)

### Cone Skillshots:
- [ ] Annie W - Test after implementing escape_distance_cone
- [ ] Rumble E - Test angle calculation
- [ ] Cassiopeia Q - Test multi-target cone AOE

### Vector Skillshots:
- [ ] Viktor E - Test two-position optimization
- [ ] Rumble R - Test AOE prediction
- [ ] Taliyah W - Test vector length vs player distance

---

## 9. CODE LOCATIONS

### Main Prediction Entry Points:
- `GeometricPred::get_prediction()` - GeometricPrediction.h:1025 (single-target)
- `CustomPredictionSDK::predict()` - CustomPredictionSDK.cpp:190 (integration)
- `CustomPredictionSDK::predict_aoe()` - CustomPredictionSDK.cpp:1712 (multi-target)

### Geometry Functions:
- `point_in_capsule()` - GeometricPrediction.h:611
- `point_in_cone()` - GeometricPrediction.h:650
- `calculate_escape_distance_circle()` - GeometricPrediction.h:593
- `calculate_escape_distance_capsule()` - GeometricPrediction.h:686
- `calculate_escape_distance_cone()` - ❌ MISSING

### AOE Prediction:
- `predict_aoe_circle()` - GeometricPrediction.h:1497
- `predict_linear_aoe()` - GeometricPrediction.h:1654
- `predict_aoe_cone()` - GeometricPrediction.h:1815
- `predict_aoe_vector()` - GeometricPrediction.h:1968

### Type Definitions:
- `pred_sdk::spell_type` - pred_sdk.hpp:30 (includes vector)
- `GeometricPred::SpellShape` - GeometricPrediction.h:77 (no vector)

---

## 10. CONCLUSION

**Linear**: ✅ Production-ready
**Circular**: ✅ Production-ready
**Cone**: ❌ **BROKEN for single-target** - Requires immediate fix
**Vector**: ⚠️ Works but needs documentation/clarification

**Overall Grade: 7/10**
- Two types fully implemented
- One type broken (cone single-target)
- One type confusing (vector dual-system)

**Recommendation**:
1. **Fix cone geometry immediately** (blocking issue for cone spells)
2. **Document vector usage** (clarify when to use AOE vs single-target API)
3. **Test all 4 types** with real spells after fixes
