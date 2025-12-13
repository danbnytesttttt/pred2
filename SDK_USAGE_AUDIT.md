# SDK Usage Audit - Hardcoded Values Analysis

## Methodology
Searched for all `constexpr`, `const`, and `#define` values in prediction code.
Cross-referenced with available SDK methods to identify what can be replaced.

---

## 1. MATHEMATICAL CONSTANTS ✅ JUSTIFIED

### GeometricPrediction.h:45-46
```cpp
constexpr float PI = 3.14159265358979323846f;
constexpr float EPSILON = 1e-6f;
```

**Status**: ✅ **KEEP - Mathematical constants**
**Reason**: Universal math constants, not game-specific
**SDK Alternative**: None (these are fundamental math constants)

---

## 2. REACTION TIME THRESHOLDS ⚠️ NEEDS REVIEW

### GeometricPrediction.h:49-52
```cpp
constexpr float REACTION_UNDODGEABLE = 0.0f;   // No time to react
constexpr float REACTION_VERY_HIGH = 0.1f;     // 100ms window
constexpr float REACTION_HIGH = 0.25f;         // 250ms window
constexpr float REACTION_MEDIUM = 0.4f;        // 400ms window
```

**Status**: ⚠️ **QUESTIONABLE - Based on human reaction research**
**Reason**:
- Human reaction times: 150-300ms (tested/researched)
- These thresholds are based on psychology research, not League-specific
- Different from game constants

**SDK Alternative**: None (human performance, not game data)

**QUESTION FOR USER**: Are these values from research or arbitrary?
- If research-based → Keep with citation
- If arbitrary → Should make user-configurable

---

## 3. MINION COLLISION CONSTANTS ❌ CAN USE SDK

### GeometricPrediction.h:58-60
```cpp
constexpr float MINION_SEARCH_RADIUS = 150.f;
constexpr float MINION_HITBOX_RADIUS = 65.f;
constexpr float MINION_RELEVANCE_RANGE = 2000.f;
```

**Status**: ❌ **REPLACE WITH SDK DATA**

### Analysis:

#### A. `MINION_HITBOX_RADIUS = 65.f` ❌ WRONG APPROACH
**Current**: Hardcoded "average" minion radius
**Problem**: Different minions have different sizes!
- Melee minions: ~55 units
- Caster minions: ~48 units
- Siege minions: ~65 units
- Super minions: ~70 units

**SDK Provides**: `minion->get_bounding_radius()`

**Fix**: Use actual bounding radius per minion
```cpp
// BEFORE: Hardcoded average
constexpr float MINION_HITBOX_RADIUS = 65.f;

// AFTER: Use SDK per minion
float minion_radius = minion->get_bounding_radius();
```

#### B. `MINION_SEARCH_RADIUS = 150.f` ⚠️ ALGORITHM PARAMETER
**Current**: Search radius for minions around spell path
**Status**: This is an algorithm efficiency parameter, not game data
**Decision**: ✅ **KEEP** - Controls search optimization, not accuracy

#### C. `MINION_RELEVANCE_RANGE = 2000.f` ✅ PERFORMANCE OPTIMIZATION
**Current**: Only check minions within 2000 units
**Status**: Performance optimization (skip distant minions)
**Decision**: ✅ **KEEP** - Prevents wasted checks on irrelevant minions

---

## 4. DAMAGE ESTIMATION CONSTANTS ❌ SHOULD USE SDK

### GeometricPrediction.h:61-63
```cpp
constexpr float LANE_BASE_DPS = 50.f;          // Estimated minion + champion poke DPS
constexpr float TOWER_DPS = 250.f;             // Tower shot DPS estimate
constexpr float TOWER_AGGRO_RANGE = 900.f;     // Tower aggro/attack range
```

**Status**: ❌ **CAN GET FROM SDK OR MEASURE**

### Analysis:

#### A. `TOWER_AGGRO_RANGE = 900.f` ❌ SHOULD USE SDK
**SDK Method**:
```cpp
// Get tower object
auto towers = g_sdk->object_manager->get_turrets();
for (auto* tower : towers) {
    float range = tower->get_attack_range();  // Actual tower attack range!
}
```

**Fix**: Use `tower->get_attack_range()` instead of hardcoding 900

#### B. `TOWER_DPS = 250.f` ❌ CAN CALCULATE FROM SDK
**SDK Provides**:
- `tower->get_attack_damage()` - base damage
- `tower->get_attack_delay()` - attack speed
- `tower->get_level()` - towers scale with time

**Fix**: Calculate actual DPS = damage / attack_delay

#### C. `LANE_BASE_DPS = 50.f` ⚠️ HEURISTIC
**Current**: Estimates combined minion + poke damage
**Problem**: This is a rough guess for minion health prediction
**Better**: Track actual incoming damage if health prediction SDK unavailable
**Decision**: ⚠️ **KEEP AS FALLBACK** - Used only when health prediction SDK unavailable

---

## 5. AOE MOVEMENT BUFFER ❌ ARBITRARY

### GeometricPrediction.h:66
```cpp
constexpr float AOE_MOVEMENT_BUFFER = 500.f;   // Extra range to account for target movement during cast
```

**Status**: ❌ **ARBITRARY - Should calculate based on spell speed**

**Problem**: 500 units is a guess
**Better approach**: Calculate based on spell travel time
```cpp
// INSTEAD OF: Fixed 500 buffer
// CALCULATE: max_distance_target_can_move = move_speed * spell_travel_time
float spell_travel_time = distance / spell_speed;
float movement_buffer = target_move_speed * spell_travel_time;
```

**Fix**: Replace with dynamic calculation

---

## 6. HUMAN REACTION TIME ⚠️ PSYCHOLOGY CONSTANT

### GeometricPrediction.h:404
```cpp
constexpr float HUMAN_REACTION_TIME = 0.20f;  // Average reaction time
```

**Status**: ⚠️ **BASED ON RESEARCH**
**Reason**: 200ms is average human reaction time (psychology research)
**SDK Alternative**: None (human performance metric)

**QUESTION**: Should this be user-configurable? Some players react faster/slower.

---

## 7. STASIS PING BUFFER ✅ JUSTIFIED

### EdgeCaseDetection.h:132
```cpp
constexpr float BASE_BUFFER = 0.016f;  // Base 16ms (1 frame @ 60Hz)
```

**Status**: ✅ **JUSTIFIED - Frame time calculation**
**Reason**: 1 frame at 60 FPS = 16.67ms ≈ 16ms
**SDK Alternative**: Could get actual frame time, but 16ms is reasonable minimum

---

## 8. SAFETY MARGINS ✅ ALGORITHM PARAMETERS

### GeometricPrediction.h:518, 669
```cpp
constexpr float SAFETY_MARGIN = 20.f;
constexpr float MIN_SAFE_DISTANCE = 0.01f;
```

**Status**: ✅ **KEEP - Algorithm stability parameters**
**Reason**: Prevent division by zero, numerical stability
**Not game-specific**: Computational safety, not League data

---

## 9. CONFIDENCE THRESHOLDS ❌ ARBITRARY

### GeometricPrediction.h:999, 1008
```cpp
constexpr float LOW_THRESHOLD = 0.6f;
constexpr float VERY_LOW_THRESHOLD = 1.0f;
```

**Status**: ❌ **ARBITRARY - Should be configurable or removed**

**Problem**: These look like they're defining what counts as "low" confidence
**Better**: Let user/champion script decide thresholds

---

## 10. WINDWALL WIDTHS ❌ HARDCODED GAME DATA

### EdgeCaseDetection.h:1227
```cpp
static const float widths[] = { 320.f, 390.f, 460.f, 530.f, 600.f };
```

**Context**: Yasuo windwall width by level
**Status**: ❌ **GAME DATA - Could change with patches**

**SDK Alternative**: Could be in spell data
**Problem**: Hardcoding patch-specific values that can change

**Better approach**:
1. Try to get from spell data
2. If not available, query width dynamically
3. If must hardcode, add comment: "// Patch 14.X values"

---

## SUMMARY OF FINDINGS

### ✅ KEEP (Justified Hardcodes)

1. **Mathematical constants** (PI, EPSILON)
2. **Performance optimizations** (MINION_RELEVANCE_RANGE)
3. **Algorithm parameters** (SAFETY_MARGIN, MIN_SAFE_DISTANCE)
4. **Frame time** (BASE_BUFFER = 16ms)

### ⚠️ QUESTIONABLE (Need User Decision)

1. **Reaction time thresholds** (100ms, 250ms, 400ms)
   - Research-based or arbitrary?
   - Should be user-configurable?

2. **Human reaction time** (200ms)
   - Average value from research
   - Should vary per player?

3. **Lane base DPS** (50.f)
   - Fallback when health prediction unavailable
   - Keep as heuristic?

### ❌ SHOULD FIX (Can Use SDK)

1. **Minion bounding radius** → Use `minion->get_bounding_radius()`
2. **Tower attack range** → Use `tower->get_attack_range()`
3. **Tower DPS** → Calculate from `get_attack_damage() / get_attack_delay()`
4. **AOE movement buffer** → Calculate from spell speed and target move speed
5. **Windwall widths** → Try to get from spell data or mark as patch-specific
6. **Confidence thresholds** → Remove or make configurable

---

## DETAILED FIX PLAN

### Priority 1: Use SDK Bounding Radius ❌

**Current**:
```cpp
// EdgeCaseDetection.h - minion collision
constexpr float MINION_HITBOX_RADIUS = 65.f;
float minion_radius = MINION_HITBOX_RADIUS;
```

**Fixed**:
```cpp
// Use actual bounding radius from SDK
float minion_radius = minion->get_bounding_radius();
```

**Impact**: More accurate collision detection for different minion types

---

### Priority 2: Use SDK Tower Data ❌

**Current**:
```cpp
constexpr float TOWER_DPS = 250.f;
constexpr float TOWER_AGGRO_RANGE = 900.f;
```

**Fixed**:
```cpp
// Get actual tower stats from SDK
float tower_range = tower->get_attack_range();
float tower_damage = tower->get_attack_damage();
float tower_attack_speed = tower->get_attack_delay();
float tower_dps = tower_damage / tower_attack_speed;
```

**Impact**: Accurate for all tower types (base, tier 2, tier 3, nexus)

---

### Priority 3: Dynamic AOE Buffer ❌

**Current**:
```cpp
constexpr float AOE_MOVEMENT_BUFFER = 500.f;  // Arbitrary!
```

**Fixed**:
```cpp
// Calculate based on spell travel time and target speed
float distance_to_target = (target_pos - source_pos).magnitude();
float spell_travel_time = distance_to_target / spell_speed;
float movement_buffer = target->get_move_speed() * spell_travel_time;
```

**Impact**: Accurate buffer that scales with distance and target speed

---

### Priority 4: Remove Hardcoded Minion Radius Constant ❌

**Location**: GeometricPrediction.h:59, EdgeCaseDetection.h (duplicated)

**Action**: Delete the constant, always use SDK
```cpp
// DELETE THIS:
constexpr float MINION_HITBOX_RADIUS = 65.f;

// ALWAYS USE:
float minion_radius = minion->get_bounding_radius();
```

---

### Priority 5: Windwall Width from Spell Data ❌

**Current**: Hardcoded array by level
**Better**: Try to get from spell data first
```cpp
// Try SDK spell data first
float windwall_width = 0.f;
auto yasuo_spell = yasuo->get_spell(W_SLOT);
if (yasuo_spell && yasuo_spell->get_spell_data()) {
    // Try to get width from spell data
    windwall_width = yasuo_spell->get_spell_data()->get_width();
}

// Fallback to level-based (with patch version comment)
if (windwall_width == 0.f) {
    // Patch 14.23 values - may need update
    static const float widths[] = { 320.f, 390.f, 460.f, 530.f, 600.f };
    windwall_width = widths[spell_level - 1];
}
```

---

## QUESTIONS FOR USER

1. **Reaction time thresholds** (100ms, 250ms, 400ms): Are these from research or arbitrary?
2. **Should reaction times be user-configurable?** Some players are faster/slower
3. **Lane base DPS** (50.f): Keep as fallback heuristic or remove entirely?
4. **Confidence thresholds**: Should these be in champion scripts instead of hardcoded?
5. **Windwall widths**: Acceptable to hardcode with patch version comment?

---

## IMPLEMENTATION CHECKLIST

- [ ] Replace MINION_HITBOX_RADIUS with SDK get_bounding_radius()
- [ ] Replace TOWER_AGGRO_RANGE with SDK get_attack_range()
- [ ] Calculate TOWER_DPS from SDK (damage / attack_delay)
- [ ] Replace AOE_MOVEMENT_BUFFER with dynamic calculation
- [ ] Try to get Windwall width from spell data, fallback to array
- [ ] Add patch version comments to remaining hardcoded game values
- [ ] Document which constants are algorithm parameters vs game data
- [ ] Make reaction time thresholds configurable (if user wants)
