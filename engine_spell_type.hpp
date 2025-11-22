#pragma once

enum engine_spell_type : uint8_t
{
    self = 0,
    targetted = 1,
    circular = 2,
    conical = 4,
    circular_self = 5,
    heal_shield = 6,
    linear_special = 7,
    linear = 8,
    vector = 10,
    circular_special = 12,
    linear_special_2 = 13,
    invalid = 17,
};