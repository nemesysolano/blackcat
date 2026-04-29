#ifndef __ANGLES_H__
#define __ANGLES_H__
#include <cstdint>
#include <vector>
#include <span>

void calculate_price_time_angles(
    /* in */ std::span<const double> close_price, 
    /* in */ std::span<const double> high_price, 
    /* in */ std::span<const double> low_price, 
    /* out */ std::span<double> angles
);

// FIX: Accept span instead of const vector&
std::vector<double> calculate_price_time_angles(
    std::span<const double> close_price, 
    std::span<const double> high_price, 
    std::span<const double> low_price
);

void calculate_volume_time_angles(
    /* in */ const std::span<const double> volume, 
    /* out */ std::span<double> angles
);

// FIX: Accept span instead of const vector&
std::vector<double> calculate_volume_time_angles(std::span<const double> volume);

#endif // __ANGLES_H__