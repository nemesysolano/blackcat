#include "indicators.h"
#include "prices.h"
#include "probabilities.h"
#include "angles.h"
#include "stats.h"
#include <limits>
#include <vector>
using namespace std;

// 1. Core Logic Overload
void price_time_indicators(
    /* in */ std::span<const double> close_price, /* in */ std::span<const double> high_price, /* in */ std::span<const double> low_price, /* in */ std::span<const double> volume, /* in */ size_t lookback_periods,
    /* out */ std::span<double> LP, 
    /* out */ std::span<double> Lambda, 
    /* out */ std::span<double> y
){
    // Calculate Log Returns
    calculate_log_returns(close_price, LP);
    
    // Calculate Acceleration
    calculate_acceleration(LP, Lambda);
    
    // Calculate raw probabilities 
    F(close_price, high_price, low_price, volume, y);    
    Y(y);    
}

// 2. Vector Wrapper 1
void price_time_indicators(
    /* in */ std::span<const double> close_price, /* in */ std::span<const double> high_price, /* in */ std::span<const double> low_price, /* in */ std::span<const double> volume, /* in */ size_t lookback_periods,
    /* out */ std::vector<double> & LP,
    /* out */ std::vector<double> & Lambda,
    /* out */ std::vector<double> & y
) {
    span<double> LP_span(LP);
    span<double> Lambda_span(Lambda);
    span<double> f_span(y);

    price_time_indicators(close_price, high_price, low_price, volume, lookback_periods, LP_span, Lambda_span, f_span);
}

// 3. Vector Wrapper 2
void price_time_indicators(
    /* in */ const std::vector<double> & close_price, /* in */ const std::vector<double> & high_price, /* in */ const std::vector<double> & low_price, /* in */ const std::vector<double> & volume, /* in */ size_t lookback_periods,
    /* out */ std::vector<double> & LP,
    /* out */ std::vector<double> & Lambda,
    /* out */ std::vector<double> & y
) {
    span<const double> close_price_span(close_price);
    span<const double> high_price_span(high_price);
    span<const double> low_price_span(low_price);
    span<const double> volume_span(volume);

    price_time_indicators(close_price_span, high_price_span, low_price_span, volume_span, lookback_periods, LP, Lambda, y);
}

// 4. Cython C-Bridge
void price_time_indicators_cy(
    const double* close_price, 
    const double* high_price, 
    const double* low_price, 
    const double* volume, 
    int N, 
    size_t lookback_periods,
    double* LP, 
    double* Lambda, 
    double* y
) {
    std::span<const double> close_span(close_price, N);
    std::span<const double> high_span(high_price, N);
    std::span<const double> low_span(low_price, N);
    std::span<const double> vol_span(volume, N);
    
    std::span<double> LP_span(LP, N);
    std::span<double> Lambda_span(Lambda, N);
    std::span<double> f_span(y, N);

    price_time_indicators(close_span, high_span, low_span, vol_span, lookback_periods, LP_span, Lambda_span, f_span);
}