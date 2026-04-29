#ifndef __INDICATORS_H__
#define __INDICATORS_H__
#include <span>
#include <vector>
#include <cstddef>

// 1. Core Span Overload
void price_time_indicators(
    /* in */ std::span<const double> close_price, /* in */ std::span<const double> high_price, /* in */ std::span<const double> low_price, /* in */ std::span<const double> volume,
    /* in */ size_t lookback_periods,
    /* out */ std::span<double> LP, /* out */ std::span<double> Lambda, /* out */ std::span<double> f, 
    /* out */ std::span<double> f_mean, /* out */ std::span<double> f_stdev
);

// 2. Vector Wrapper 1
void price_time_indicators(
    /* in */ std::span<const double> close_price, /* in */ std::span<const double> high_price, /* in */ std::span<const double> low_price, /* in */ std::span<const double> volume,
    /* in */ size_t lookback_periods,
    /* out */ std::vector<double> & LP, /* out */ std::vector<double> & Lambda,  /* out */ std::vector<double> & f, 
    /* out */ std::vector<double> & f_mean, /* out */ std::vector<double> & f_stdev
);

// 3. Vector Wrapper 2
void price_time_indicators(
    /* in */ const std::vector<double> & close_price, /* in */ const std::vector<double> & high_price, /* in */ const std::vector<double> & low_price, /* in */ const std::vector<double> & volume,
    /* in */ size_t lookback_periods,
    /* out */ std::vector<double> & LP, /* out */ std::vector<double> & Lambda,  /* out */ std::vector<double> & f, 
    /* out */ std::vector<double> & f_mean, /* out */ std::vector<double> & f_stdev
);

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
    double* f,
    double* f_mean,
    double* f_stdev
);

#endif