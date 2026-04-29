#ifndef __STATS_H__
#define __STATS_H__
#include <vector>
#include <span>

void rolling_mean(/* in */ std::span<const double> source, /* in */ size_t lookback_periods, /* out */ std::span<double> mean);
std::vector<double> rolling_mean(/* in */ const std::vector<double> & source, /* in */ size_t lookback_periods);

void rolling_stdev(/* in */ std::span<const double> source, /* in */ std::span<const double> mean, /* in */ size_t lookback_periods, /* out */ std::span<double> stdev);
std::vector<double> rolling_stdev(/* in */ const std::vector<double> &  source, /* in */ const std::vector<double> &  mean, /* in */ size_t lookback_periods);

#endif