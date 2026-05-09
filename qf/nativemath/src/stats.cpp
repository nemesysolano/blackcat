#include "stats.h"
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

void rolling_mean(/* in */ std::span<const double> source, /* in */ size_t lookback_periods, /* out */ std::span<double> mean) {
    size_t N = source.size();
    
    std::fill(mean.begin(), mean.end(), std::numeric_limits<double>::quiet_NaN());
    
    if (lookback_periods <= 0 || N == 0) return;

    double current_sum = 0.0;
    size_t valid_count = 0;

    for (size_t t = 0; t < N; ++t) {
        
        if (std::isnan(source[t])) {
            current_sum = 0.0;
            valid_count = 0;
        } else {
            current_sum += source[t];
            valid_count++;
        }

        if (valid_count > lookback_periods) {
            current_sum -= source[t - lookback_periods];
            valid_count--;
        }

        if (valid_count == lookback_periods) {
            mean[t] = current_sum / static_cast<double>(lookback_periods);
        }
    }
}

std::vector<double> rolling_mean(/* in */ const std::vector<double> &  source, /* in */ size_t lookback_periods){
    vector<double> mean(source.size());
    span<double> mean_span(mean);
    rolling_mean(source, lookback_periods, mean_span);
    return mean; 
}

void rolling_stdev(/* in */ std::span<const double> source, /* in */ std::span<const double> mean, /* in */ size_t lookback_periods, /* out */ std::span<double> stdev) {
    size_t N = source.size();
    std::fill(stdev.begin(), stdev.end(), std::numeric_limits<double>::quiet_NaN());
    
    if (lookback_periods <= 1 || N == 0) return;

    double current_sq_sum = 0.0;
    size_t valid_count = 0;

    for (size_t t = 0; t < N; ++t) {
        
        // FIX: Only reset on source NaN. Do NOT reset on mean NaN.
        // Both functions must process the exact same sliding window simultaneously.
        if (std::isnan(source[t])) {
            current_sq_sum = 0.0;
            valid_count = 0;
        } else {
            current_sq_sum += source[t] * source[t];
            valid_count++;
        }

        // Evict the oldest element from the squared sum window
        if (valid_count > lookback_periods) {
            current_sq_sum -= source[t - lookback_periods] * source[t - lookback_periods];
            valid_count--;
        }

        // Only compute the standard deviation if the window is full AND the mean is ready
        if (valid_count == lookback_periods && !std::isnan(mean[t])) {
            // Algebraic identity: Sum(x_i - mu)^2 = Sum(x_i^2) - N * mu^2
            double variance_num = current_sq_sum - (static_cast<double>(lookback_periods) * mean[t] * mean[t]);
            
            // Floating-point safeguard against catastrophic cancellation
            if (variance_num < 0.0) {
                variance_num = 0.0;
            }
            
            stdev[t] = std::sqrt(variance_num / static_cast<double>(lookback_periods - 1));
        }
    }
}

std::vector<double> rolling_stdev(/* in */ const std::vector<double> & source, /* in */ const std::vector<double> &  mean, /* in */ size_t lookback_periods){
    vector<double> stdev(source.size());
    span<double> stdev_span(stdev);
    rolling_stdev(source, mean, lookback_periods, stdev_span);
    
    return stdev;
}