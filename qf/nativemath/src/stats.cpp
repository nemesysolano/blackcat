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
    
    // 1. Pre-fill outputs with NaNs
    std::fill(stdev.begin(), stdev.end(), std::numeric_limits<double>::quiet_NaN());
    
    // 2. Sample standard deviation requires at least 2 periods for Bessel's correction (N - 1)
    if (lookback_periods < 2 || N == 0) return;

    double current_sq_sum = 0.0;
    size_t valid_count = 0;

    // 3. O(N) Sliding Window for Sum of Squares
    for (size_t t = 0; t < N; ++t) {
        if (std::isnan(source[t])) {
            // NaN disruption resets the memory window
            current_sq_sum = 0.0;
            valid_count = 0;
        } else {
            // Add square of the new element
            current_sq_sum += source[t] * source[t];
            valid_count++;
        }

        // Eject the oldest squared element falling out of the window
        if (valid_count > lookback_periods) {
            current_sq_sum -= source[t - lookback_periods] * source[t - lookback_periods];
            valid_count--;
        }

        // 4. Calculate standard deviation once the window is fully formed
        if (valid_count == lookback_periods && !std::isnan(mean[t])) {
            // Algebraic identity: Sum(x_i - mu)^2 = Sum(x_i^2) - N * mu^2
            double variance_num = current_sq_sum - (static_cast<double>(lookback_periods) * mean[t] * mean[t]);
            
            // Floating-point safeguard: catastrophic cancellation can sometimes 
            // leave a tiny negative number (e.g., -1e-16) when the true variance is 0.
            if (variance_num < 0.0) {
                variance_num = 0.0;
            }
            
            // Sample standard deviation (Bessel's correction)
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