#include "entries.h"
#include <cmath>
using namespace std;

int calculate_fractional_signal(double L0, double L, double Lambda, double Lambda_hat, double f, double f_mean, double f_std) {
    // Probability bounds for f
    double upper_f_bound = f_mean + (f_std * F_STD_K_FACTOR);
    double lower_f_bound = f_mean - (f_std * F_STD_K_FACTOR);

    // Calculate the percentage difference between predicted future values and current values
    double Lambda_ratio = std::exp(Lambda_hat) / (Lambda + 5e-8);
    double L_ratio = std::exp(L) / (L0 + 5e-8);
    
    double Lambda_relative = std::abs(Lambda_ratio);
    double L_relative = std::abs(L_ratio);

    // If force magnitude is greater than momentum magnitude then the system
    // is gaining energy and we can consider mean reversion or strong alignment signals
    if (Lambda_relative > L_relative) {
        bool reversion_trigger = (f > upper_f_bound);
        bool alignment_trigger = (f >= lower_f_bound && f <= upper_f_bound);
        
        if (reversion_trigger) {
            if (L_ratio > 0 && Lambda_ratio < 0) {
                return MEAN_REVERSION_SHORT;
            } else if (L_ratio < 0 && Lambda_ratio > 0) {
                return MEAN_REVERSION_LONG;
            } else {
                return STALL;
            }
        } 
        else if (alignment_trigger) {
            if (L_ratio > 0 && Lambda_ratio > 0) {
                return STRONG_BULLISH;
            } else if (L_ratio < 0 && Lambda_ratio < 0) {
                return STRONG_BEARISH;
            } else {
                return STALL;
            }
        }
    }
    
    return STALL;
}

int calculate_fractional_signal_cy(double L0, double L, double Lambda, double Lambda_hat, double f, double f_mean, double f_std) {
    return calculate_fractional_signal(L0, L, Lambda, Lambda_hat, f, f_mean, f_std);
}