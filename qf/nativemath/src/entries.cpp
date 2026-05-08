#include "entries.h"
#include <cmath>
using namespace std;

int calculate_fractional_signal(double L, double L_hat, double Lambda, double Lambda_hat, double f0, double f, double f_mean, double f_std, double order) { //
    if (order > 0.5) {
        return STALL;
    }

    double upper_f_bound = f_mean + (f_std * F_STD_K_FACTOR);
    double lower_f_bound = f_mean - (f_std * F_STD_K_FACTOR);

    // 1. Verify exact Phase Alignment (Signs)
    int sign_L = (L > 0) ? 1 : ((L < 0) ? -1 : 0);
    int sign_L_hat = (L_hat > 0) ? 1 : ((L_hat < 0) ? -1 : 0); // <-- FIXED
    int sign_Lambda = (Lambda > 0) ? 1 : ((Lambda < 0) ? -1 : 0);
    int sign_Lambda_hat = (Lambda_hat > 0) ? 1 : ((Lambda_hat < 0) ? -1 : 0);

    bool all_align = (sign_L == sign_L_hat) && (sign_L_hat == sign_Lambda) && (sign_Lambda == sign_Lambda_hat);
    bool reversion_align = (sign_L == sign_L_hat) && (sign_Lambda == sign_Lambda_hat) && (sign_L != sign_Lambda);

    // 2. Magnitude Ratios (Safe absolute values)
    double Lambda_ratio = std::exp(Lambda_hat) / (std::abs(Lambda) + 5e-8);
    double L_ratio = std::exp(L_hat) / (std::abs(L) + 5e-8);    

    // 3. Structural States
    bool reversion_trigger = (f > upper_f_bound); // Continuous check
    bool alignment_trigger = (f >= lower_f_bound && f <= upper_f_bound);

    if (reversion_trigger && reversion_align) {
        if (sign_L > 0) return MEAN_REVERSION_SHORT; // L is positive, Lambda is negative
        if (sign_L < 0) return MEAN_REVERSION_LONG;
    } else if (alignment_trigger && all_align) {
        if (sign_L > 0) return STRONG_BULLISH;
        if (sign_L < 0) return STRONG_BEARISH;
    } 

    return STALL;
}

int calculate_fractional_signal_cy(double L0, double L, double Lambda, double Lambda_hat, double f0,  double f, double f_mean, double f_std, double order) {
    return calculate_fractional_signal(L0, L, Lambda, Lambda_hat, f0, f, f_mean, f_std, order);
}