#include "entries.h"
#include <cmath>
using namespace std;

#include "entries.h"
#include <cmath>
using namespace std;

int calculate_fractional_signal(    
    double L, double L_hat, double Lambda, double Lambda_hat, double y, double y_mean, double y_std, double order, double energy_signal, double thrust_signal
) { 
    // 1. BURN-IN & NOISE FAILSAFES
    if (order > 0.5) {
        return STALL;
    }
    if (std::isnan(y_mean) || std::isnan(y_std) || std::isnan(y)) {
        return STALL;
    }

    double upper_f_bound = y_mean + (y_std * F_STD_K_FACTOR);
    double lower_f_bound = y_mean - (y_std * F_STD_K_FACTOR);

    // 2. Verify exact Phase Alignment (Signs)
    int sign_L = (L > 0) ? 1 : ((L < 0) ? -1 : 0);
    int sign_L_hat = (L_hat > 0) ? 1 : ((L_hat < 0) ? -1 : 0); 
    int sign_Lambda = (Lambda > 0) ? 1 : ((Lambda < 0) ? -1 : 0);
    int sign_Lambda_hat = (Lambda_hat > 0) ? 1 : ((Lambda_hat < 0) ? -1 : 0);
    
    // DUAL-GATE: Micro-Momentum Signs
    int sign_energy = (energy_signal > 0) ? 1 : ((energy_signal < 0) ? -1 : 0);
    int sign_thrust = (thrust_signal > 0) ? 1 : ((thrust_signal < 0) ? -1 : 0);

    // GATE 1: Trend alignment demands Conviction/Thrust (e^lambda)
    bool all_align = (sign_L == sign_L_hat) && (sign_L_hat == sign_Lambda) && (sign_Lambda == sign_Lambda_hat) && (sign_Lambda_hat == sign_thrust);
    
    // GATE 2: Reversal alignment demands Energy/Inflection (|Lambda|)
    bool reversion_align = (sign_L == sign_L_hat) && (sign_Lambda == sign_Lambda_hat) && (sign_L != sign_Lambda) && (sign_Lambda_hat == sign_energy);

    // 3. Magnitude Ratios (Safe absolute values)
    double Lambda_ratio = std::abs(Lambda_hat) / (std::abs(Lambda) + 5e-8);
    double L_ratio = std::abs(L_hat) / (std::abs(L) + 5e-8);

    // 4. Structural States
    bool reversion_trigger = ((y < lower_f_bound) || (y > upper_f_bound)) && abs(sign_thrust) > abs(sign_energy); // Thrust must be more extreme than Energy to trigger a reversal
    bool alignment_trigger = (y >= lower_f_bound) && (y <= upper_f_bound) && abs(sign_thrust) <= abs(sign_energy); // Both must align and be non-zero for a trend signal

    // 5. Execution Logic
    if(Lambda_ratio > L_ratio && order < 1) {
        if (reversion_trigger && reversion_align) {
            if (sign_L > 0) return MEAN_REVERSION_SHORT; // L is positive, Lambda is negative
            if (sign_L < 0) return MEAN_REVERSION_LONG;
        } else if (alignment_trigger && all_align) {
            if (sign_L > 0) return STRONG_BULLISH;
            if (sign_L < 0) return STRONG_BEARISH;
        } 
    }

    return STALL;
}

int calculate_fractional_signal_cy(
    double L, double L_hat, double Lambda, double Lambda_hat, double y, double y_mean, double y_std, double order, double energy_signal, double thrust_signal) {
    return calculate_fractional_signal(L, L_hat, Lambda, Lambda_hat, y, y_mean, y_std, order, energy_signal, thrust_signal);
}