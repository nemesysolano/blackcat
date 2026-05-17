#include "sizing.h"
#include "entries.h" // For STRONG_BULLISH and MEAN_REVERSION_LONG constants
#include <cmath>
#include <algorithm>

using namespace std;

double round_level(double value) {
    return std::round(value * 10000.0) / 10000.0;
}

// Helper function to replicate np.sign() behavior
inline int sign(double val) {
    return (0.0 < val) - (val < 0.0);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}


unique_ptr<LEVELS> calculate_levels(
    int signal, 
    double L0, 
    double L, 
    double Lambda, 
    double Lambda_hat, 
    int direction_bias, 
    double f, 
    double f_mean, 
    double f_stdev, 
    double current_price, 
    double low_price, 
    double high_price,
    double order,          // Represents S (Fractional Order / Entropy)
    double energy_signal,  // Magnitude-weighted micro-momentum
    double thrust_signal   // Conviction-weighted micro-momentum
) {
    if (signal == STALL) return make_unique<LEVELS>(0.0, 0.0, 0);

    int side = (signal == STRONG_BULLISH || signal == MEAN_REVERSION_LONG) ? 1 : -1;

    // 1. Base Potential Barrier adjusted by Entropy (order)
    double base_vol = std::max(high_price - low_price, current_price * 0.005);
    double potential_barrier = base_vol * (1.0 + order);

    // 2. Volatility Stress (Z-Score) dampened by Entropy
    double z_score = std::abs(f - f_mean) / (f_stdev + 1e-9);
    double noise_adjustment = 1.0 / (1.0 + order);
    double total_buffer = potential_barrier * (1.0 + (z_score * noise_adjustment));
    
    // 3. Structural Memory
    double Lambda_ratio = std::exp(Lambda_hat) / (Lambda + 1.0);
    double memory_scalar = std::clamp(std::log1p(Lambda_ratio), 0.5, 2.0);

    // 4. Kinetic Velocity (Activating L and L0)
    // If current momentum (L) exceeds baseline (L0), stretch the TP to capture the surge.
    double L_ratio = std::exp(L) / (L0 + 1e-8);
    double velocity_scalar = std::clamp(L_ratio, 0.8, 2.0);
        
    // 5. Macro Directional Bias (Activating direction_bias)
    // Reward trend-following with larger targets; force quick exits on counter-trend setups.
    double bias_scalar = 1.0;
    if (direction_bias != 0) {
        bias_scalar = (side == direction_bias) ? 1.5 : 0.8; 
    }

    // 6. Final Level Determination
    // Stop loss remains rooted in pure noise/entropy survival.
    double stop_loss = round_level(current_price - (side * total_buffer));
    
    // Take Profit is multiplied by Memory, Velocity, and Macro Bias.
    double tp_distance = total_buffer * memory_scalar * velocity_scalar * bias_scalar * 1.5;
    double take_profit = round_level(current_price + (side * tp_distance));

    return std::make_unique<LEVELS>(take_profit, stop_loss, side);
}

// Update the Cython wrapper to pass the 'order'
void calculate_levels_cy(
    int signal, double L0, double L, double Lambda, double Lambda_hat, int direction_bias, 
    double f, double f_mean, double f_stdev, double current_price, 
    double low_price, double high_price, double order,
    double energy_signal, double thrust_signal,
    double * take_profit, double * stop_loss, int * signal_direction
) {
    auto res = calculate_levels(signal, L0, L, Lambda, Lambda_hat, direction_bias, f, f_mean, f_stdev, current_price, low_price, high_price, order, energy_signal, thrust_signal);
    *take_profit = res->take_profit;
    *stop_loss = res->stop_loss;
    *signal_direction = res->signal_direction;
}

unique_ptr<SIZING> calculate_fractional_qty(
    double entry_price, double stop_loss, double current_capital, double L0, double L, double Lambda, double Lambda_hat, double max_leverage_allowed, double platform_commission, double order
) {
    if (stop_loss == 0.0 || entry_price == stop_loss || order > 0.5) {
        return std::make_unique<SIZING>(0, 1.0);
    }
        
    // 1. Strict Risk-Based Quantity 
    // Guarantees base calculation never exceeds MAX_RISK_PER_TRADE
    double DAILY_LOSS_LIMIT = 1.0; 
    double effective_capital = current_capital * DAILY_LOSS_LIMIT;
    double cash_risk = effective_capital * MAX_RISK_PER_TRADE; 
    double price_risk_per_share = std::abs(entry_price - stop_loss);

    if (price_risk_per_share == 0.0) {
        return std::make_unique<SIZING>(0, 1.0);
    }

    // 2. Conviction Scaling (The Drawdown Shield)
    double total_force_magnitude = std::abs(Lambda + Lambda_hat);
    bool agreement = (sign(Lambda_hat) == sign(Lambda));

    double Lambda_ratio = std::exp(Lambda_hat) / (Lambda + 1e-8);
    double L_ratio = std::exp(L) / (L0 + 1e-8);
    double Lambda_relative = std::abs(Lambda_ratio);
    double L_relative = std::abs(L_ratio);
    double resonance_ratio = Lambda_relative / (L_relative + 1e-8);

    double conviction_multiplier = 0.0;
    if (agreement && (sigmoid(total_force_magnitude) - order > 0.0)) {
        // Full throttle: Physics and Predictions align perfectly
        conviction_multiplier = 20 * (sigmoid(total_force_magnitude) - order);
    } else {
        // Drawdown Shield: Slash risk allocation to 20% on structural disagreement
        conviction_multiplier = 1;
    }

    // Penalty for Hallucination
    if (resonance_ratio > 5) {
        conviction_multiplier *= 0.5;
    }

    // 3. Entropy-Based Capital Protection (The Fractional S Integration)
    // If S is high (noise), confidence drops and size is reduced.
    // If S is low (memory), confidence is high and size is maintained.
    double memory_confidence = std::clamp(1.0 - order, 0.1, 1.0);
    conviction_multiplier *= memory_confidence;

    // Apply the combined multipliers to our base quantity
    double base_qty = (cash_risk / price_risk_per_share) * conviction_multiplier;
    int ideal_qty = static_cast<int>(std::floor(base_qty));

    // 4. SAFETY NET: Economic Viability Filter
    // If the trade cannot naturally cover the commission buffer on a 1% move, ABORT it.
    if (platform_commission > 0.0) {
        double expected_profit = ideal_qty * entry_price * 0.01;
        if (expected_profit <= (platform_commission * 2.0)) { // Round-trip cost
            return std::make_unique<SIZING>(0, 1.0);
        }
    }

    // 5. Differentiated Leverage Calculation (Scalar Approach)
    double ideal_notational = ideal_qty * entry_price;
    double actual_leverage = 1.0;
    
    // Leverage Scalar: Combines Entropy (order/S) and Kinetic Surge (L_ratio)
    // memory_confidence = 1.0 - S. L_ratio = exp(L)/L0.
    double leverage_scalar = memory_confidence * std::clamp(L_ratio, 0.5, 2.0);
    double dynamic_leverage_limit = std::max(1.0, max_leverage_allowed * leverage_scalar);

    if (ideal_notational > current_capital) {
        double required_leverage = ideal_notational / current_capital;
        actual_leverage = std::min(required_leverage, dynamic_leverage_limit);
    }

    // 6. Final Quantity Verification
    double buying_power = current_capital * actual_leverage;
    double max_affordable_qty = buying_power / entry_price;
    double qty = std::min(static_cast<double>(ideal_qty), max_affordable_qty);

    // Add a 1e-7 epsilon to counteract floating-point binary truncation
    int final_qty = static_cast<int>(std::floor(qty + 1e-7));
    
    // Round leverage to 2 decimal places
    double rounded_leverage = std::round(actual_leverage * 100.0) / 100.0;

    return std::make_unique<SIZING>(final_qty, rounded_leverage);
}

// Cython wrapper
void calculate_fractional_qty_cy(
    double entry_price, double stop_loss, double current_capital, double L0, double L, double Lambda, double Lambda_hat, double max_leverage_allowed, double platform_commission, double order,
    int * qty, 
    double * leverage
) {
    auto sizing = calculate_fractional_qty(
        entry_price, stop_loss, current_capital, L0, L, Lambda, Lambda_hat, max_leverage_allowed, platform_commission, order
    );
    
    *qty = sizing->qty;
    *leverage = sizing->leverage;
}

// Append to sizing.cpp

unique_ptr<PHYSICS_CLOSE> fractional_physics_close(
    int current_index, 
    int entry_index, 
    double entry_price, 
    int quantity, 
    int side, 
    double current_price, 
    double Lambda, 
    double Lambda_hat, 
    double f, 
    double f_mean, 
    double f_std
) {
    int steps_in_trade = current_index - entry_index;
    double Lambda_ratio = std::exp(Lambda_hat) / (Lambda + 1e-8);

    // Calculate Instantaneous Noise Level
    double z_score = (f - f_mean) / (f_std + 1e-9);

    // The "Black Swan" Ejector (Active at ALL steps)
    if (z_score > 3.0) {
        double profit_loss = (current_price - entry_price) * quantity * side;
        int exit_reason = (profit_loss > 0.0) ? 1 : -1;
        return make_unique<PHYSICS_CLOSE>(exit_reason, profit_loss, 1); // 1 = 3-Sigma Volatility Ejection
    }

    // --- DYNAMIC GRACE PERIOD CALCULATION ---
    
    // 1. Profit Cushion: Winners get more patience (+1.5 steps max), losers get less (-1.5 steps max)
    // Multiplying margin by 100 scales standard stock percentage moves to a clean [-1, 1] sigmoid curve
    double profit_margin = ((current_price - entry_price) / entry_price) * side;
    double profit_cushion = sigmoid(profit_margin * 100.0) * 1.5;

    // 2. Continuous Noise Penalty: Subtracts steps in high volatility, adds steps in low volatility
    double noise_penalty = std::clamp(z_score, -2.0, 2.0);

    // Calculate Dynamic Steps 
    // (Base 4.0 perfectly centers your previous successful 3-to-5 step average)
    double raw_grace = 4.0 + profit_cushion - noise_penalty;

    // Constrain the final grace period strictly between 1 and 8 steps
    int max_grace_steps = static_cast<int>(std::clamp(std::round(raw_grace), 1.0, 8.0));

    // Evaluate Physics Flip against the dynamic grace threshold
    if (steps_in_trade < max_grace_steps && (sign(Lambda_ratio) != sign(side))) {
        double profit_loss = (current_price - entry_price) * quantity * side;
        int exit_reason = (profit_loss > 0.0) ? 1 : -1;
        return make_unique<PHYSICS_CLOSE>(exit_reason, profit_loss, 2); // 2 = Physics Flip
    }

    // No close conditions met
    return make_unique<PHYSICS_CLOSE>(0, 0.0, 0);
}

// Cython wrapper function
void fractional_physics_close_cy(
    int current_index, 
    int entry_index, 
    double entry_price, 
    int quantity, 
    int side, 
    double current_price, 
    double Lambda, 
    double Lambda_hat, 
    double f, 
    double f_mean, 
    double f_std,
    int * exit_reason,
    double * profit_loss,
    int * physics_close
) {
    auto close_result = fractional_physics_close(
        current_index, entry_index, entry_price, quantity, side, 
        current_price, Lambda, Lambda_hat, f, f_mean, f_std
    );
    
    *exit_reason = close_result->exit_reason;
    *profit_loss = close_result->profit_loss;
    *physics_close = close_result->physics_close;
}


unique_ptr<UPDATE_LEVELS> fractional_update_levels(
    int side, double stop_loss, double take_profit, double entry_price, double low_price, double high_price, double L, double Lambda
) {
    double conviction = sigmoid(Lambda * side /* 10.0 */);
    double exhaustion = sigmoid(max(0.0, L * side) /*  5.0 */);
    double alpha_score = conviction - (0.3 * exhaustion);

    double be_threshold = clamp(0.5 - (0.15 * alpha_score), 0.35, 0.65);
    double lock_threshold = clamp(0.75 - (0.10 * alpha_score), 0.65, 0.85);

    double tp_distance = abs(take_profit - entry_price);
    
    // SAFETY GUARD: Prevent Division by Zero
    if (tp_distance == 0.0) {
        return make_unique<UPDATE_LEVELS>(0.0, 0.0);
    }

    double current_best = (side == 1) ? high_price : low_price;
    double progress = ((current_best - entry_price) * side) / tp_distance;

    double new_sl = stop_loss;
    double lock_in_price = entry_price;

    if (progress >= lock_threshold) {
        lock_in_price = entry_price + (side * tp_distance * 0.5);
        new_sl = (side == 1) ? max(stop_loss, lock_in_price) : min(stop_loss, lock_in_price);
    } else if (progress >= be_threshold) {
        new_sl = (side == 1) ? max(stop_loss, entry_price) : min(stop_loss, entry_price);
    }

    if (new_sl != stop_loss) {
        double sl_shift = new_sl - stop_loss;
        
        // We expand TP by 50% of the distance the SL just moved
        double tp_expansion_ratio = 0.5;
        double new_tp = take_profit + (sl_shift * tp_expansion_ratio);
        
        // 3. Security Guard: Ensure TP remains strictly ahead of SL by at least a safe buffer
        double min_tp_buffer = tp_distance * 0.25;
        
        if (side == 1) {
            new_tp = max(new_tp, new_sl + min_tp_buffer);
        } else if(side == -1) {
            new_tp = min(new_tp, new_sl - min_tp_buffer);
        }
        
        return make_unique<UPDATE_LEVELS>(new_sl, new_tp);
    }

    return make_unique<UPDATE_LEVELS>(0.0, 0.0);   
}


void fractional_update_levels_cy(
    int side, double stop_loss, double take_profit, double entry_price, double low_price, double high_price, double L, double Lambda,
    double * new_stop_loss,
    double * new_take_profit
) {
    auto updated = fractional_update_levels(side, stop_loss, take_profit, entry_price, low_price, high_price, L, Lambda);
    
    *new_stop_loss = updated->new_stop_loss;
    *new_take_profit = updated->new_take_profit;
}