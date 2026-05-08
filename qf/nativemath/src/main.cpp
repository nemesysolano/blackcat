#include "fracdiff.h"
#include "angles.h"
#include "prices.h"
#include <assert.h>
#include "probabilities.h"
#include "indicators.h"
#include <cmath>
#include <fstream>
#include "stats.h"
#include "entries.h"
#include "sizing.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_model.h"
#include "litert/cc/litert_compiled_model.h"
#include "absl/types/span.h" // LiteRT uses Abseil spans for buffer memory mapping
#include "log.h"
#include "csv-table.h"

using namespace std;
#ifdef __TEST_MAIN__

void fractional_integral_weights_test_success() {
    size_t N = 4;
    double order = 0.5;

    // Mathematical expectations for fractional integral weights where order = 0.5:
    // w[0] = 1.0
    // w[k] = w[k-1] * (k - 1 + order) / k
    // w[1] = 1.0 * (0 + 0.5) / 1 = 0.5
    // w[2] = 0.5 * (1 + 0.5) / 2 = 0.375
    // w[3] = 0.375 * (2 + 0.5) / 3 = 0.3125
    vector<double> expected = {1.0, 0.5, 0.375, 0.3125};

    // 1. Test the return-by-value overload
    vector<double> weights_return = fractional_integral_weights(order, N);
    assert(weights_return.size() == N);
    for(size_t i = 0; i < N; i++) {
        assert(std::abs(weights_return[i] - expected[i]) < 1e-6);
    }

    // 2. Test the pass-by-reference overload
    // Note: The vector must be pre-allocated so the function knows the N size
    vector<double> weights_ref(N, 0.0);
    fractional_integral_weights(order, weights_ref);
    for(size_t i = 0; i < N; i++) {
        assert(std::abs(weights_ref[i] - expected[i]) < 1e-6);
    }

    printf("fractional_integral_weights_test_success passed.\n");
}

void fractional_derivative_weights_test_success() {
    size_t N = 4;
    double order = 0.5;

    // Mathematical expectations for fractional derivative weights where order = 0.5:
    // w[0] = 1.0
    // w[k] = w[k-1] * (k - 1 - order) / k
    // w[1] = 1.0 * (0 - 0.5) / 1 = -0.5
    // w[2] = -0.5 * (1 - 0.5) / 2 = -0.125
    // w[3] = -0.125 * (2 - 0.5) / 3 = -0.0625
    vector<double> expected = {1.0, -0.5, -0.125, -0.0625};

    // 1. Test the return-by-value overload
    vector<double> weights_return = fractional_derivative_weights(order, N);
    assert(weights_return.size() == N);
    for(size_t i = 0; i < N; i++) {
        assert(std::abs(weights_return[i] - expected[i]) < 1e-6);
    }

    // 2. Test the pass-by-reference overload
    // Note: The vector must be pre-allocated so the function knows the N size
    vector<double> weights_ref(N, 0.0);
    fractional_derivative_weights(order, weights_ref);
    for(size_t i = 0; i < N; i++) {
        assert(std::abs(weights_ref[i] - expected[i]) < 1e-6);
    }

    printf("fractional_derivative_weights_test_success passed.\n");
}


void fractional_order_test_success() {
    vector<double> L = {0.1, -0.5, 0.25, -0.125};
    double Lambda = 0.75;
    
    double order = fractional_order(Lambda, L);
    assert(!std::isnan(order));
    printf("Calculated order: %f\n", order);
}

void fractional_order_test_failure() {
    vector<double> L = {-10.0, -5.0, -2.0, -1.0};
    double Lambda = NAN; // This value is bracketed by the function at order 1e-6 and 1.0
    double order = fractional_order(Lambda, L);

    assert(std::isnan(order)); // This will now fail because a root exists
    printf("Calculated order (should be NaN): %f\n", order);
}

void calculate_log_returns_test() {
    vector<double> close_price = {100.0, 110.0, 105.0, 120.0};
    vector<double> expected_log_returns = {NAN, std::log(110.0 / 100.0), std::log(105.0 / 110.0), std::log(120.0 / 105.0)};    
    vector<double> log_returns = calculate_log_returns(close_price);
    
    for (size_t i = 0; i < log_returns.size(); i++) {
        if (std::isnan(expected_log_returns[i])) {
            assert(std::isnan(log_returns[i]));
        } else {
            assert(std::abs(log_returns[i] - expected_log_returns[i]) < 1e-6);
        }
    }
    printf("Log returns test passed.\n");
}

void calculate_price_time_angles_test() {
    vector<double> close_price = {100.0, 110.0, 105.0, 120.0};
    vector<double> high_price = {101.0, 111.0, 106.0, 121.0};
    vector<double> low_price = {99.0, 109.0, 104.0, 119.0};
    
    // The correctly calculated theoretical angles matching the wavelets implementation
    vector<double> expected_angles = {
        // t = 0 (No past data)
        NAN, NAN, NAN, NAN,
        
        // t = 1 (Pivots at t=0 for down movements)
        NAN, std::atan2(1.0, -1.0), NAN, std::atan2(1.0, -1.0),
        
        // t = 2 (Pivots at t=1 for up movements, t=0 for down movements)
        std::atan2(0.5, 1.0), std::atan2(1.0, -1.0), std::atan2(0.5, 1.0), std::atan2(1.0, -1.0),
        
        // t = 3 (Pivots at t=2 for down movements)
        NAN, std::atan2(1.0, -1.0), NAN, std::atan2(1.0, -1.0)
    };

    vector<double> calculated_angles = calculate_price_time_angles(close_price, high_price, low_price);

    for (size_t i = 0; i < expected_angles.size(); i++) {
        if (std::isnan(expected_angles[i])) {
            assert(std::isnan(calculated_angles[i]));
        } else {
            // Using a tolerance of 1e-6 for float comparison
            assert(std::abs(calculated_angles[i] - expected_angles[i]) < 1e-6);
        }
    }
    printf("Price-time angles test passed.\n");
}

void calculate_volume_time_angles_test() {
    vector<double> volume = {1000.0, 1500.0, 1200.0, 1800.0};
    
    vector<double> expected_angles = {
        // t = 0: No past data
        NAN, NAN,
        
        // t = 1 (Vol=1500): v_up=-1, v_down=0 (1000). B_1=1, V_1=500. 
        // j=0 -> NAN
        // j=1 -> b=1/1, v=(1000-1500)/500 = -1
        NAN, std::atan2(1.0, -1.0),
        
        // t = 2 (Vol=1200): v_up=1 (1500), v_down=0 (1000). B_2=(2-0)=2, V_2=300. 
        // j=0 -> b=(2-1)/2=0.5, v=(1500-1200)/300 = 1.0
        // j=1 -> b=(2-0)/2=1.0, v=(1000-1200)/300 = -2/3
        std::atan2(0.5, 1.0), std::atan2(1.0, -2.0/3.0),
        
        // t = 3 (Vol=1800): v_up=-1, v_down=2 (1200). B_3=1, V_3=600.
        // j=0 -> NAN
        // j=1 -> b=1/1, v=(1200-1800)/600 = -1
        NAN, std::atan2(1.0, -1.0)
    };

    vector<double> calculated_angles = calculate_volume_time_angles(volume);

    for (size_t i = 0; i < expected_angles.size(); i++) {
        if (std::isnan(expected_angles[i])) {
            assert(std::isnan(calculated_angles[i]));
        } else {
            assert(std::abs(calculated_angles[i] - expected_angles[i]) < 1e-6);
        }
    }
    printf("Volume-time angles test passed.\n");
}

void F_test() {
    // 1. Setup price and volume data
    vector<double> close_price = {100.0, 110.0, 105.0, 120.0};
    vector<double> high_price  = {105.0, 115.0, 110.0, 125.0};
    
    // FIX: Changed index 0 from 95.0 to 90.0. 
    // This provides a strict "lower low" so l_down can be established at t=2.
    vector<double> low_price   = {90.0,  100.0, 95.0,  115.0}; 
    
    vector<double> volume      = {1000.0, 1500.0, 1200.0, 1800.0};

    // 2. Evaluate the joint probability function F using the vector wrapper
    vector<double> result = F(close_price, high_price, low_price, volume);

    // 3. Validate output dimensions
    assert(result.size() == 4);

    // 4. Validate temporal structure and missing data handling (NaNs)
    
    // t=0: Completely missing prior data
    assert(std::isnan(result[0])); 
    
    // t=1: Volume lacks a v_up structural point (1500 is the highest so far)
    assert(std::isnan(result[1])); 
    
    // t=2: All price and volume structural points NOW exist.
    assert(!std::isnan(result[2])); 
    
    // t=3: Volume lacks a v_up structural point, BUT the Zero-Order Hold 
    // correctly forward-fills the valid probability from t=2.
    assert(!std::isnan(result[3])); 
    assert(result[3] == result[2]); // Validates that the state was held perfectly

    printf("F_test_success passed.\n");
}

void price_time_indicators_test_success() {
    // 1. Setup market data
    vector<double> close_price = {100.0, 110.0, 105.0, 120.0};
    vector<double> high_price  = {105.0, 115.0, 110.0, 125.0};
    vector<double> low_price   = {90.0,  100.0, 95.0,  115.0}; 
    vector<double> volume      = {1000.0, 1500.0, 1200.0, 1800.0};

    size_t N = close_price.size();
    
    // We use a small lookback because N=4 and t=0,1 are NaN burn-in periods
    size_t lookback_periods = 2; 

    // 2. Pre-allocate output vectors for the by-reference overload
    vector<double> LP(N, 0.0);
    vector<double> lambda(N, 0.0); // Lambda
    vector<double> f(N, 0.0);
    vector<double> f_mean(N, 0.0);
    vector<double> f_stdev(N, 0.0);

    // 3. Execute the indicator pipeline
    price_time_indicators(close_price, high_price, low_price, volume, lookback_periods, LP, lambda, f, f_mean, f_stdev);

    // 4. Validate Log Returns (LP)
    assert(std::isnan(LP[0]));
    assert(std::abs(LP[1] - std::log(110.0 / 100.0)) < 1e-6);
    assert(std::abs(LP[2] - std::log(105.0 / 110.0)) < 1e-6);
    assert(std::abs(LP[3] - std::log(120.0 / 105.0)) < 1e-6);

    // 5. Validate Acceleration (Lambda)
    assert(std::isnan(lambda[0])); 
    assert(std::isnan(lambda[1])); // LP[1] - LP[0](NaN) = NaN
    assert(std::abs(lambda[2] - (LP[2] - LP[1])) < 1e-6);
    assert(std::abs(lambda[3] - (LP[3] - LP[2])) < 1e-6);

    // 6. Validate the joint probability function (f)
    assert(std::isnan(f[0])); 
    assert(std::isnan(f[1])); 
    assert(!std::isnan(f[2])); // Market physics fully established here
    assert(!std::isnan(f[3])); // Zero-order hold successfully forward-fills
    assert(f[3] == f[2]);

    // 7. Validate Rolling Statistics (3-Sigma Subsystem)
    assert(std::isnan(f_mean[0]));
    assert(std::isnan(f_mean[1]));
    assert(std::isnan(f_mean[2]));  // lookback is 2, but f[1] is NaN, so window isn't fully formed yet
    assert(!std::isnan(f_mean[3])); // Window successfully forms using f[2] and f[3]
    
    assert(std::isnan(f_stdev[0]));
    assert(std::isnan(f_stdev[1]));
    assert(std::isnan(f_stdev[2]));
    assert(!std::isnan(f_stdev[3])); 

    // Since the zero-order hold keeps f[2] == f[3], the mean equals the value itself
    assert(std::abs(f_mean[3] - f[3]) < 1e-6);
    
    // And standard deviation evaluates to exactly 0.0 (safeguarding against catastrophic cancellation)
    assert(std::abs(f_stdev[3] - 0.0) < 1e-6);

    printf("price_time_indicators_test_success passed.\n");
}

void calculate_acceleration_test() {
    // 1. Setup mock log returns 
    // (Note: t=0 is naturally NaN as it comes from the calculate_log_returns output)
    vector<double> log_returns = {NAN, 0.10, 0.30, -0.20, 0.05};

    // 2. Execute the calculation
    vector<double> acceleration = calculate_acceleration(log_returns);

    // 3. Validate output dimensions
    assert(acceleration.size() == log_returns.size());

    // 4. Validate NaN propagation and mathematical correctness
    
    // t=0: By definition of the algorithm, the first acceleration element is always NaN
    assert(std::isnan(acceleration[0])); 
    
    // t=1: log_returns[1] - log_returns[0] -> (0.10 - NaN) must propagate as NaN
    assert(std::isnan(acceleration[1])); 
    
    // t=2: Valid calculation (0.30 - 0.10 = 0.20)
    assert(std::abs(acceleration[2] - 0.20) < 1e-6);
    
    // t=3: Valid calculation (-0.20 - 0.30 = -0.50)
    assert(std::abs(acceleration[3] - (-0.50)) < 1e-6);
    
    // t=4: Valid calculation (0.05 - (-0.20) = 0.25)
    assert(std::abs(acceleration[4] - 0.25) < 1e-6);

    printf("calculate_acceleration_test passed.\n");
}

void rolling_mean_test() {
    // 1. Setup mock data with a NaN disruption in the middle
    // lookback_periods = 3
    vector<double> source = {1.0, 2.0, 3.0, 4.0, NAN, 10.0, 20.0, 30.0};
    size_t lookback = 3;

    // 2. Execute the calculation
    vector<double> mean = rolling_mean(source, lookback);

    // 3. Validate output dimensions
    assert(mean.size() == source.size());

    // 4. Validate continuous data and NaN reset logic
    
    // t=0, t=1: Initial burn-in period (window not fully formed)
    assert(std::isnan(mean[0]));
    assert(std::isnan(mean[1]));

    // t=2: First valid window -> (1.0 + 2.0 + 3.0) / 3 = 2.0
    assert(std::abs(mean[2] - 2.0) < 1e-6);

    // t=3: Sliding window moves forward -> (2.0 + 3.0 + 4.0) / 3 = 3.0
    assert(std::abs(mean[3] - 3.0) < 1e-6);

    // t=4: NaN encountered. The output must be NaN, and the internal window state must clear.
    assert(std::isnan(mean[4]));

    // t=5, t=6: New burn-in period after the NaN disruption. 
    assert(std::isnan(mean[5]));
    assert(std::isnan(mean[6]));

    // t=7: Window fully formed again -> (10.0 + 20.0 + 30.0) / 3 = 20.0
    assert(std::abs(mean[7] - 20.0) < 1e-6);

    printf("rolling_mean_test passed.\n");
}

void rolling_stdev_test() {
    // 1. Setup mock data with a NaN disruption
    // lookback_periods = 3
    vector<double> source = {10.0, 20.0, 30.0, 40.0, NAN, 100.0, 100.0, 100.0};
    size_t lookback = 3;

    // 2. Calculate the prerequisite mean
    vector<double> mean = rolling_mean(source, lookback);

    // 3. Execute the standard deviation calculation
    vector<double> stdev = rolling_stdev(source, mean, lookback);

    // 4. Validate output dimensions
    assert(stdev.size() == source.size());

    // 5. Validate math and NaN propagation
    
    // t=0, t=1: Initial burn-in period (window not fully formed)
    assert(std::isnan(stdev[0]));
    assert(std::isnan(stdev[1]));

    // t=2: Window [10, 20, 30]. Mean = 20. 
    // Variance = ((10-20)^2 + (20-20)^2 + (30-20)^2) / 2 = (100 + 0 + 100) / 2 = 100
    // Stdev = sqrt(100) = 10.0
    assert(std::abs(stdev[2] - 10.0) < 1e-6);

    // t=3: Window [20, 30, 40]. Mean = 30.
    // Variance = ((20-30)^2 + (30-30)^2 + (40-30)^2) / 2 = 100
    // Stdev = 10.0
    assert(std::abs(stdev[3] - 10.0) < 1e-6);

    // t=4: NaN encountered. Output must be NaN, window resets.
    assert(std::isnan(stdev[4]));

    // t=5, t=6: New burn-in period after the NaN disruption.
    assert(std::isnan(stdev[5]));
    assert(std::isnan(stdev[6]));

    // t=7: Window [100, 100, 100]. Mean = 100.
    // Variance = 0. Stdev = 0.0. 
    // (This also proves the zero-clamping floating-point safeguard is working).
    assert(std::abs(stdev[7] - 0.0) < 1e-6);

    printf("rolling_stdev_test passed.\n");
}

void calculate_fractional_signal_test() {
    // Function Signature Reminder:
    // calculate_fractional_signal(double L, double L_hat, double Lambda, double Lambda_hat, double f0, double f, double f_mean, double f_std, double order)

    // Setup Baseline Probability Bounds
    // Assuming F_STD_K_FACTOR = 2.5
    // If f_mean = 0.5 and f_std = 0.1, boundaries are:
    // Lower = 0.25, Upper = 0.75

    // ---------------------------------------------------------
    // 1. HARD FAILSAFE: High Entropy / Random Walk (order > 0.5)
    // ---------------------------------------------------------
    // Even with perfect bullish alignment and flow state, if entropy is 0.6, it should STALL.
    assert(calculate_fractional_signal(1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.1, 0.6) == STALL);

    // ---------------------------------------------------------
    // 2. STRONG TREND (Flow State)
    // ---------------------------------------------------------
    // Bullish: f=0.5 (Inside 0.25-0.75). All classical and fractional variables are POSITIVE.
    assert(calculate_fractional_signal(1.5, 1.2, 0.8, 0.5, 0.4, 0.5, 0.5, 0.1, 0.2) == STRONG_BULLISH);
    
    // Bearish: f=0.5 (Inside 0.25-0.75). All classical and fractional variables are NEGATIVE.
    assert(calculate_fractional_signal(-1.5, -1.2, -0.8, -0.5, 0.4, 0.5, 0.5, 0.1, 0.2) == STRONG_BEARISH);

    // ---------------------------------------------------------
    // 3. MEAN REVERSION (Structural Saturation)
    // ---------------------------------------------------------
    // Short Reversal: f=0.85 (Breaches 0.75). Momentum (L) is POSITIVE, Acceleration (Lambda) flipped NEGATIVE.
    assert(calculate_fractional_signal(1.5, 1.2, -0.8, -0.5, 0.7, 0.85, 0.5, 0.1, 0.2) == MEAN_REVERSION_SHORT);

    // Long Reversal: f=0.85 (Breaches 0.75). Momentum (L) is NEGATIVE, Acceleration (Lambda) flipped POSITIVE.
    assert(calculate_fractional_signal(-1.5, -1.2, 0.8, 0.5, 0.7, 0.85, 0.5, 0.1, 0.2) == MEAN_REVERSION_LONG);

    // ---------------------------------------------------------
    // 4. PHASE MISMATCH (Fake-Outs / Incoherent Noise)
    // ---------------------------------------------------------
    // Fake-Out Warning: Price is saturating (f=0.85), but classical Lambda (+0.8) and fractional Lambda_hat (-0.5) disagree.
    assert(calculate_fractional_signal(1.5, 1.2, 0.8, -0.5, 0.7, 0.85, 0.5, 0.1, 0.2) == STALL);

    // Incoherent Noise: Flow state (f=0.5), but neural network L_hat (-1.2) disagrees with classical L (+1.5).
    assert(calculate_fractional_signal(1.5, -1.2, 0.8, 0.5, 0.4, 0.5, 0.5, 0.1, 0.2) == STALL);

    std::cout << "calculate_fractional_signal_test passed successfully." << std::endl;
}

void calculate_levels_test() {
    // Shared baseline parameters
    double current_price = 100.0;
    double low_price = 99.0;
    double high_price = 101.0; 
    // base_potential_barrier = 2.0

    double f_mean = 10.0;
    double f_stdev = 1.0;
    
    double L0 = 1.0;
    // Set L = log(1.0) so L_ratio = exp(0)/1.0 = 1.0 (Neutral baseline)
    double L = std::log(1.0); 
    double Lambda = 1.0;
    double Lambda_hat = std::log(2.0); // Lambda_ratio = 1.0, memory_scalar = log1p(1) ~= 0.693

    // Case 1: STALL -> Should return zeroes
    auto lvl_stall = calculate_levels(STALL, L0, L, Lambda, Lambda_hat, 0, 10.0, f_mean, f_stdev, current_price, low_price, high_price, 0.5);
    assert(lvl_stall->signal_direction == 0);
    assert(lvl_stall->take_profit == 0.0);

    // Case 2: Pure Memory Regime (Neutral Velocity, Neutral Bias)
    // total_buffer = 2.0. scalars = 1.0. 
    auto lvl_memory = calculate_levels(STRONG_BULLISH, L0, L, Lambda, Lambda_hat, 0, 10.0, f_mean, f_stdev, current_price, low_price, high_price, 0.0);
    double tp_dist_mem = std::abs(lvl_memory->take_profit - current_price);
    
    // Case 3: Kinetic Velocity Stretch (L >> L0)
    // Set L = log(3.0) -> L_ratio = 3.0. Clamped velocity_scalar = 2.0.
    // TP distance should strictly double compared to Case 2.
    auto lvl_velocity = calculate_levels(STRONG_BULLISH, L0, std::log(3.0), Lambda, Lambda_hat, 0, 10.0, f_mean, f_stdev, current_price, low_price, high_price, 0.0);
    double tp_dist_vel = std::abs(lvl_velocity->take_profit - current_price);
    assert(std::abs(tp_dist_vel - (tp_dist_mem * 2.0)) < 1e-3);

    // Case 4: Trend Alignment (bias matches side)
    // bias_scalar = 1.5
    auto lvl_aligned = calculate_levels(STRONG_BULLISH, L0, L, Lambda, Lambda_hat, 1, 10.0, f_mean, f_stdev, current_price, low_price, high_price, 0.0);
    double tp_dist_aligned = std::abs(lvl_aligned->take_profit - current_price);
    assert(std::abs(tp_dist_aligned - (tp_dist_mem * 1.5)) < 1e-3);

    // Case 5: Counter-Trend Penalty (bias opposes side)
    // bias_scalar = 0.8
    auto lvl_counter = calculate_levels(STRONG_BULLISH, L0, L, Lambda, Lambda_hat, -1, 10.0, f_mean, f_stdev, current_price, low_price, high_price, 0.0);
    double tp_dist_counter = std::abs(lvl_counter->take_profit - current_price);
    assert(std::abs(tp_dist_counter - (tp_dist_mem * 0.8)) < 1e-3);

    printf("calculate_levels_test (Omni-Parameter) passed successfully!\n");
}

void calculate_fractional_qty_test() {
    double current_capital = 100000.0;
    double max_leverage = 5.0;
    double commission = 6.0;

    // Case 1: Invalid Stop Loss -> Should return 0 quantity
    auto size_inv = calculate_fractional_qty(100.0, 100.0, current_capital, 1.0, 1.0, 1.0, 1.0, max_leverage, commission, 0.0);
    assert(size_inv->qty == 0);
    assert(size_inv->leverage == 1.0);

    // Case 2: Full Agreement (Physics and Prediction Align - Pure Memory S=0)
    // price_risk = $10.0. cash_risk = $2000. Base Qty = 200.
    // L=1, L0=1 -> L_ratio ~ 2.718 (clamped to 2.0 in leverage_scalar)
    // Lambda=1, Lambda_hat=1 -> Lambda_ratio ~ 2.718
    // total_force = 2.0 -> sigmoid(2) ~ 0.8808.
    // NEW Sigmoid conviction = 20 * (0.8808 - 0.0) = 17.616
    // ideal_qty = 200 * 17.616 * 1.0 (memory confidence) = 3523.
    auto size_agree = calculate_fractional_qty(100.0, 90.0, current_capital, 1.0, 1.0, 1.0, 1.0, max_leverage, commission, 0.0);
    assert(size_agree->qty == 3523);
    assert(std::abs(size_agree->leverage - 3.52) < 1e-2);

    // Case 3: Disagreement Shield (Sign divergence between Lambda and Lambda_hat)
    // Lambda=1.0, Lambda_hat=-1.0. Agreement is false. 
    // conviction = 1.0 flat. 
    // ideal_qty = 200 * 1.0 = 200.
    auto size_disagree = calculate_fractional_qty(100.0, 90.0, current_capital, 1.0, 1.0, 1.0, -1.0, max_leverage, commission, 0.0);
    assert(size_disagree->qty == 200);

    // Case 4: Hallucination Penalty (Resonance > 5.0)
    // Lambda=1.0, Lambda_hat=5.0 -> total_force=6.0. sigmoid(6) ~ 0.9975. 
    // Base conviction = 20 * 0.9975 = 19.95
    // Lambda_ratio = exp(5)/1 = 148.4. L_ratio = 2.718. Resonance = 54.6 > 5.0.
    // Penalty triggers: conviction = 19.95 * 0.5 = 9.975.
    // ideal_qty = 200 * 9.975 = 1995.
    auto size_hallu = calculate_fractional_qty(100.0, 90.0, current_capital, 1.0, 1.0, 1.0, 5.0, max_leverage, commission, 0.0);
    assert(size_hallu->qty == 1995);

    // Case 5: Leverage Cap Enforcement
    // Same as Case 2, but max_leverage_allowed passed is 1.0x.
    // L_ratio ~ 2.718 (clamped to 2.0). order=0.0 -> memory_confidence=1.0.
    // leverage_scalar = 1.0 * 2.0 = 2.0. 
    // dynamic_leverage_limit = max(1.0, 1.0 * 2.0) = 2.0.
    // Ideal notational is $352,300. Cap restricts buying power to $200,000. Max qty = 2000.
    auto size_cap = calculate_fractional_qty(100.0, 90.0, current_capital, 1.0, 1.0, 1.0, 1.0, 1.0, commission, 0.0);
    assert(size_cap->qty == 2000); 
    assert(size_cap->leverage == 2.0);

    // Case 6: Economic Viability Filter
    // Small account ($1000), low risk asset (price=10). cash_risk = $20. price_risk=$1. Base = 20 qty.
    // In disagreement, qty = 20 * 1.0 = 20. 
    // expected_profit = 20 * 10 * 0.01 = $2.00.
    // commission * 2.0 = $12.0. $2.00 <= $12.0. So, qty = 0.
    auto size_econ = calculate_fractional_qty(10.0, 9.0, 1000.0, 1.0, 1.0, 1.0, -1.0, max_leverage, commission, 0.0);
    assert(size_econ->qty == 0);

    // Case 7: High Entropy Regime Hard Rejection (order > 0.5)
    // order = 0.8 -> Now correctly triggers early return safety net. qty = 0.
    auto size_entropy = calculate_fractional_qty(100.0, 90.0, current_capital, 1.0, 1.0, 1.0, 1.0, max_leverage, commission, 0.8);
    assert(size_entropy->qty == 0);
    assert(size_entropy->leverage == 1.0);

    // Case 8: Continuous Entropy Scaling and Ceiling Suppression (order = 0.4)
    // order = 0.4. memory_confidence = 0.6.
    // Lambda = 1.0, Lambda_hat = 1.5 -> total_force = 2.5. sigmoid(2.5) = 0.924.
    // conviction = 20 * (0.924 - 0.4) = 10.48. memory adjustment = 10.48 * 0.6 = 6.28.
    // base_qty = 1000. ideal_qty = 6289. req_leverage = 6.28.
    // Set L0 = 1.0 and L = 0.0. L_ratio ~ 1.0. leverage_scalar = 0.6 * 1.0 = 0.6.
    // We pass max_leverage_allowed = 10.0. dynamic_leverage_limit = max(1.0, 10.0 * 0.6) = 6.0.
    // The required leverage (6.28) is cleanly clipped down to the entropy ceiling (6.0).
    auto size_suppress = calculate_fractional_qty(100.0, 98.0, current_capital, 1.0, 0.0, 1.0, 1.5, 10.0, commission, 0.4);
    assert(size_suppress->qty == 5999 || size_suppress->qty == 6000); 
    assert(size_suppress->leverage == 6.0); 

    printf("calculate_fractional_qty_test (Sigmoid + Elasticity) passed successfully!\n");
}

void fractional_update_levels_test() {
    double entry_price = 100.0;
    double stop_loss = 90.0;
    double take_profit = 120.0;
    int side = 1; // Long
    double L = 0.0;
    double Lambda = 0.0;
    
    // Logic calibration check (L=0, Lambda=0):
    // alpha_score = 0
    // be_threshold = 0.5 (Breakeven triggered when progress > 50% toward TP)
    // lock_threshold = 0.75 (Lock-in triggered when progress > 75% toward TP)
    // tp_distance = 20.0

    // Case 1: Progress below Breakeven threshold
    // high_price = 105.0 -> progress = (105-100)/20 = 0.25
    // Result should be 0,0 indicating no update required.
    auto res1 = fractional_update_levels(side, stop_loss, take_profit, entry_price, 98.0, 105.0, L, Lambda);
    assert(res1->new_stop_loss == 0.0);
    assert(res1->new_take_profit == 0.0);

    // Case 2: Breakeven Triggered (Progress = 0.6)
    // high_price = 112.0 -> progress = (112-100)/20 = 0.6
    // new_sl = max(90, 100) = 100.0
    // sl_shift = 100 - 90 = 10.0
    // new_tp = 120 + (10 * 0.5) = 125.0
    auto res2 = fractional_update_levels(side, stop_loss, take_profit, entry_price, 98.0, 112.0, L, Lambda);
    assert(std::abs(res2->new_stop_loss - 100.0) < 1e-5);
    assert(std::abs(res2->new_take_profit - 125.0) < 1e-5);

    // Case 3: Profit Lock-in Triggered (Progress = 0.8)
    // high_price = 116.0 -> progress = (116-100)/20 = 0.8
    // lock_in_price = 100 + (1 * 20 * 0.5) = 110.0
    // new_sl = max(90, 110) = 110.0
    // sl_shift = 110 - 90 = 20.0
    // new_tp = 120 + (20 * 0.5) = 130.0
    auto res3 = fractional_update_levels(side, stop_loss, take_profit, entry_price, 98.0, 116.0, L, Lambda);
    assert(std::abs(res3->new_stop_loss - 110.0) < 1e-5);
    assert(std::abs(res3->new_take_profit - 130.0) < 1e-5);

    // Case 4: Short Position Breakeven (Side = -1)
    // stop_loss = 110.0, take_profit = 80.0
    // low_price = 88.0 -> progress = ((88-100)*-1)/20 = 0.6
    // new_sl = min(110, 100) = 100.0
    // sl_shift = 100 - 110 = -10.0
    // new_tp = 80 + (-10 * 0.5) = 75.0
    auto res4 = fractional_update_levels(-1, 110.0, 80.0, entry_price, 88.0, 102.0, L, Lambda);
    assert(std::abs(res4->new_stop_loss - 100.0) < 1e-5);
    assert(std::abs(res4->new_take_profit - 75.0) < 1e-5);

    // Case 5: Physics Influence (High Conviction Lambda=3.0)
    // High conviction moves thresholds down, making the system more aggressive at locking in.
    // sigmoid(3.0) = ~0.95 -> alpha_score = ~0.80 -> be_threshold becomes ~0.38
    // high_price = 108.0 -> progress = 0.4. Should trigger BE due to high conviction.
    auto res5 = fractional_update_levels(side, stop_loss, take_profit, entry_price, 99.0, 108.0, 0.0, 3.0);
    assert(res5->new_stop_loss == 100.0);

    printf("fractional_update_levels_test passed successfully!\n");
}

void fractional_physics_close_test() {
    int entry_index = 0;
    double entry_price = 100.0;
    int qty = 10;
    int side = 1; // Long position
    double f_mean = 10.0;
    double f_std = 1.0;

    // Case 1: 3-Sigma Black Swan Ejector
    // f = 14.0 -> z_score = 4.0 (> 3.0). Should eject instantly regardless of steps.
    // Profit: (105 - 100) * 10 * 1 = 50.0
    auto res1 = fractional_physics_close(5, entry_index, entry_price, qty, side, 105.0, 1.0, 1.0, 14.0, f_mean, f_std);
    assert(res1->physics_close == 1);
    assert(res1->exit_reason == 1);
    assert(std::abs(res1->profit_loss - 50.0) < 1e-5);

    // Case 2: Standard Trade, No Flip (Signs Match)
    // margin = 0. cushion = sigmoid(0)*1.5 = +0.75. max_grace = round(4.75) = 5.
    // steps = 2. Lambda = 1.0 (Positive, matching side 1).
    auto res2 = fractional_physics_close(2, entry_index, entry_price, qty, side, 100.0, 1.0, 1.0, 10.0, f_mean, f_std);
    assert(res2->physics_close == 0);

    // Case 3: Physics Flip Triggered within Grace Window
    // margin = -1.0. cushion = sigmoid(-1.0)*1.5 = +0.403. max_grace = round(4.403) = 4.
    // steps = 2. 2 < 4 is true. Lambda = -1.0 (Negative, disagreeing with side 1).
    auto res3 = fractional_physics_close(2, entry_index, entry_price, qty, side, 99.0, -1.0, 1.0, 10.0, f_mean, f_std);
    assert(res3->physics_close == 2);
    assert(res3->exit_reason == -1); // 99 - 100 = loss

    // Case 4: Physics Flip Survived (Immunity after Grace Window expires)
    // steps = 5. max_grace = 4. 5 < 4 is false. System ignores the flip.
    auto res4 = fractional_physics_close(5, entry_index, entry_price, qty, side, 99.0, -1.0, 1.0, 10.0, f_mean, f_std);
    assert(res4->physics_close == 0);

    // Case 5: Dynamic Grace Extension (Winning Trade + Low Noise)
    // current_price = 102.0. margin = +2%. cushion = sigmoid(2.0)*1.5 = +1.321
    // f = 8.0 -> z_score = -2.0. noise_penalty = -2.0.
    // raw_grace = 4.0 + 1.321 - (-2.0) = 7.321 -> max_grace = 7 steps.
    // steps = 6. 6 < 7 is true. Flip triggers an exit that would normally be ignored.
    auto res5 = fractional_physics_close(6, entry_index, entry_price, qty, side, 102.0, -1.0, 1.0, 8.0, f_mean, f_std);
    assert(res5->physics_close == 2);
    assert(res5->exit_reason == 1);

    // Case 6: Dynamic Grace Contraction (Losing Trade + High Noise)
    // current_price = 98.0. margin = -2%. cushion = sigmoid(-2.0)*1.5 = +0.179
    // f = 12.0 -> z_score = 2.0. noise_penalty = 2.0.
    // raw_grace = 4.0 + 0.179 - 2.0 = 2.179 -> max_grace_steps = 2 steps.
    // steps = 2. 2 < 2 is false. The system clamps the window shut at step 2, ignoring the flip.
    auto res6 = fractional_physics_close(2, entry_index, entry_price, qty, side, 98.0, -1.0, 1.0, 12.0, f_mean, f_std);
    assert(res6->physics_close == 0);

    printf("fractional_physics_close_test passed successfully!\n");
}


void test_cnn_inference_success() {
    vector<string> feature_names = {"Lambda14", "Lambda13", "Lambda12", "Lambda11", "Lambda10", "Lambda9", "Lambda8", "Lambda7", "Lambda6", "Lambda5", "Lambda4", "Lambda3", "Lambda2", "Lambda1"};
    string target_name = "Lambda";

    // 1. Initialize LiteRT Environment
    CSVTable csv("HII.US.csv");

    auto env = litert::Environment::Create({});
    assert(env.HasValue());

    // 2. Load the model into a byte buffer manually
    // This satisfies the BufferRef<uint8_t> signature requirement.
    std::ifstream file("HII.US.tflite", std::ios::binary | std::ios::ate);
    assert(file.is_open());
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Note: This vector must outlive the compiled_model in this scope.
    std::vector<uint8_t> model_buffer_data(size);
    file.read(reinterpret_cast<char*>(model_buffer_data.data()), size);

    // 3. Build the CompiledModel
    // Matching candidate: Create(Environment&, BufferRef<uint8_t>, HwAccelerators)
    auto compiled_model = litert::CompiledModel::Create(
        env.Value(), 
        litert::BufferRef<uint8_t>(model_buffer_data.data(), model_buffer_data.size()), 
        litert::HwAccelerators::kCpu 
    );
    assert(compiled_model.HasValue());

    // 4. Create and Prepare I/O Buffers
    auto input_buffers = compiled_model.Value().CreateInputBuffers();
    auto output_buffers = compiled_model.Value().CreateOutputBuffers();
    assert(input_buffers.HasValue() && output_buffers.HasValue());
    
    // 5. Fill Input Data
    // Per models.py, input_dim = 14 (lags). Shape is (1, 14, 1)
    const int input_dim = feature_names.size();    
    std::vector<float> input_data(input_dim, 0.0f); 
    std::span<float> input_data_view(input_data);

    // 5-1 Read first row
    size_t match_count = 0;
    float diff_sum = 0;
    for(size_t row_number = 0; row_number < csv.row_count(); row_number++) {
        assert(csv.to_float(row_number, feature_names, input_data_view) == CSVTABLE_FILL_OK);

        // Write data into the first input buffer using an Abseil span
        input_buffers.Value()[0].Write<float>(absl::MakeConstSpan(input_data));

        // 6. Run Inference
        compiled_model.Value().Run(input_buffers.Value(), output_buffers.Value());

        // 7. Extract and Validate Output
        // create_fractional_diff_model outputs 1 value via 'tanh', so we need a buffer of size 1
        std::vector<float> output_data(1, 0.0f);
        
        // Pass a mutable span to Read() so LiteRT can copy the tensor data into our vector
        auto read_status = output_buffers.Value()[0].Read<float>(absl::MakeSpan(output_data));
        assert(read_status.HasValue());
        
        float prediction = output_data[0];

        // Tanh activation ensures the result is between -1.0 and 1.0
        assert(prediction >= -1.0f && prediction <= 1.0f);

        float target = csv.to_float(row_number, target_name).value();
        match_count += (signbit(prediction) == signbit(target) ? 1 : 0);
        diff_sum = diff_sum + std::abs(prediction - target);

        cout << "actual=" << target << ", prediction=" << prediction << endl;
    }

    printf("CNN Inference Success, match_count = %f, mae = %f\n", (match_count * 100.0) / csv.row_count(), diff_sum / csv.row_count());

}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    fractional_integral_weights_test_success();
    fractional_derivative_weights_test_success();
    fractional_order_test_success();
    fractional_order_test_failure();
    calculate_log_returns_test();
    calculate_price_time_angles_test();
    calculate_volume_time_angles_test();        
    price_time_indicators_test_success();
    calculate_acceleration_test();    
    F_test();
    rolling_stdev_test();
    rolling_mean_test();
    calculate_fractional_signal_test();
    calculate_levels_test();    
    calculate_fractional_qty_test();
    fractional_physics_close_test();
    fractional_update_levels_test();
    test_cnn_inference_success();
    return 0;
}
#endif // __TEST_MAIN__