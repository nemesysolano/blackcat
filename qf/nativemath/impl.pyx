# distutils: language = c++
import numpy as np
cimport numpy as cnp
cnp.import_array()

# 1. Bind to the C++ bridge functions
cdef extern from "indicators.h":
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
    )

cdef extern from "fracdiff.h":
    double fractional_order_cy(double Lambda_val, int N, const double* L)
    double fractional_integral_cy(int N, const double* weights, const double* values)
    # Declaration for the fractional weights generator
    void fractional_integral_weights_cy(double order, int N, double* weights)

# Add this under section "# 1. Bind to the C++ bridge functions"
cdef extern from "entries.h":
    int calculate_fractional_signal_cy(double L0, double L, double Lambda, double Lambda_hat, double f, double f_mean, double f_std, double order)

# Add this under the existing `cdef extern` blocks (around line 30):
cdef extern from "sizing.h":
    void calculate_levels_cy(
        int signal, double L0, double L, double Lambda, double Lambda_hat, int direction_bias, 
        double f, double f_mean, double f_stdev, double current_price, 
        double low_price, double high_price, double order, # Add this
        double * take_profit, double * stop_loss, int * signal_direction
    )

    void calculate_fractional_qty_cy(
        double entry_price, 
        double stop_loss, 
        double current_capital, 
        double L0, 
        double L, 
        double Lambda, 
        double Lambda_hat, 
        double max_leverage_allowed, 
        double platform_commission, 
        double order, 
        int * qty, 
        double * leverage
    )
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
            int * stallness_reason
        )
    void fractional_update_levels_cy(
            int side, 
            double stop_loss, 
            double take_profit, 
            double entry_price, 
            double low_price, 
            double high_price, 
            double L, 
            double Lambda,
            double * new_stop_loss,
            double * new_take_profit
        )

# 2. Python-facing function for indicators
def get_price_time_indicators(double[:] close_price, double[:] high_price, double[:] low_price, double[:] volume, size_t lookback_periods):
    cdef int N = close_price.shape[0]
    
    if high_price.shape[0] != N or low_price.shape[0] != N or volume.shape[0] != N:
        raise ValueError("All input arrays must have the same length.")
        
    cdef cnp.ndarray[cnp.float64_t, ndim=1] LP_out = np.empty(N, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] Lambda_out = np.empty(N, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] f_out = np.empty(N, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] f_mean_out = np.empty(N, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] f_stdev_out = np.empty(N, dtype=np.float64)
    
    cdef double[:] LP_view = LP_out
    cdef double[:] Lambda_view = Lambda_out
    cdef double[:] f_view = f_out
    cdef double[:] f_mean_view = f_mean_out
    cdef double[:] f_stdev_view = f_stdev_out
    
    price_time_indicators_cy(
        &close_price[0], 
        &high_price[0], 
        &low_price[0], 
        &volume[0], 
        N, 
        lookback_periods,
        &LP_view[0], 
        &Lambda_view[0],
        &f_view[0],
        &f_mean_view[0],
        &f_stdev_view[0]
    )
    
    return LP_out, Lambda_out, f_out, f_mean_out, f_stdev_out

# 3. Python-facing function for fractional order
def get_fractional_order(double Lambda_val, double[:] L):
    cdef int N = L.shape[0]
    return fractional_order_cy(Lambda_val, N, &L[0])

# 4. Python-facing function for fractional integral
def get_fractional_integral(double[:] weights, double[:] values):
    """
    Computes the fractional integral (dot product) using the C++ backend.
    """
    cdef int N = weights.shape[0]
    if values.shape[0] != N:
        raise ValueError("Weights and values arrays must have the same length.")
    return fractional_integral_cy(N, &weights[0], &values[0])

# 5. New Python-facing function for fractional integral weights
def get_fractional_integral_weights(double order, int N):
    """
    Generates fractional integral weights of length N for a given order.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
        
    cdef cnp.ndarray[cnp.float64_t, ndim=1] weights_out = np.empty(N, dtype=np.float64)
    fractional_integral_weights_cy(order, N, &weights_out[0])
    
    return weights_out

# Add this at the bottom of the file as a new Python-facing function
# 6. Python-facing function for trading signals
def get_fractional_signal(double L0, double L, double Lambda, double Lambda_hat, double f, double f_mean, double f_std, double order):
    """
    Computes the trading signal based on fractional force and momentum parameters using the C++ backend.
    Returns integers corresponding to STALL (0), STRONG_BULLISH (1), STRONG_BEARISH (-1), etc.
    """
    return calculate_fractional_signal_cy(L0, L, Lambda, Lambda_hat, f, f_mean, f_std, order)    

# Add this at the very bottom of the file:
# 7. Python-facing function for position sizing levels
def get_levels(int signal, double L0, double L, double Lambda, double Lambda_hat, 
               int direction_bias, double f, double f_mean, double f_stdev, 
               double current_price, double low_price, double high_price, double order): # Add order
    cdef double take_profit = 0.0
    cdef double stop_loss = 0.0
    cdef int signal_direction = 0
    
    calculate_levels_cy(
        signal, L0, L, Lambda, Lambda_hat, direction_bias, f, f_mean, f_stdev, 
        current_price, low_price, high_price, order, # Pass order
        &take_profit, &stop_loss, &signal_direction
    )
    
    return take_profit, stop_loss, signal_direction

def get_fractional_qty(double entry_price, double stop_loss, double current_capital, 
                       double L0, double L, double Lambda, double Lambda_hat, 
                       double max_leverage_allowed, double platform_commission, double order):
    """
    Calculates the ideal position size and leverage based on capital risk limits
    and fractional physics conviction scaling.
    Returns: (qty, leverage)
    """
    # Initialize variables to hold the output from C++ pointers
    cdef int qty = 0
    cdef double leverage = 0.0
    
    # Call the C++ wrapper function, passing memory addresses for the outputs
    calculate_fractional_qty_cy(
        entry_price, stop_loss, current_capital, 
        L0, L, Lambda, Lambda_hat, 
        max_leverage_allowed, platform_commission, order, 
        &qty, &leverage
    )
    
    return qty, leverage    

# 9. Python-facing function for dynamic physics close evaluation
def get_fractional_physics_close(int current_index, int entry_index, double entry_price, 
                          int quantity, int side, double current_price, 
                          double Lambda, double Lambda_hat, 
                          double f, double f_mean, double f_std):
    """
    Evaluates whether a trade should be closed due to a 3-Sigma volatility 
    ejection or a dynamic grace-period physics flip.
    Returns: (exit_reason, profit_loss, stallness_reason)
    """
    cdef int exit_reason = 0
    cdef double profit_loss = 0.0
    cdef int stallness_reason = 0
    
    fractional_physics_close_cy(
        current_index, entry_index, entry_price, quantity, side,
        current_price, Lambda, Lambda_hat, f, f_mean, f_std,
        &exit_reason, &profit_loss, &stallness_reason
    )
    
    return exit_reason, profit_loss, stallness_reason    

# 10. Python-facing function for dynamic level updates
def get_fractional_update_levels(int side, double stop_loss, double take_profit, 
                         double entry_price, double low_price, double high_price, 
                         double L, double Lambda):
    """
    Checks for physics-based conviction/exhaustion milestones and 
    dynamically ratchets SL and expands TP targets.
    Returns: (new_stop_loss, new_take_profit) or (0.0, 0.0) if no update.
    """
    cdef double new_sl = 0.0
    cdef double new_tp = 0.0
    
    fractional_update_levels_cy(
        side, stop_loss, take_profit, entry_price, low_price, high_price, L, Lambda,
        &new_sl, &new_tp
    )
    
    return new_sl, new_tp    