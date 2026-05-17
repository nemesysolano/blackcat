#ifndef __SIZING_H__
#define __SIZING_H__
#include <memory>
#define MAX_RISK_PER_TRADE 0.02
#define TRANSACTION_COMMISSION 0.02 
#define EXIT_WITH_PROFIT 1
#define EXIT_WITH_LOSS -1
#define PHYSICS_CLOSE_3_SIGMA 1 // 3-Sigma Volatility Ejection
#define PHYSICS_CLOSE_FLIP 2 // Physics Flip
#define STALL 0

typedef struct LEVELS_STRUCT {
    double take_profit;
    double stop_loss;
    int signal_direction;
    LEVELS_STRUCT(double take_profit, double stop_loss, int signal_direction) : take_profit(take_profit), stop_loss(stop_loss), signal_direction(signal_direction) {}
} LEVELS;


std::unique_ptr<LEVELS> calculate_levels(
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
    double order,    
    double energy_signal, 
    double thrust_signal
);

void calculate_levels_cy(
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
    double order,
    double energy_signal, 
    double thrust_signal,    
    double * take_profit,
    double * stop_loss,
    int * signal_direction
);

typedef struct SIZING_STRUCT {
    int qty;
    double leverage;
    SIZING_STRUCT(int qty, double leverage) : qty(qty), leverage(leverage) {}
} SIZING;

std::unique_ptr<SIZING> calculate_fractional_qty(double entry_price, double stop_loss, double current_capital, double L0, double L, double Lambda, double Lambda_hat, double max_leverage_allowed, double platform_commission, double order);
void calculate_fractional_qty_cy(double entry_price, double stop_loss, double current_capital, double L0, double L, double Lambda, double Lambda_hat, double max_leverage_allowed, double platform_commission, double order, int * qty, double * leverage);

typedef struct PHYSICS_CLOSE_STRUCT {
    int exit_reason;
    double profit_loss;
    int physics_close;
} PHYSICS_CLOSE;

std::unique_ptr<PHYSICS_CLOSE> fractional_physics_close(int current_index, int entry_index, double entry_price, int quantity, int side, double current_price, double Lambda, double Lambda_hat, double f, double f_mean, double f_std);
void fractional_physics_close_cy(
    int current_index, int entry_index, double entry_price, int quantity, int side, double current_price, double Lambda, double Lambda_hat, double f, double f_mean, double f_std,
    int * exit_reason,
    double * profit_loss,
    int * physics_close
);

typedef struct UPDATE_LEVELS_STRUCT {
    double new_stop_loss;
    double new_take_profit;
} UPDATE_LEVELS;

std::unique_ptr<UPDATE_LEVELS> fractional_update_levels(
    int side, double stop_loss, double take_profit, double entry_price, double low_price, double high_price, double L, double Lambda
);

void fractional_update_levels_cy(
    int side, double stop_loss, double take_profit, double entry_price, double low_price, double high_price, double L, double Lambda,
    double * new_stop_loss,
    double * new_take_profit
);
#endif