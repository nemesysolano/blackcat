#ifndef __ENTRIES_H__
#define __ENTRIES_H__
#include <span>
#include <vector>

#define STRONG_BULLISH 1
#define STRONG_BEARISH -1
#define STALL 0
#define MEAN_REVERSION_LONG 2
#define MEAN_REVERSION_SHORT -2
#define F_STD_K_FACTOR  2.5

int calculate_fractional_signal(double L, double L_hat, double Lambda, double Lambda_hat,  double y, double y_mean, double y_std, double order, double energy_signal, double thrust_signal);
int calculate_fractional_signal_cy(double L, double L_hat, double Lambda, double Lambda_hat, double y, double y_mean, double y_std, double order, double energy_signal, double thrust_signal);
int calculate_fractional_relaxed_signal(double L, double L_hat, double Lambda, double Lambda_hat, double y, double y_mean, double y_std, double order, double energy_signal, double thrust_signal);
int calculate_fractional_relaxed_signal_cy(double L, double L_hat, double Lambda, double Lambda_hat, double y, double y_mean, double y_std, double order, double energy_signal, double thrust_signal);
 
#endif // __ENTRIES_H__