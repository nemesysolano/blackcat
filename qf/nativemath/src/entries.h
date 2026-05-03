#ifndef __ENTRIES_H__
#define __ENTRIES_H__

#define STRONG_BULLISH 1
#define STRONG_BEARISH -1
#define STALL 0
#define MEAN_REVERSION_LONG 2
#define MEAN_REVERSION_SHORT -2
#define F_STD_K_FACTOR  2.5

int calculate_fractional_signal(double L0, double L, double Lambda, double Lambda_hat, double f0, double f, double f_mean, double f_std, double order);
int calculate_fractional_signal_cy(double L0, double L, double Lambda, double Lambda_hat, double f0, double f, double f_mean, double f_std, double order);
#endif // __ENTRIES_H__