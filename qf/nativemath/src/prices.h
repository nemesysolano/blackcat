#ifndef __PRICES_H__
#define __PRICES_H__
#include <vector>
#include <span>


void calculate_log_returns(/* in */ std::span<const double> close_price, /* out */ std::span<double> log_returns);
std::vector<double> calculate_log_returns(const std::vector<double> & close_price);
void calculate_acceleration(/* in */ std::span<const double> log_returns, /* out */ std::span<double> acceleration);
std::vector<double> calculate_acceleration(const std::vector<double> & log_returns);
#endif