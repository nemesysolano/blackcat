#ifndef __FRACDIFF_H__
#define __FRACDIFF_H__

#include <stdint.h>
#include <cmath>
#include <vector>
#include <span>
#include <functional>

// MUST BE DECLARED BEFORE IT IS USED
using unary_func_t = std::function<double(double)>;

double dot_product(std::span<const double> weights, std::span<const double> values);
double fractional_integral(const std::span<const double> weights, std::span<const double> values);
double fractional_integral(const std::vector<double> & weights, const std::vector<double> & values);
std:: vector<double> fractional_integral_weights(double order, size_t N);
void fractional_integral_weights(double order, std::span<double> & output);
void fractional_integral_weights(double order, std::vector<double> & output);
void fractional_derivative_weights(double order, std::span<double> output);
void fractional_derivative_weights(double order, std::vector<double> & output);
std:: vector<double> fractional_derivative_weights(double order, size_t N);
double fractional_order(double Lambda, std::span<const double> L);
double fractional_order(double Lambda, const std::vector<double> & L);
double fractional_order_cy(double Lambda, int N, const double * L);
double fractional_integral_cy(int N, const double * weights, const double * values);
void fractional_integral_weights_cy(double order, int N, double * weights);
#endif // __FRACDIFF_H__