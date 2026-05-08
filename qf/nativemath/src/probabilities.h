#ifndef __PROBABILITIES_H__
#define __PROBABILITIES_H__
#include <vector>
#include <span>
#include <cmath>
#include "compat.h"
double w (double phi1, double phi2, double phi3, double phi4);
double W(double phi1, double phi2, double phi3, double phi4);
double h (double phi1, double phi2);
double H(double phi1, double phi2);
double K(double z);
double Q(double z);
double FH (double v1, double v2);
double FW(double theta1, double theta2, double theta3, double theta4);
void F(/* in */ std::span<const double> Θ, /* in */ std::span<const double> phi, /* out */ std::span<double> result);
void Y(/* in, out */ std::span<double> y) ;
void F(/* in */ std::span<const double> close_price, /* in */ std::span<const double> high_price, /* in */ std::span<const double> low_price, /* in */ std::span<const double> volume, /* out */ std::span<double> result);
std::vector<double> F(/* in */ const std::vector<double> & close_price, /* in */ const std::vector<double> & high_price, /* in */ const std::vector<double> & low_price, /* in */ const std::vector<double> & volume);
#endif