
#ifndef __OHLC_H__
#define __OHLC_H__

#include <vector>
#include <span>

double energy_weighed_average(/* in */ std::span<const double> open_price, /* in */ std::span<const double> high_price, /* in */ std::span<const double> low_price, /* in */ std::span<const double> close_price);
double energy_weighed_average(/* in */ const std::vector<double> & open_price, /* in */ const std::vector<double> & high_price, /* in */ const std::vector<double> & low_price, /* in */ const std::vector<double> & close_price);
double energy_weighed_average_cy(/* in */ const double * open_price, /* in */ const double * high_price, /* in */ const double * low_price, /* in */ const double * close_price, /* in */ int N);

double thrust_weighed_average(/* in */ std::span<const double> open_price, /* in */ std::span<const double> high_price, /* in */ std::span<const double> low_price, /* in */ std::span<const double> close_price);
double thrust_weighed_average(/* in */ const std::vector<double> & open_price, /* in */ const std::vector<double> & high_price, /* in */ const std::vector<double> & low_price, /* in */ const std::vector<double> & close_price);
double thrust_weighed_average_cy(/* in */ const double * open_price, /* in */ const double * high_price, /* in */ const double * low_price, /* in */ const double * close_price, /* in */ int N);
#endif