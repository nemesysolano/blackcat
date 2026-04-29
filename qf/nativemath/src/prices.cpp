#include "prices.h"
#include <cmath>
#include <limits>
#include <vector>
using namespace std;

void calculate_log_returns(std::span<const double> close_price, std::span<double> log_returns) {
    int32_t N = close_price.size();    
    
    log_returns[0] = NAN;    
    for (int32_t t = 1; t < N; t++) {
        log_returns[t] = std::log(close_price[t] / close_price[t-1] + std::numeric_limits<double>::epsilon());
    }
}

std::vector<double> calculate_log_returns(const vector<double> & close_price) {
    int32_t N = close_price.size();    
    vector<double> log_returns(N);

    span close_price_span(close_price);
    span log_returns_span(log_returns);
    calculate_log_returns(close_price_span, log_returns_span);

    return log_returns;
}

void calculate_acceleration(/* in */ std::span<const double> log_returns, /* out */ std::span<double> acceleration) {
    int32_t N = log_returns.size();    
    
    acceleration[0] = NAN;    
    for (int32_t t = 1; t < N; t++) {
        acceleration[t] = log_returns[t] - log_returns[t-1];
    }
}

std::vector<double> calculate_acceleration(const std::vector<double> & log_returns) {
    int32_t N = log_returns.size();    
    vector<double> acceleration(N);

    span log_returns_span(log_returns);
    span acceleration_span(acceleration);
    calculate_acceleration(log_returns_span, acceleration_span);

    return acceleration;
}