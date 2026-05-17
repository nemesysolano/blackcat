
#include "ohlc.h"
#include <cmath>
using namespace std;
#define MINIMUM_BARS 3

double energy_weighed_average(/* in */ std::span<const double> open_price, /* in */ std::span<const double> high_price, /* in */ std::span<const double> low_price, /* in */ std::span<const double> close_price) {
    if(open_price.size() != high_price.size() || open_price.size() != low_price.size() || open_price.size() != close_price.size() || open_price.size() < MINIMUM_BARS) {
        return 0;
    }

    double weighted_B_sum = 0.0;
    double Λ_sum =  0.0;
    
    for(size_t index = MINIMUM_BARS - 1; index < open_price.size(); index++) {
        double đ_1 = high_price[index] - close_price[index];
        double đ_2 = open_price[index] - low_price[index];
/*      double đ_3 = high_price[index] - open_price[index]; */
        double đ_4 = close_price[index] - low_price[index];        
        double Λ = log(close_price[index]/( close_price[index-1] + 1e-6)) - log(close_price[index-1]/( close_price[index-2] + 1e-6));
        
        // Failsafe for doji/flat candles
        if (abs((đ_1 + đ_4)) < 1e-9) continue;
        
        double B = (đ_4 - đ_2) / (đ_1 + đ_4);
        
        weighted_B_sum += (abs(Λ) * B); // Lambda acts as the weight
        Λ_sum += abs(Λ);                // Sum of weights
    }
    
    if (std::abs(Λ_sum) < 1e-9) return 0.0; // Prevent division by zero
    
    return weighted_B_sum / Λ_sum;
}

double energy_weighed_average(/* in */ const std::vector<double> & open_price, /* in */ const std::vector<double> & high_price, /* in */ const std::vector<double> & low_price, /* in */ const std::vector<double> & close_price) {
    span<const double> open_span(open_price);
    span<const double> high_span(high_price);
    span<const double> low_span(low_price);
    span<const double> close_span(close_price);
    return energy_weighed_average(open_span, high_span, low_span, close_span);
}

double energy_weighed_average_cy(/* in */ const double * open_price, /* in */ const double * high_price, /* in */ const double * low_price, /* in */ const double * close_price, /* in */ int N) {
    span<const double> open_span(open_price, N);
    span<const double> high_span(high_price, N);
    span<const double> low_span(low_price, N);
    span<const double> close_span(close_price, N);
    return energy_weighed_average(open_span, high_span, low_span, close_span);
}

double thrust_weighed_average(/* in */ std::span<const double> open_price, /* in */ std::span<const double> high_price, /* in */ std::span<const double> low_price, /* in */ std::span<const double> close_price) {
    if(open_price.size() != high_price.size() || open_price.size() != low_price.size() || open_price.size() != close_price.size() || open_price.size() < MINIMUM_BARS) {
        return 0;
    }

    double weighted_B_sum = 0.0;
    double Λ_sum =  0.0;
    
    for(size_t index = MINIMUM_BARS - 1; index < open_price.size(); index++) {
        double đ_1 = high_price[index] - close_price[index];
        double đ_2 = open_price[index] - low_price[index];
/*      double đ_3 = high_price[index] - open_price[index]; */
        double đ_4 = close_price[index] - low_price[index];        
        double Λ = log(close_price[index]/( close_price[index-1] + 1e-6)) - log(close_price[index-1]/( close_price[index-2] + 1e-6));
        
        // Failsafe for doji/flat candles
        if (abs((đ_1 + đ_4)) < 1e-9) continue;
        
        double B = (đ_4 - đ_2) / (đ_1 + đ_4);        
        double λ = exp((B > 0) ? Λ : -Λ);
        
        weighted_B_sum += (λ * B); // Lambda acts as the weight
        Λ_sum += abs(λ);           // Sum of weights
    }
    
    if (std::abs(Λ_sum) < 1e-9) return 0.0; // Prevent division by zero
    
    return weighted_B_sum / Λ_sum;
}

double thrust_weighed_average(/* in */ const std::vector<double> & open_price, /* in */ const std::vector<double> & high_price, /* in */ const std::vector<double> & low_price, /* in */ const std::vector<double> & close_price) {
    span<const double> open_span(open_price);
    span<const double> high_span(high_price);
    span<const double> low_span(low_price);
    span<const double> close_span(close_price);
    return thrust_weighed_average(open_span, high_span, low_span, close_span);
}

double thrust_weighed_average_cy(/* in */ const double * open_price, /* in */ const double * high_price, /* in */ const double * low_price, /* in */ const double * close_price, /* in */ int N) {
    span<const double> open_span(open_price, N);
    span<const double> high_span(high_price, N);
    span<const double> low_span(low_price, N);
    span<const double> close_span(close_price, N);
    return thrust_weighed_average(open_span, high_span, low_span, close_span);
}