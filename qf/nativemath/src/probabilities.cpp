#include "probabilities.h"
#include "angles.h"
#include <numbers>
#include <cmath>
#include <algorithm>

using namespace std;

double w (double phi1, double phi2, double phi3, double phi4) {
    return (std::cos(phi1) + std::sin(phi1) + std::cos(phi2) + std::sin(phi2) + std::cos(phi3) + std::sin(phi3) + std::cos(phi4) + std::sin(phi4)) / (4 * std::sqrt(2));
}

double W(double phi1, double phi2, double phi3, double phi4) {
    return (128.0/(std::pow(M_PI,4) + 2*std::pow(M_PI,3) + 48*std::pow(M_PI,2))) * std::pow(w(phi1, phi2, phi3, phi4),2);
}

double h (double phi1, double phi2) {
    return (std::cos(phi1) + std::sin(phi1) + std::cos(phi2) + std::sin(phi2)) / (2 * std::sqrt(2));
}

double H(double phi1, double phi2) {
      return (16.0/(std::pow(M_PI,2) + 2*M_PI + 16.0)) * std::pow(h(phi1, phi2),2);
}

double K(double z) {
    return 1.0 + std::sin(z) - std::cos(z);
}

double Q(double z) {
    return z + std::pow(std::sin(z), 2);
}

double FH (double v1, double v2) {
    static const double constant = ((16.0/(std::pow(M_PI,2) + 2*M_PI + 16.0))/8.0);
    return constant * (v2 * Q(v1) + v1 * Q(v2) + 2.0 * K(v1) * K(v2));
}

double FW(double theta1, double theta2, double theta3, double theta4) {
    static const double constant = (128.0 / (std::pow(M_PI, 4) + 2 * std::pow(M_PI, 3) + 48 * std::pow(M_PI, 2))) / 32.0;

    double q1 = Q(theta1), q2 = Q(theta2), q3 = Q(theta3), q4 = Q(theta4);
    double k1 = K(theta1), k2 = K(theta2), k3 = K(theta3), k4 = K(theta4);

    double squared_terms = 
        q1 * theta2 * theta3 * theta4 +
        q2 * theta1 * theta3 * theta4 +
        q3 * theta1 * theta2 * theta4 +
        q4 * theta1 * theta2 * theta3;

    double cross_product_terms = 2.0 * (
        k1 * k2 * theta3 * theta4 +
        k1 * k3 * theta2 * theta4 +
        k1 * k4 * theta2 * theta3 +
        k2 * k3 * theta1 * theta4 +
        k2 * k4 * theta1 * theta3 +
        k3 * k4 * theta1 * theta2
    );

    return constant * (squared_terms + cross_product_terms);
}

void F(/* in */ span<const double> Θ, /* in */ span<const double> phi, /* out */ span<double> result) {
    size_t N = phi.size() / 2;
    
    // Track the current state of the market physics
    double last_known_prob = std::numeric_limits<double>::quiet_NaN();
    
    for (size_t t = 0; t < N; ++t) {
        // Bounds checking
        if (t * 4 + 3 >= Θ.size() || t * 2 + 1 >= phi.size()) {
            result[t] = last_known_prob;
            continue;
        }

        double theta1 = Θ[t * 4 + 0]; 
        double theta2 = Θ[t * 4 + 1]; 
        double theta3 = Θ[t * 4 + 2]; 
        double theta4 = Θ[t * 4 + 3]; 
        
        double phi1 = phi[t * 2 + 0];   
        double phi2 = phi[t * 2 + 1];   

        // If the market structure is fully formed on this tick, update our known state
        if (!std::isnan(theta1) && !std::isnan(theta2) && !std::isnan(theta3) && !std::isnan(theta4) && 
            !std::isnan(phi1) && !std::isnan(phi2)) {
            
            last_known_prob = (FW(theta1, theta2, theta3, theta4) + FH(phi1, phi2)) / 2.0;
        }
        
        // Zero-Order Hold: Write the latest valid state to the result array.
        // If we haven't found the very first structural point yet (burn-in), this remains NaN safely.
        result[t] = last_known_prob;
    }        
}


void Y(/* in, out */ std::span<double> y) {
    size_t N = y.size();
    
    for (size_t t = 0; t < N; t++) {
        double f = y[t];
        
        if (std::isnan(f)) {
            continue;
        }

        // Clamp to prevent log(0) or division by zero (Inf / -Inf)
        if (f < 1e-9) {
            f = 1e-9;
        } else if (f > 1.0 - 1e-9) {
            f = 1.0 - 1e-9;
        }
        
        // In-place Logit Transform
        y[t] = std::log(f / (1.0 - f));
    }
}

void F(/* in */ std::span<const double> close_price, /* in */ std::span<const double> high_price, /* in */ std::span<const double> low_price, /* in */ std::span<const double> volume, /* out */ std::span<double> result) {    
    std::vector<double> Θ = calculate_price_time_angles(close_price, high_price, low_price);
    std::vector<double> phi = calculate_volume_time_angles(volume);    
    F(Θ, phi, result);
}

vector<double> F(/* in */ const std::vector<double> & close_price, /* in */ const std::vector<double> & high_price, /* in */ const std::vector<double> & low_price, /* in */ const std::vector<double> & volume) {    
    span close_price_span(close_price);
    span high_price_span(high_price);
    span low_price_span(low_price);
    span volume_span(volume);
    
    vector<double> result(close_price.size());
    span result_span(result);
    F(close_price_span, high_price_span, low_price_span, volume_span, result_span);
    
    return result;
}