#include "fracdiff.h"
#include <stdint.h>
#include <cmath>
#include <vector>
#include <span>
using namespace std;


double dot_product(std::span<const double> weights, std::span<const double> values) {
    size_t N = min(weights.size(), values.size());
    double output = 0.0;
    for (size_t i = 0; i < N; i++) {
        output += weights[i] * values[i];
    }

    return output;
}

double fractional_integral(const span<const double> weights, span<const double> values) {
    auto w = span(weights);
    auto v = span(values);
    return dot_product(w, v);
}

double fractional_integral(const vector<double> & weights, const vector<double> & values) {
    auto w = span(weights);
    auto v = span(values);
    return fractional_integral(w, v);
}


vector<double> fractional_integral_weights(double order, size_t N) {
    vector<double> output(N);
    fractional_integral_weights(order, output);
    return output;
}

void fractional_integral_weights(double order, span<double> & output) {
    size_t N = output.size();
    if (N <= 0) return;

    output[0] = 1.0;
    for (size_t k = 1; k < N; k++) {
        output[k] = output[k - 1] * (k - 1 + order) / (double)k;
    }
}

void fractional_integral_weights(double order, std::vector<double> & output) {
    span s(output);
    fractional_integral_weights(order, s);
}

void fractional_derivative_weights(double order, span<double> output) {
    size_t N = output.size();
    if (N <= 0) return;

    output[0] = 1.0;
    for (size_t k = 1; k < N; k++) {
        output[k] = output[k - 1] * (k - 1 - order) / (double)k;
    }    
}

void fractional_derivative_weights(double order, vector<double> & output) {
    span s(output);
    fractional_derivative_weights(order, s);
}

vector<double> fractional_derivative_weights(double order, size_t N) {
    vector<double> output(N);
    fractional_derivative_weights(order, output);

    return output;
}

double brentq_root(unary_func_t f, double a, double b) {
    double fa = f(a);
    double fb = f(b);

    if (fa * fb >= 0) return a; // Root not bracketed

    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a;
    double fc = fa;
    bool mflag = true;
    double d = 0;
    double s = b;
    double fs = fb;
    double eps = 1e-12;
    double tol = 1e-10;

    for (int iter = 0; iter < 100; ++iter) {
        if (std::abs(fb) < tol || std::abs(b - a) < tol) return b;

        if (std::abs(fa - fc) > eps && std::abs(fb - fc) > eps) {
            // Inverse quadratic interpolation
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) +
                (b * fa * fc) / ((fb - fa) * (fb - fc)) +
                (c * fa * fb) / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        // Conditions to decide whether to use bisection
        bool cond1 = (s < (3 * a + b) / 4.0 && s > b) || (s > (3 * a + b) / 4.0 && s < b);
        bool cond2 = mflag && (std::abs(s - b) >= std::abs(b - c) / 2.0);
        bool cond3 = !mflag && (std::abs(s - b) >= std::abs(c - d) / 2.0);
        bool cond4 = mflag && (std::abs(b - c) < tol);
        bool cond5 = !mflag && (std::abs(c - d) < tol);

        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }

        fs = f(s);
        d = c;
        c = b;
        fc = fb;

        if (fa * fs < 0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }

    return NAN; 
}


double fractional_order(double Lambda, std::span<const double> L){
    size_t N = L.size();
    vector<double> weights(N);
    span w(weights);

    unary_func_t objective_function = [Lambda, &L, &w](double order) -> double {
        fractional_integral_weights(order, w);                
        double current_lambda = dot_product(w, L);
        return current_lambda - Lambda;
    };

    double order = brentq_root(objective_function, 1e-6, 1.0);
    return order; // Return the found order
}

double fractional_order(double Lambda, const vector<double> & L) {
    span l(L);
    return fractional_order(Lambda, l);
}

double fractional_order_cy(double Lambda, int N, const double * L) {
    std::span<const double> L_span(L, N);
    return fractional_order(Lambda, L_span);
}

double fractional_integral_cy(int N, const double * weights, const double * values) {
    std::span<const double> w(weights, N);
    std::span<const double> v(values, N);
    return fractional_integral(w, v);
}

void fractional_integral_weights_cy(double order, int N, double * weights){
    std::span<double> w(weights, N);
    fractional_integral_weights(order, w);
}