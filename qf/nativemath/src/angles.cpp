#include "angles.h"
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

using namespace std;

void step1_calculate_extreme_indexes(span<const double> high_price, span<const double> low_price, int32_t N, vector<vector<int32_t>>& indexes) {
    for (int32_t t = 1; t < N; t++) {
        int32_t h_up = -1, h_down = -1, l_up = -1, l_down = -1;
        for (int32_t i = t - 1; i >= 0; i--) {
            if (h_up == -1 && high_price[i] > high_price[t]) h_up = i;
            if (h_down == -1 && high_price[i] < high_price[t]) h_down = i;
            if (l_up == -1 && low_price[i] > low_price[t]) l_up = i;
            if (l_down == -1 && low_price[i] < low_price[t]) l_down = i;
            if (h_up != -1 && h_down != -1 && l_up != -1 && l_down != -1) break;
        }
        indexes[t][0] = h_up;
        indexes[t][1] = h_down;
        indexes[t][2] = l_up;
        indexes[t][3] = l_down;
    }
}

void step2_calculate_B(const vector<vector<int32_t>>& indexes, int32_t N, vector<double>& B) {
    for (int32_t t = 1; t < N; t++) {
        int32_t min_idx = t; 
        bool found = false;
        for (int32_t j = 0; j < 4; j++) {
            if (indexes[t][j] != -1) {
                if (indexes[t][j] < min_idx) {
                    min_idx = indexes[t][j];
                }
                found = true;
            }
        }
        if (found) {
            B[t] = (double)(t - min_idx);
        }
    }
}

void step3_calculate_b(const vector<vector<int32_t>>& indexes, const vector<double>& B, int32_t N, vector<vector<double>>& b) {
    for (int32_t t = 1; t < N; t++) {
        if (B[t] > 0) {
            for (int32_t j = 0; j < 4; j++) {
                if (indexes[t][j] != -1) {
                    b[t][j] = (double)(t - indexes[t][j]) / B[t];
                }
            }
        }
    }
}

void step4_calculate_C(span<const double> high_price, span<const double> low_price, const vector<vector<int32_t>>& indexes, int32_t N, vector<double>& C) {
    for (int32_t t = 1; t < N; t++) {
        double max_diff = 0.0;
        if (indexes[t][0] != -1) max_diff = std::max(max_diff, std::abs(high_price[indexes[t][0]] - high_price[t]));
        if (indexes[t][1] != -1) max_diff = std::max(max_diff, std::abs(high_price[indexes[t][1]] - high_price[t]));
        if (indexes[t][2] != -1) max_diff = std::max(max_diff, std::abs(low_price[indexes[t][2]] - low_price[t]));
        if (indexes[t][3] != -1) max_diff = std::max(max_diff, std::abs(low_price[indexes[t][3]] - low_price[t]));
        C[t] = max_diff;
    }
}

void step5_calculate_c(span<const double> high_price, span<const double> low_price, const vector<vector<int32_t>>& indexes, const vector<double>& C, int32_t N, vector<vector<double>>& c) {
    double eps = std::numeric_limits<double>::epsilon();
    for (int32_t t = 1; t < N; t++) {
        if (C[t] > 0) {
            if (indexes[t][0] != -1) c[t][0] = (high_price[indexes[t][0]] - high_price[t]) / (C[t] + eps);
            if (indexes[t][1] != -1) c[t][1] = (high_price[indexes[t][1]] - high_price[t]) / (C[t] + eps);
            if (indexes[t][2] != -1) c[t][2] = (low_price[indexes[t][2]] - low_price[t]) / (C[t] + eps);
            if (indexes[t][3] != -1) c[t][3] = (low_price[indexes[t][3]] - low_price[t]) / (C[t] + eps);
        }
    }
}

void step6_calculate_angles(const vector<vector<double>>& b, const vector<vector<double>>& c, int32_t N, span<double> angles) {
    for (int32_t t = 1; t < N; t++) {
        for (int32_t j = 0; j < 4; j++) {
            if (!std::isnan(b[t][j]) && !std::isnan(c[t][j])) {
                angles[t * 4 + j] = std::atan2(b[t][j], c[t][j]);
            }
        }
    }
}

void calculate_price_time_angles(
    /* in */ std::span<const double> close_price, 
    /* in */ std::span<const double> high_price, 
    /* in */ std::span<const double> low_price, 
    /* out */ std::span<double> angles
) {

#ifdef _WINDOWS
    int32_t N = (int32_t)close_price.size();
#else
    int32_t N = close_price.size();
#endif    
    if (N <= 0) return;

    std::fill(angles.begin(), angles.end(), std::numeric_limits<double>::quiet_NaN());

    vector<vector<int32_t>> indexes(N, vector<int32_t>(4, -1));
    vector<double> B(N, -1.0);
    vector<double> C(N, 0.0);
    vector<vector<double>> b(N, vector<double>(4, std::numeric_limits<double>::quiet_NaN()));
    vector<vector<double>> c(N, vector<double>(4, std::numeric_limits<double>::quiet_NaN()));
    
    step1_calculate_extreme_indexes(high_price, low_price, N, indexes);
    step2_calculate_B(indexes, N, B);
    step3_calculate_b(indexes, B, N, b);
    step4_calculate_C(high_price, low_price, indexes, N, C);
    step5_calculate_c(high_price, low_price, indexes, C, N, c);
    step6_calculate_angles(b, c, N, angles);
}

vector<double> calculate_price_time_angles(
    std::span<const double> close_price, 
    std::span<const double> high_price, 
    std::span<const double> low_price
) {
#ifdef _WINDOWS
    int32_t N = (int32_t)close_price.size();
#else
    int32_t N = close_price.size();
#endif 
    
    vector<double> angles(N * 4); 
    
    // std::vector implicitly converts to std::span for the output argument
    calculate_price_time_angles(close_price, high_price, low_price, angles);
    return angles;
}

void step1_calculate_volume_extreme_indexes(span<const double> volume, int32_t N, vector<vector<int32_t>>& indexes) {
    for (int32_t t = 1; t < N; t++) {
        int32_t v_up = -1, v_down = -1;
        for (int32_t i = t - 1; i >= 0; i--) {
            if (v_up == -1 && volume[i] > volume[t]) v_up = i;
            if (v_down == -1 && volume[i] < volume[t]) v_down = i;
            if (v_up != -1 && v_down != -1) break;
        }
        indexes[t][0] = v_up;
        indexes[t][1] = v_down;
    }
}

void step2_calculate_volume_angles(span<const double> volume, const vector<vector<int32_t>>& indexes, int32_t N, span<double> angles) {
    double eps = std::numeric_limits<double>::epsilon();

    for (int32_t t = 1; t < N; t++) {
        int32_t min_idx = t;
        bool found = false;
        
        for (int32_t j = 0; j < 2; j++) {
            if (indexes[t][j] != -1) {
                if (indexes[t][j] < min_idx) min_idx = indexes[t][j];
                found = true;
            }
        }

        if (!found) continue;

        double B_t = (double)(t - min_idx);
        double V_t = 0.0;
        
        for (int32_t j = 0; j < 2; j++) {
            if (indexes[t][j] != -1) {
                V_t = std::max(V_t, std::abs(volume[indexes[t][j]] - volume[t]));
            }
        }

        for (int32_t j = 0; j < 2; j++) {
            if (indexes[t][j] != -1 && V_t > 0) {
                double b_tj = (double)(t - indexes[t][j]) / B_t;
                double v_tj = (volume[indexes[t][j]] - volume[t]) / (V_t + eps);
                angles[t * 2 + j] = std::atan2(b_tj, v_tj);
            }
        }
    }
}

void calculate_volume_time_angles(
    /* in */ const std::span<const double> volume, 
    /* out */ std::span<double> angles
) {
    #ifdef _WINDOWS
        int32_t N = (int32_t)volume.size();
    #else
        int32_t N = volume.size();
    #endif    
    if (N <= 0) return;

    std::fill(angles.begin(), angles.end(), std::numeric_limits<double>::quiet_NaN());

    vector<vector<int32_t>> indexes(N, vector<int32_t>(2, -1));

    step1_calculate_volume_extreme_indexes(volume, N, indexes);
    step2_calculate_volume_angles(volume, indexes, N, angles);
}

vector<double> calculate_volume_time_angles(std::span<const double> volume) {
    #ifdef _WINDOWS
        int32_t N = (int32_t)volume.size();
    #else
        int32_t N = volume.size();
    #endif 
    vector<double> angles(N * 2); 
    
    // std::vector implicitly converts to std::span for the output argument
    calculate_volume_time_angles(volume, angles);
    return angles;
}