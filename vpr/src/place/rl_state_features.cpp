#include "rl_state_features.h"
#include <cmath>
#include <algorithm>

bool RLStateFeatures::is_valid() const {
    return std::isfinite(annealing_temperature_progress) &&
           std::isfinite(worst_path_slack_ratio) &&
           std::isfinite(recent_acceptance_rate) &&
           std::isfinite(critical_block_density) &&
           std::isfinite(timing_vs_wirelength_imbalance) &&
           std::isfinite(moves_since_improvement);
}

float RLStateFeatures::normalize_slack_ratio(float raw_slack) {
    // Clipping to [-10, 10] range
    return std::max(-10.0f, std::min(10.0f, raw_slack));
}

float RLStateFeatures::normalize_moves_count(int moves, int window_size) {
    if (moves <= 0) return 0.0f;
    if (window_size <= 0) window_size = 1000;

    // Simple ratio with clipping to [0, 1]
    float ratio = static_cast<float>(moves) / static_cast<float>(window_size);
    return std::min(1.0f, ratio);
}

std::string RLStateFeatures::to_string() const {
    return "RLStateFeatures{"
           "temp_progress=" + std::to_string(annealing_temperature_progress) +
           ", slack_ratio=" + std::to_string(worst_path_slack_ratio) +
           ", accept_rate=" + std::to_string(recent_acceptance_rate) +
           ", crit_density=" + std::to_string(critical_block_density) +
           ", cost_imbalance=" + std::to_string(timing_vs_wirelength_imbalance) +
           ", moves_stale=" + std::to_string(moves_since_improvement) +
           "}";
}
