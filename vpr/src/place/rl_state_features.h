#ifndef VPR_RL_STATE_FEATURES_H
#define VPR_RL_STATE_FEATURES_H

#include <string>

/// Multi-feature state representation for RL-based placement
struct RLStateFeatures {
    // Annealing progress (0 = start, 1 = end)
    float annealing_temperature_progress = 0.0f;

    // Timing urgency indicator (clipped to [-10, 10])
    float worst_path_slack_ratio = 0.0f;

    // Fraction of recent moves accepted (0-1)
    float recent_acceptance_rate = 0.0f;

    // Fraction or average of timing-critical blocks (0-1)
    float critical_block_density = 0.0f;

    // Which cost component dominates (-1 to 1)
    float timing_vs_wirelength_imbalance = 0.0f;

    // Stagnation indicator (clipped to [0, 1])
    float moves_since_improvement = 0.0f;

    /**
     * @brief Check if all features are valid (no NaN or Inf)
     */
    bool is_valid() const;

    /**
     * @brief Normalize slack ratio using clipping for bounded output
     * @param raw_slack Raw slack value (can be unbounded)
     * @return Clipped value in range [-1, 1]
     */
    static float normalize_slack_ratio(float raw_slack);

    /**
     * @brief Normalize moves count using simple ratio and clipping
     * @param moves Number of moves since improvement
     * @param window_size Typical window size for normalization
     * @return Normalized value in range [0, 1]
     */
    static float normalize_moves_count(int moves, int window_size = 1000);

    /**
     * @brief Serialize state features to string (for debugging/logging)
     */
    std::string to_string() const;
};

#endif /* VPR_RL_STATE_FEATURES_H */
