#ifndef VPR_RL_CHECKPOINT_H
#define VPR_RL_CHECKPOINT_H

#include "rl_state_features.h"

#include <string>
#include <vector>
#include <fstream>

/**
 * @brief A single RL checkpoint storing Q-values at a particular state
 *
 * Checkpoints capture the learned Q-values at specific points during training
 * so they can be used to initialize future placement runs.
 */
struct RLCheckpoint {
    RLStateFeatures state;         ///< State features at checkpoint
    std::vector<float> q_values;   ///< Q-values learned at this state
    double timestamp;              ///< When checkpoint was created (for debugging)
};

/// @brief Shared utilities for RL checkpoint operations
namespace rl_checkpoint {

/// Feature weights for state distance calculation (higher = more important)
constexpr float WEIGHT_TEMP_PROGRESS = 1.0f;
constexpr float WEIGHT_SLACK_RATIO = 1.0f;
constexpr float WEIGHT_ACCEPTANCE_RATE = 1.0f;
constexpr float WEIGHT_CRIT_DENSITY = 1.0f;
constexpr float WEIGHT_COST_IMBALANCE = 1.0f;
constexpr float WEIGHT_MOVES_STALE = 1.0f;

/**
 * @brief Compute weighted Euclidean distance between two states
 * @param s1 First state
 * @param s2 Second state
 * @return Weighted distance (lower = more similar)
 */
float compute_state_distance(const RLStateFeatures& s1, const RLStateFeatures& s2);

/**
 * @brief Parse a JSON line into a checkpoint
 * @param line JSON line in compact format
 * @param cp Output checkpoint
 * @return true if parse succeeded
 */
bool parse_checkpoint_line(const std::string& line, RLCheckpoint& cp);

}  // namespace rl_checkpoint

/**
 * @brief Manages saving and loading of RL checkpoints for Q-value initialization
 *
 * Uses a streaming design with bounded memory:
 * - In training mode, checkpoints are buffered in memory up to a threshold
 * - When the threshold is exceeded, the buffer is flushed to disk
 * - Uses JSON Lines format for append-friendly writes
 * - Destructor automatically flushes any remaining buffered checkpoints
 *
 * In inference mode, all checkpoints are loaded into memory for fast lookup.
 */
class RLCheckpointManager {
  public:
    /// Default buffer size (number of checkpoints before flushing to disk)
    static constexpr size_t DEFAULT_BUFFER_SIZE = 100;

    /**
     * @brief Construct checkpoint manager
     * @param buffer_size Number of checkpoints to buffer before flushing (training mode)
     */
    explicit RLCheckpointManager(size_t buffer_size = DEFAULT_BUFFER_SIZE);

    /// Destructor flushes any remaining buffered checkpoints
    ~RLCheckpointManager();

    // Non-copyable due to file handle
    RLCheckpointManager(const RLCheckpointManager&) = delete;
    RLCheckpointManager& operator=(const RLCheckpointManager&) = delete;

    // Movable
    RLCheckpointManager(RLCheckpointManager&&) = default;
    RLCheckpointManager& operator=(RLCheckpointManager&&) = default;

    /**
     * @brief Initialize for training mode (writing checkpoints)
     * @param filename Path to output file (will be created/truncated)
     * @return true if file opened successfully
     */
    bool init_for_training(const std::string& filename);

    /**
     * @brief Save a checkpoint with current state and Q-values
     *
     * Adds checkpoint to internal buffer. When buffer exceeds threshold,
     * automatically flushes to disk. This provides:
     * - Low average latency (just a vector push)
     * - Bounded memory usage (periodic flushes)
     * - Data durability (incremental writes)
     *
     * @param state Current state features
     * @param q_values Current Q-values from the agent
     */
    void save_checkpoint(const RLStateFeatures& state,
                         const std::vector<float>& q_values);

    /**
     * @brief Find the checkpoint nearest to the given state
     * @param state State features to match
     * @return Pointer to nearest checkpoint, or nullptr if no checkpoints
     */
    const RLCheckpoint* find_nearest_checkpoint(const RLStateFeatures& state) const;

    /**
     * @brief Load checkpoints from a file (JSON Lines or legacy JSON format)
     * @param filename Path to input file
     * @return true if successful, false otherwise
     */
    bool load(const std::string& filename);

    /// Get the number of stored checkpoints (buffer + already flushed)
    size_t num_checkpoints() const { return total_checkpoints_written_ + buffer_.size(); }

    /// Get number of checkpoints currently in buffer
    size_t buffer_count() const { return buffer_.size(); }

    /// Manually flush buffer to disk (called automatically when buffer is full)
    bool flush();

  private:
    /**
     * @brief Write a single checkpoint as a JSON line
     * @param cp Checkpoint to write
     * @return true if write succeeded
     */
    bool write_checkpoint_line(const RLCheckpoint& cp);

    /**
     * @brief Load from legacy JSON format (array of checkpoints)
     * @param filename Path to file
     * @return true if successful
     */
    bool load_legacy_format(const std::string& filename);

    /**
     * @brief Load from JSON Lines format
     * @param filename Path to file
     * @return true if successful
     */
    bool load_jsonl_format(const std::string& filename);

  private:
    std::vector<RLCheckpoint> buffer_;           ///< In-memory checkpoint buffer
    std::vector<RLCheckpoint> checkpoints_;      ///< Loaded checkpoints (inference mode)
    std::ofstream output_file_;                  ///< Output file for streaming writes
    std::string output_filename_;                ///< Path to output file
    size_t buffer_threshold_;                    ///< Flush when buffer exceeds this size
    size_t total_checkpoints_written_ = 0;       ///< Count of checkpoints flushed to disk
    bool training_mode_ = false;                 ///< True if initialized for training
};

#endif /* VPR_RL_CHECKPOINT_H */
