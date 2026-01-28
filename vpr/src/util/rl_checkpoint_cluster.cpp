/**
 * @file rl_checkpoint_cluster.cpp
 * @brief k-means clustering tool for RL checkpoint files
 *
 * This tool clusters checkpoints from a training run to reduce the number of
 * checkpoints while preserving representative Q-values. Uses mini-batch k-means
 * with streaming I/O for memory efficiency.
 *
 * Usage: rl_checkpoint_cluster <input.jsonl> <output.jsonl> <num_clusters>
 *
 * Algorithm:
 * 1. First pass: Count checkpoints, sample initial centroids using reservoir sampling
 * 2. Second pass: Mini-batch k-means iterations with streaming reads
 * 3. Third pass: Assign all points to final clusters and compute averaged Q-values
 * 4. Write clustered centroids to output file
 *
 * This tool reuses RLCheckpoint, RLStateFeatures, and shared functions from VPR's
 * placement code to ensure consistency with the checkpoint file format.
 */

#include "rl_checkpoint.h"
#include "rl_state_features.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

/**
 * @brief Centroid for k-means clustering
 *
 * Stores accumulated state features and Q-values for averaging.
 * Uses the same RLStateFeatures structure as the VPR checkpoint system.
 */
struct Centroid {
    RLStateFeatures state;
    std::vector<double> q_value_sum;  // Use double for accumulation precision
    size_t count;

    Centroid() noexcept : count(0) {}

    void reset(size_t q_size) {
        state = RLStateFeatures();  // Reset to defaults
        q_value_sum.assign(q_size, 0.0);
        count = 0;
    }
};



/**
 * @brief Write a centroid as a checkpoint line (JSON Lines format)
 */
static void write_checkpoint_line(std::ostream& out, const Centroid& centroid) {
    out << "{";
    out << "\"state\":{";
    out << "\"atp\":" << centroid.state.annealing_temperature_progress << ",";
    out << "\"wps\":" << centroid.state.worst_path_slack_ratio << ",";
    out << "\"rar\":" << centroid.state.recent_acceptance_rate << ",";
    out << "\"cbd\":" << centroid.state.critical_block_density << ",";
    out << "\"twi\":" << centroid.state.timing_vs_wirelength_imbalance << ",";
    out << "\"msi\":" << centroid.state.moves_since_improvement;
    out << "},";
    out << "\"q\":[";
    for (size_t i = 0; i < centroid.q_value_sum.size(); ++i) {
        // Output averaged Q-value
        float avg_q = (centroid.count > 0) ?
            static_cast<float>(centroid.q_value_sum[i] / centroid.count) : 0.0f;
        out << avg_q;
        if (i < centroid.q_value_sum.size() - 1) out << ",";
    }
    out << "],";
    // Use current time as timestamp
    auto now = std::chrono::system_clock::now();
    double timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();
    out << "\"t\":" << timestamp;
    out << "}\n";
}

/**
 * @brief Count checkpoints and determine Q-value size by reading first line
 */
static bool first_pass(const std::string& filename, size_t& count, size_t& q_size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open input file: " << filename << std::endl;
        return false;
    }

    count = 0;
    q_size = 0;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line.find_first_not_of(" \t\n\r") == std::string::npos) {
            continue;
        }

        // Parse first valid line to get Q-value size
        if (count == 0) {
            RLCheckpoint cp;
            if (!rl_checkpoint::parse_checkpoint_line(line, cp)) {
                std::cerr << "Error: Cannot parse first checkpoint line" << std::endl;
                return false;
            }
            q_size = cp.q_values.size();
        }
        ++count;
    }

    return count > 0 && q_size > 0;
}

/**
 * @brief Initialize centroids using reservoir sampling
 *
 * This is a memory-efficient alternative to k-means++ for very large datasets.
 * Randomly samples k checkpoints from the file in a single pass.
 */
static bool initialize_centroids_reservoir(const std::string& filename,
                                           std::vector<Centroid>& centroids,
                                           size_t k, size_t q_size,
                                           std::mt19937& rng) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    // Initialize centroids with proper Q-value size
    centroids.resize(k);
    for (auto& c : centroids) {
        c.reset(q_size);
    }

    // Reservoir sampling: keep k random samples
    std::vector<RLCheckpoint> reservoir(k);
    std::string line;
    size_t seen = 0;

    while (std::getline(file, line)) {
        if (line.empty() || line.find_first_not_of(" \t\n\r") == std::string::npos) {
            continue;
        }

        RLCheckpoint cp;
        if (!rl_checkpoint::parse_checkpoint_line(line, cp)) continue;

        if (seen < k) {
            // Fill reservoir initially
            reservoir[seen] = std::move(cp);
        } else {
            // Replace with probability k/seen
            std::uniform_int_distribution<size_t> dist(0, seen);
            size_t j = dist(rng);
            if (j < k) {
                reservoir[j] = std::move(cp);
            }
        }
        ++seen;
    }

    // Copy reservoir to centroids
    for (size_t i = 0; i < k && i < reservoir.size(); ++i) {
        centroids[i].state = reservoir[i].state;
        centroids[i].q_value_sum.assign(reservoir[i].q_values.begin(),
                                        reservoir[i].q_values.end());
        centroids[i].count = 1;
    }

    return true;
}

/**
 * @brief Find nearest centroid for a given state
 */
static size_t find_nearest_centroid(const RLStateFeatures& state,
                                    const std::vector<Centroid>& centroids) {
    size_t nearest = 0;
    float min_dist = std::numeric_limits<float>::max();

    for (size_t i = 0; i < centroids.size(); ++i) {
        float dist = rl_checkpoint::compute_state_distance(state, centroids[i].state);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = i;
        }
    }

    return nearest;
}

/**
 * @brief Accumulate state features into centroid
 */
static void accumulate_state(Centroid& centroid, const RLStateFeatures& state) {
    centroid.state.annealing_temperature_progress += state.annealing_temperature_progress;
    centroid.state.worst_path_slack_ratio += state.worst_path_slack_ratio;
    centroid.state.recent_acceptance_rate += state.recent_acceptance_rate;
    centroid.state.critical_block_density += state.critical_block_density;
    centroid.state.timing_vs_wirelength_imbalance += state.timing_vs_wirelength_imbalance;
    centroid.state.moves_since_improvement += state.moves_since_improvement;
}

/**
 * @brief Average accumulated state features
 */
static void average_state(Centroid& centroid) {
    if (centroid.count > 0) {
        float n = static_cast<float>(centroid.count);
        centroid.state.annealing_temperature_progress /= n;
        centroid.state.worst_path_slack_ratio /= n;
        centroid.state.recent_acceptance_rate /= n;
        centroid.state.critical_block_density /= n;
        centroid.state.timing_vs_wirelength_imbalance /= n;
        centroid.state.moves_since_improvement /= n;
    }
}

/**
 * @brief Accumulate a single checkpoint into the appropriate new centroid
 */
static inline void accumulate_checkpoint_into_new_centroids(
        const RLCheckpoint& checkpoint,
        const std::vector<Centroid>& centroids,
        std::vector<Centroid>& new_centroids,
        size_t q_size) {
    // Validate Q-value vector size to avoid out-of-bounds
    if (checkpoint.q_values.size() != q_size) {
        return;  // Skip malformed entries
    }

    // Find nearest centroid
    size_t nearest = find_nearest_centroid(checkpoint.state, centroids);

    // Accumulate state features
    accumulate_state(new_centroids[nearest], checkpoint.state);

    // Accumulate Q-values
    for (size_t q = 0; q < checkpoint.q_values.size(); ++q) {
        new_centroids[nearest].q_value_sum[q] += checkpoint.q_values[q];
    }

    // Increment count
    new_centroids[nearest].count++;
}

/**
 * @brief Mini-batch k-means iteration
 *
 * Reads a batch of checkpoints, assigns to nearest centroids, and updates
 * centroids incrementally. This is more memory-efficient than standard k-means.
 */
static bool mini_batch_kmeans_iteration(const std::string& filename,
                                        std::vector<Centroid>& centroids,
                                        size_t batch_size) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    size_t q_size = centroids[0].q_value_sum.size();

    // Temporary accumulators for this iteration
    std::vector<Centroid> new_centroids(centroids.size());
    for (size_t i = 0; i < centroids.size(); ++i) {
        new_centroids[i].reset(q_size);
    }

    std::string line;
    std::vector<RLCheckpoint> batch;
    batch.reserve(batch_size);

    while (std::getline(file, line)) {
        if (line.empty() || line.find_first_not_of(" \t\n\r") == std::string::npos) {
            continue;
        }

        RLCheckpoint cp;
        if (!rl_checkpoint::parse_checkpoint_line(line, cp)) continue;
        batch.push_back(std::move(cp));

        // Process batch when full
        if (batch.size() >= batch_size) {
            for (const auto& checkpoint : batch) {
                accumulate_checkpoint_into_new_centroids(
                    checkpoint, centroids, new_centroids, q_size);
            }
            batch.clear();
        }
    }

    // Process remaining batch
    for (const auto& checkpoint : batch) {
        accumulate_checkpoint_into_new_centroids(
            checkpoint, centroids, new_centroids, q_size);
    }

    // Update centroids with averaged values
    for (size_t i = 0; i < centroids.size(); ++i) {
        if (new_centroids[i].count > 0) {
            average_state(new_centroids[i]);
            centroids[i].state = new_centroids[i].state;
            centroids[i].q_value_sum = std::move(new_centroids[i].q_value_sum);
            centroids[i].count = new_centroids[i].count;
        }
        // If count is 0, keep the centroid unchanged (no points assigned)
    }

    return true;
}

/**
 * @brief Final pass: assign all points to centroids and compute averaged Q-values
 */
static bool final_assignment_pass(const std::string& filename,
                                  std::vector<Centroid>& centroids) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    size_t q_size = centroids[0].q_value_sum.size();

    // Save current centroid positions for assignment
    std::vector<RLStateFeatures> centroid_positions(centroids.size());
    for (size_t i = 0; i < centroids.size(); ++i) {
        centroid_positions[i] = centroids[i].state;
    }

    // Reset centroids for final accumulation
    for (auto& c : centroids) {
        c.reset(q_size);
    }

    // Assign all points and accumulate
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line.find_first_not_of(" \t\n\r") == std::string::npos) {
            continue;
        }

        RLCheckpoint cp;
        if (!rl_checkpoint::parse_checkpoint_line(line, cp)) continue;

        // Find nearest centroid using saved positions
        size_t nearest = 0;
        float min_dist = std::numeric_limits<float>::max();
        for (size_t i = 0; i < centroid_positions.size(); ++i) {
            float dist = rl_checkpoint::compute_state_distance(cp.state, centroid_positions[i]);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = i;
            }
        }

        // Accumulate
        accumulate_state(centroids[nearest], cp.state);
        for (size_t q = 0; q < cp.q_values.size() &&
             q < centroids[nearest].q_value_sum.size(); ++q) {
            centroids[nearest].q_value_sum[q] += cp.q_values[q];
        }
        centroids[nearest].count++;
    }

    // Compute final averages for state features
    for (auto& c : centroids) {
        average_state(c);
    }

    return true;
}

/**
 * @brief Write clustered centroids to output file
 */
static bool write_output(const std::string& filename,
                         const std::vector<Centroid>& centroids) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open output file: " << filename << std::endl;
        return false;
    }

    file << std::setprecision(8);

    for (const auto& centroid : centroids) {
        if (centroid.count > 0) {
            write_checkpoint_line(file, centroid);
        }
    }

    return file.good();
}

static void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name
              << " <input.jsonl> <output.jsonl> <num_clusters> [options]\n"
              << "\n"
              << "Options:\n"
              << "  --iterations N    Number of k-means iterations (default: 10)\n"
              << "  --batch-size N    Mini-batch size (default: 1000)\n"
              << "  --seed N          Random seed (default: time-based)\n"
              << "  --verbose         Print progress information\n"
              << "\n"
              << "Clusters RL checkpoint states and averages Q-values per cluster.\n"
              << "Uses mini-batch k-means for memory efficiency with large files.\n";
}

int main(int argc, char* argv[]) {
    // Default parameters
    size_t num_clusters = 0;
    size_t num_iterations = 10;
    size_t batch_size = 1000;
    unsigned int seed = std::random_device{}();
    bool verbose = false;
    std::string input_file;
    std::string output_file;

    // Parse arguments
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--iterations" && i + 1 < argc) {
            num_iterations = std::stoul(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::stoul(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoul(argv[++i]);
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] != '-') {
            // Positional argument
            switch (positional) {
                case 0: input_file = arg; break;
                case 1: output_file = arg; break;
                case 2: num_clusters = std::stoul(arg); break;
                default:
                    std::cerr << "Unknown positional argument: " << arg << std::endl;
                    print_usage(argv[0]);
                    return 1;
            }
            ++positional;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (positional < 3) {
        std::cerr << "Error: Missing required arguments\n";
        print_usage(argv[0]);
        return 1;
    }

    if (num_clusters == 0) {
        std::cerr << "Error: num_clusters must be > 0\n";
        return 1;
    }

    // Initialize RNG
    std::mt19937 rng(seed);

    // First pass: count checkpoints and get Q-value size
    size_t total_count, q_size;
    if (verbose) {
        std::cout << "First pass: counting checkpoints..." << std::endl;
    }

    if (!first_pass(input_file, total_count, q_size)) {
        std::cerr << "Error: Failed to read input file or file is empty" << std::endl;
        return 1;
    }

    if (verbose) {
        std::cout << "Found " << total_count << " checkpoints with "
                  << q_size << " Q-values each" << std::endl;
    }

    // Adjust num_clusters if needed
    if (num_clusters > total_count) {
        std::cerr << "Warning: num_clusters (" << num_clusters
                  << ") > total checkpoints (" << total_count
                  << "), reducing to " << total_count << std::endl;
        num_clusters = total_count;
    }

    // Initialize centroids using reservoir sampling
    if (verbose) {
        std::cout << "Initializing " << num_clusters
                  << " centroids using reservoir sampling..." << std::endl;
    }

    std::vector<Centroid> centroids;
    if (!initialize_centroids_reservoir(input_file, centroids, num_clusters,
                                        q_size, rng)) {
        std::cerr << "Error: Failed to initialize centroids" << std::endl;
        return 1;
    }

    // Mini-batch k-means iterations
    if (verbose) {
        std::cout << "Running " << num_iterations
                  << " mini-batch k-means iterations..." << std::endl;
    }

    for (size_t iter = 0; iter < num_iterations; ++iter) {
        if (!mini_batch_kmeans_iteration(input_file, centroids, batch_size)) {
            std::cerr << "Error: Failed during k-means iteration " << iter << std::endl;
            return 1;
        }

        if (verbose) {
            std::cout << "  Iteration " << (iter + 1) << "/" << num_iterations
                      << " complete" << std::endl;
        }
    }

    // Final assignment pass
    if (verbose) {
        std::cout << "Final pass: computing averaged Q-values..." << std::endl;
    }

    if (!final_assignment_pass(input_file, centroids)) {
        std::cerr << "Error: Failed during final assignment pass" << std::endl;
        return 1;
    }

    // Write output
    if (verbose) {
        std::cout << "Writing " << num_clusters << " clustered checkpoints to "
                  << output_file << "..." << std::endl;
    }

    if (!write_output(output_file, centroids)) {
        std::cerr << "Error: Failed to write output file" << std::endl;
        return 1;
    }

    // Print summary
    if (verbose) {
        std::cout << "\nClustering complete!\n"
                  << "  Input: " << total_count << " checkpoints\n"
                  << "  Output: " << num_clusters << " clustered checkpoints\n"
                  << "  Reduction: " << std::fixed << std::setprecision(1)
                  << (100.0 * (1.0 - static_cast<double>(num_clusters) / total_count))
                  << "%\n";

        // Print cluster sizes
        std::cout << "\nCluster sizes:\n";
        for (size_t i = 0; i < centroids.size(); ++i) {
            if (centroids[i].count > 0) {
                std::cout << "  Cluster " << i << ": " << centroids[i].count
                          << " checkpoints (" << std::setprecision(1)
                          << (100.0 * centroids[i].count / total_count) << "%)\n";
            }
        }
    }

    return 0;
}
