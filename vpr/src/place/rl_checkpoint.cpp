#include "rl_checkpoint.h"
#include "vtr_log.h"

#include <cmath>
#include <chrono>
#include <algorithm>
#include <sstream>

// ============================================================================
// Namespace functions (shared with rl_checkpoint_cluster)
// ============================================================================

float rl_checkpoint::compute_state_distance(const RLStateFeatures& s1,
                                            const RLStateFeatures& s2) {
    float diff_temp = s1.annealing_temperature_progress - s2.annealing_temperature_progress;
    float diff_slack = s1.worst_path_slack_ratio - s2.worst_path_slack_ratio;
    float diff_accept = s1.recent_acceptance_rate - s2.recent_acceptance_rate;
    float diff_crit = s1.critical_block_density - s2.critical_block_density;
    float diff_imbal = s1.timing_vs_wirelength_imbalance - s2.timing_vs_wirelength_imbalance;
    float diff_stale = s1.moves_since_improvement - s2.moves_since_improvement;

    // Normalize slack_ratio to similar scale as other features
    // slack_ratio is in [-10, 10], so divide by 10 to get [-1, 1]
    diff_slack /= 10.0f;

    float weighted_sum = WEIGHT_TEMP_PROGRESS * diff_temp * diff_temp +
                         WEIGHT_SLACK_RATIO * diff_slack * diff_slack +
                         WEIGHT_ACCEPTANCE_RATE * diff_accept * diff_accept +
                         WEIGHT_CRIT_DENSITY * diff_crit * diff_crit +
                         WEIGHT_COST_IMBALANCE * diff_imbal * diff_imbal +
                         WEIGHT_MOVES_STALE * diff_stale * diff_stale;

    return std::sqrt(weighted_sum);
}

bool rl_checkpoint::parse_checkpoint_line(const std::string& line, RLCheckpoint& cp) {
    // Parse compact JSON format: {"state":{...},"q":[...],"t":...}
    auto find_value = [&line](const std::string& key) -> std::string {
        size_t pos = line.find("\"" + key + "\":");
        if (pos == std::string::npos) return "";
        pos = line.find(':', pos) + 1;
        size_t end = line.find_first_of(",}]", pos);
        return line.substr(pos, end - pos);
    };

    auto find_float = [&find_value](const std::string& key) -> float {
        std::string val = find_value(key);
        if (val.empty()) return 0.0f;
        try {
            return std::stof(val);
        } catch (...) {
            return 0.0f;
        }
    };

    // Parse state (using abbreviated keys)
    cp.state.annealing_temperature_progress = find_float("atp");
    cp.state.worst_path_slack_ratio = find_float("wps");
    cp.state.recent_acceptance_rate = find_float("rar");
    cp.state.critical_block_density = find_float("cbd");
    cp.state.timing_vs_wirelength_imbalance = find_float("twi");
    cp.state.moves_since_improvement = find_float("msi");

    // Parse q_values array
    size_t q_start = line.find("\"q\":[");
    if (q_start == std::string::npos) return false;
    q_start = line.find('[', q_start) + 1;
    size_t q_end = line.find(']', q_start);
    if (q_end == std::string::npos) return false;

    cp.q_values.clear();
    std::string q_str = line.substr(q_start, q_end - q_start);
    std::stringstream ss(q_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            cp.q_values.push_back(std::stof(token));
        } catch (...) {
            // Skip malformed values
        }
    }

    if (cp.q_values.empty()) return false;

    // Parse metadata
    cp.timestamp = static_cast<double>(find_float("t"));

    return true;
}

// ============================================================================
// RLCheckpointManager implementation
// ============================================================================

RLCheckpointManager::RLCheckpointManager(size_t buffer_size)
    : buffer_threshold_(buffer_size) {
    buffer_.reserve(buffer_size);
}

RLCheckpointManager::~RLCheckpointManager() {
    // Flush any remaining buffered checkpoints
    if (training_mode_ && !buffer_.empty()) {
        flush();
    }
    if (output_file_.is_open()) {
        output_file_.close();
    }
}

bool RLCheckpointManager::init_for_training(const std::string& filename) {
    if (filename.empty()) {
        return false;
    }

    output_filename_ = filename;
    output_file_.open(filename, std::ios::out | std::ios::trunc);
    if (!output_file_.is_open()) {
        VTR_LOG_ERROR("Failed to open checkpoint file for writing: %s\n", filename.c_str());
        return false;
    }

    training_mode_ = true;
    total_checkpoints_written_ = 0;
    buffer_.clear();

    return true;
}

void RLCheckpointManager::save_checkpoint(const RLStateFeatures& state,
                                          const std::vector<float>& q_values) {
    if (!training_mode_) {
        return;
    }

    RLCheckpoint checkpoint;
    checkpoint.state = state;
    checkpoint.q_values = q_values;

    // Record timestamp for debugging
    auto now = std::chrono::system_clock::now();
    checkpoint.timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();

    buffer_.push_back(std::move(checkpoint));

    // Flush to disk if buffer exceeds threshold
    if (buffer_.size() >= buffer_threshold_) {
        flush();
    }
}

bool RLCheckpointManager::flush() {
    if (!training_mode_ || !output_file_.is_open() || buffer_.empty()) {
        return true;
    }

    for (const auto& cp : buffer_) {
        if (!write_checkpoint_line(cp)) {
            VTR_LOG_ERROR("Failed to write checkpoint to file\n");
            return false;
        }
    }

    // Ensure data is written to disk
    output_file_.flush();

    total_checkpoints_written_ += buffer_.size();
    buffer_.clear();

    return true;
}

bool RLCheckpointManager::write_checkpoint_line(const RLCheckpoint& cp) {
    if (!output_file_.is_open()) {
        return false;
    }

    // Write as single-line JSON (JSON Lines format)
    output_file_ << "{";
    output_file_ << "\"state\":{";
    output_file_ << "\"atp\":" << cp.state.annealing_temperature_progress << ",";
    output_file_ << "\"wps\":" << cp.state.worst_path_slack_ratio << ",";
    output_file_ << "\"rar\":" << cp.state.recent_acceptance_rate << ",";
    output_file_ << "\"cbd\":" << cp.state.critical_block_density << ",";
    output_file_ << "\"twi\":" << cp.state.timing_vs_wirelength_imbalance << ",";
    output_file_ << "\"msi\":" << cp.state.moves_since_improvement;
    output_file_ << "},";
    output_file_ << "\"q\":[";
    for (size_t i = 0; i < cp.q_values.size(); ++i) {
        output_file_ << cp.q_values[i];
        if (i < cp.q_values.size() - 1) output_file_ << ",";
    }
    output_file_ << "],";
    output_file_ << "\"t\":" << cp.timestamp;
    output_file_ << "}\n";

    return output_file_.good();
}

bool RLCheckpointManager::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        VTR_LOG_ERROR("Failed to open checkpoint file for reading: %s\n", filename.c_str());
        return false;
    }

    checkpoints_.clear();

    // Peek at first character to determine format
    char first_char = '\0';
    file >> std::ws;  // Skip whitespace
    first_char = file.peek();
    file.close();

    if (first_char == '{') {
        // Could be JSON Lines (each line starts with '{') or legacy format
        // Try JSON Lines first
        if (load_jsonl_format(filename)) {
            return true;
        }
        // Fall back to legacy format
        return load_legacy_format(filename);
    } else {
        VTR_LOG_ERROR("Unknown checkpoint file format: %s\n", filename.c_str());
        return false;
    }
}

bool RLCheckpointManager::load_jsonl_format(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    checkpoints_.clear();
    std::string line;
    int line_num = 0;

    while (std::getline(file, line)) {
        line_num++;
        // Skip empty lines
        if (line.empty() || line.find_first_not_of(" \t\n\r") == std::string::npos) {
            continue;
        }

        RLCheckpoint cp;
        if (rl_checkpoint::parse_checkpoint_line(line, cp)) {
            checkpoints_.push_back(std::move(cp));
        } else {
            // If first line fails to parse as JSON Lines, this might be legacy format
            if (line_num == 1) {
                checkpoints_.clear();
                return false;
            }
            VTR_LOG_WARN("Failed to parse checkpoint at line %d, skipping\n", line_num);
        }
    }

    if (!checkpoints_.empty()) {
        VTR_LOG("Loaded %zu RL checkpoints from %s (JSON Lines format)\n",
                checkpoints_.size(), filename.c_str());
        return true;
    }

    return false;
}


bool RLCheckpointManager::load_legacy_format(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    checkpoints_.clear();

    // Read entire file
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    // Find each checkpoint block (legacy format uses full key names)
    size_t pos = 0;
    while ((pos = content.find("\"state\":", pos)) != std::string::npos) {
        RLCheckpoint cp;

        // Parse state features
        auto parse_float = [&content](size_t start, const std::string& key) -> float {
            size_t key_pos = content.find(key, start);
            if (key_pos == std::string::npos) return 0.0f;
            size_t colon = content.find(':', key_pos);
            if (colon == std::string::npos) return 0.0f;
            size_t end = content.find_first_of(",}\n", colon + 1);
            std::string val = content.substr(colon + 1, end - colon - 1);
            try {
                return std::stof(val);
            } catch (...) {
                return 0.0f;
            }
        };

        cp.state.annealing_temperature_progress = parse_float(pos, "annealing_temperature_progress");
        cp.state.worst_path_slack_ratio = parse_float(pos, "worst_path_slack_ratio");
        cp.state.recent_acceptance_rate = parse_float(pos, "recent_acceptance_rate");
        cp.state.critical_block_density = parse_float(pos, "critical_block_density");
        cp.state.timing_vs_wirelength_imbalance = parse_float(pos, "timing_vs_wirelength_imbalance");
        cp.state.moves_since_improvement = parse_float(pos, "moves_since_improvement");

        // Parse q_values array
        size_t q_start = content.find("\"q_values\":", pos);
        if (q_start != std::string::npos) {
            size_t arr_start = content.find('[', q_start);
            size_t arr_end = content.find(']', arr_start);
            if (arr_start != std::string::npos && arr_end != std::string::npos) {
                std::string arr_content = content.substr(arr_start + 1, arr_end - arr_start - 1);
                std::stringstream ss(arr_content);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    // Trim whitespace
                    token.erase(0, token.find_first_not_of(" \t\n"));
                    token.erase(token.find_last_not_of(" \t\n") + 1);
                    if (!token.empty()) {
                        try {
                            cp.q_values.push_back(std::stof(token));
                        } catch (...) {
                            // Skip malformed values
                        }
                    }
                }
            }
        }

        // Parse num_samples and timestamp
        cp.timestamp = static_cast<double>(parse_float(pos, "timestamp"));

        if (!cp.q_values.empty()) {
            checkpoints_.push_back(std::move(cp));
        }

        pos = content.find("}", pos) + 1;
    }

    if (!checkpoints_.empty()) {
        VTR_LOG("Loaded %zu RL checkpoints from %s (legacy format)\n",
                checkpoints_.size(), filename.c_str());
        return true;
    }

    return false;
}

const RLCheckpoint* RLCheckpointManager::find_nearest_checkpoint(const RLStateFeatures& state) const {
    if (checkpoints_.empty()) {
        return nullptr;
    }

    const RLCheckpoint* nearest = nullptr;
    float min_distance = std::numeric_limits<float>::max();

    for (const auto& checkpoint : checkpoints_) {
        float distance = rl_checkpoint::compute_state_distance(state, checkpoint.state);
        if (distance < min_distance) {
            min_distance = distance;
            nearest = &checkpoint;
        }
    }

    return nearest;
}
