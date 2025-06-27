#include "mlm_dataset.h"
#include <iostream> // For cerr
#include <sstream>  // For string parsing in parse_log_line
#include <algorithm> // For std::shuffle, std::remove_if
#include <random>   // For std::mt19937, std::uniform_real_distribution

// Helper function to parse a log line (candump format by default)
CANMessage parse_log_line(const std::string& line, const std::string& dataset_type) {
    CANMessage msg;
    std::stringstream ss(line);
    std::string part;

    ss >> part;
    if (!part.empty() && part.front() == '(' && part.back() == ')') {
        msg.timestamp = part.substr(1, part.length() - 2);
    } else {
        return msg;
    }

    ss >> msg.interface;
    if (msg.interface.empty()) {
        return msg;
    }

    std::string id_data_part;
    ss >> id_data_part;
    if (id_data_part.empty()) {
        return msg;
    }

    size_t hash_pos = id_data_part.find('#');
    if (hash_pos == std::string::npos) {
        return msg;
    }

    msg.can_id_hex = id_data_part.substr(0, hash_pos);
    std::string payload_str = id_data_part.substr(hash_pos + 1);

    if (msg.can_id_hex.empty()) {
        return msg;
    }

    if (payload_str.length() % 2 != 0) {
        payload_str = payload_str.substr(0, payload_str.length() - (payload_str.length() % 2));
    }

    for (size_t i = 0; i < payload_str.length(); i += 2) {
        if (i + 2 <= payload_str.length()) {
            msg.data_bytes.push_back(payload_str.substr(i, 2));
        }
    }
    msg.dlc = msg.data_bytes.size();
    return msg;
}


MLMDataset::MLMDataset(
    const std::string& file_path,
    const CANTokenizer& tokenizer,
    int seq_len,
    const std::string& dataset_type,
    float mask_prob
) : tokenizer_(tokenizer), seq_len_(seq_len), mask_prob_(mask_prob) {

    mask_token_id_ = tokenizer_.get_token_id(CANTokenizer::MASK_TOKEN);

    for(const auto& special_token_str : tokenizer_.get_special_tokens()){
        special_token_ids_set_.push_back(tokenizer_.get_token_id(special_token_str));
    }
    std::sort(special_token_ids_set_.begin(), special_token_ids_set_.end());
    special_token_ids_set_.erase(std::unique(special_token_ids_set_.begin(), special_token_ids_set_.end()), special_token_ids_set_.end());

    load_and_tokenize_data(file_path, dataset_type);

    if (token_id_stream_.empty()) {
        std::cerr << "Warning: MLMDataset token stream is empty after loading data from: " << file_path << std::endl;
    } else {
        std::cout << "Dataset: Initialization complete. Token stream length: " << token_id_stream_.size() << std::endl;
    }
}

void MLMDataset::load_and_tokenize_data(const std::string& file_path, const std::string& dataset_type) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("MLMDataset: Could not open data file: " + file_path);
    }

    std::cout << "Dataset: Loading and tokenizing data from file: " << file_path << std::endl;
    std::string line;
    std::vector<std::string> current_message_tokens;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        CANMessage msg = parse_log_line(line, dataset_type);
        if (msg.can_id_hex.empty() && msg.data_bytes.empty()) {
            continue;
        }
        current_message_tokens = msg.to_tokens(tokenizer_.ID_OFFSET);
        std::vector<int64_t> encoded_tokens = tokenizer_.encode(current_message_tokens);
        token_id_stream_.insert(token_id_stream_.end(), encoded_tokens.begin(), encoded_tokens.end());
    }
    file.close();
}

MLMExample MLMDataset::get(size_t idx) {
    if (idx + seq_len_ > token_id_stream_.size()) {
        throw std::out_of_range("MLMDataset::get() index out of range.");
    }

    std::vector<int64_t> sequence_ids;
    sequence_ids.reserve(seq_len_);
    for(size_t i = 0; i < seq_len_; ++i) {
        sequence_ids.push_back(token_id_stream_[idx + i]);
    }

    std::vector<int64_t> input_ids = sequence_ids;
    std::vector<int64_t> labels(seq_len_, -100);

    std::vector<size_t> candidate_indices;
    for (size_t i = 0; i < input_ids.size(); ++i) {
        bool is_special = false;
        for(int64_t special_id : special_token_ids_set_){
            if(input_ids[i] == special_id){
                is_special = true;
                break;
            }
        }
        if (!is_special) {
            candidate_indices.push_back(i);
        }
    }

    int num_to_mask = static_cast<int>(std::round(static_cast<float>(candidate_indices.size()) * mask_prob_));
    if (num_to_mask == 0 && mask_prob_ > 0 && !candidate_indices.empty()) {
        num_to_mask = 1;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(candidate_indices.begin(), candidate_indices.end(), g);

    std::uniform_real_distribution<> distrib(0.0, 1.0);
    int64_t vocab_size = static_cast<int64_t>(tokenizer_.get_vocab_size());
    int64_t first_regular_token_id = special_token_ids_set_.size();

    for (int i = 0; i < num_to_mask && i < candidate_indices.size(); ++i) {
        size_t masked_idx = candidate_indices[i];
        labels[masked_idx] = input_ids[masked_idx];

        float rand_val = distrib(g);
        if (rand_val < 0.8f) {
            input_ids[masked_idx] = mask_token_id_;
        } else if (rand_val < 0.9f) {
            int64_t random_token_id;
            bool is_special_random_token;
            do {
                 std::uniform_int_distribution<int64_t> token_distrib(first_regular_token_id, vocab_size - 1);
                 random_token_id = token_distrib(g);
                 is_special_random_token = false;
                 for(int64_t special_id : special_token_ids_set_){
                     if(random_token_id == special_id){
                         is_special_random_token = true;
                         break;
                     }
                 }
            } while (is_special_random_token && (vocab_size > special_token_ids_set_.size()));
            input_ids[masked_idx] = random_token_id;
        }
    }

    std::vector<int64_t> attention_mask_data(seq_len_, 1);

    auto input_ids_tensor = torch::tensor(input_ids, torch::kLong);
    auto attention_mask_tensor = torch::tensor(attention_mask_data, torch::kLong);
    auto labels_tensor = torch::tensor(labels, torch::kLong);

    // Data is a vector of tensors: {input_ids, attention_mask}
    // Target is a single tensor: labels
    return {{input_ids_tensor, attention_mask_tensor}, labels_tensor};
}

torch::optional<size_t> MLMDataset::size() const {
    if (token_id_stream_.size() < seq_len_) {
        return 0;
    }
    return token_id_stream_.size() - seq_len_ + 1;
}
