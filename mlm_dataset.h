#pragma once

#include "can_tokenizer.h" // Assuming can_tokenizer.h is in the include path
#include <torch/torch.h>
#include <string>
#include <vector>
#include <fstream> // For reading the data file

// Define a structure to hold a single CAN message (parsed from a log line)
struct CANMessage {
    std::string timestamp; // Not used by MLMDataset directly, but good to parse
    std::string interface; // e.g., "can0", not used by MLMDataset
    std::string can_id_hex;    // e.g., "18DAF110" (without '0x')
    std::vector<std::string> data_bytes; // e.g., {"02", "01", "00", "00", "00", "00", "00", "00"}
    size_t dlc; // Data Length Code, inferred from data_bytes.size()

    // Helper to convert to a list of tokens for the tokenizer
    std::vector<std::string> to_tokens(int64_t id_offset) const {
        std::vector<std::string> frame_tokens;
        // Convert CAN ID to offset string token
        try {
            unsigned long can_id_val = std::stoul(can_id_hex, nullptr, 16);
            frame_tokens.push_back(std::to_string(can_id_val + id_offset));
        } catch (const std::exception& e) {
            frame_tokens.push_back(std::to_string(0 + id_offset)); // Fallback or log error
        }
        for (const auto& byte : data_bytes) {
            frame_tokens.push_back(byte);
        }
        return frame_tokens;
    }
};

// Function to parse a single line of a log file (candump format assumed)
CANMessage parse_log_line(const std::string& line, const std::string& dataset_type = "candump");


// Define the types for Data and Target for clarity
using MLMExampleData = std::vector<torch::Tensor>; // {input_ids, attention_mask}
using MLMExampleTarget = torch::Tensor;           // labels
using MLMExample = torch::data::Example<MLMExampleData, MLMExampleTarget>;

// LibTorch custom dataset for Masked Language Modeling
class MLMDataset : public torch::data::Dataset<MLMDataset, MLMExample> {
public:
    MLMDataset(
        const std::string& file_path,
        const CANTokenizer& tokenizer, // Pass by const reference
        int seq_len,
        const std::string& dataset_type = "candump", // "candump" or "hcrl"
        float mask_prob = 0.15
    );

    MLMExample get(size_t idx) override;

    torch::optional<size_t> size() const override;

private:
    const CANTokenizer& tokenizer_; // Store as const reference
    int seq_len_;
    float mask_prob_;
    std::vector<int64_t> token_id_stream_;

    int64_t mask_token_id_;
    std::vector<int64_t> special_token_ids_set_;

    void load_and_tokenize_data(const std::string& file_path, const std::string& dataset_type);
};
