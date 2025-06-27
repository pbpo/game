#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <set> // For build_vocab unique IDs

// Forward declaration for potential DataFrame-like structure if needed,
// but build_vocab might just take raw data lists.
// For now, assuming build_vocab will take a vector of CAN IDs strings and a vector of data token lists.
// Or, more simply, a pre-processed list of all unique CAN IDs and all unique data tokens.

class CANTokenizer {
public:
    CANTokenizer();

    // Adds a token to the vocabulary. Internal helper.
    void _add_token(const std::string& token);

    // Builds vocabulary from unique CAN IDs and unique data tokens.
    // This simplifies from Python's DataFrame input.
    // Caller is responsible for extracting unique IDs and data tokens first.
    void build_vocab_from_unique_tokens(
        const std::set<std::string>& unique_can_ids, // e.g., {"01A", "02B"}
        const std::set<std::string>& unique_data_tokens // e.g., {"11", "AA", "FF"}
    );

    // Saves the vocabulary (token_to_id map) to a JSON file.
    // Requires a JSON library (e.g., nlohmann/json).
    bool save_vocab(const std::string& file_path) const;

    // Loads the vocabulary (token_to_id map) from a JSON file.
    // Requires a JSON library.
    bool load_vocab(const std::string& file_path);

    // Encodes a list of tokens into a list of integer IDs.
    std::vector<int64_t> encode(const std::vector<std::string>& tokens) const;

    // Decodes a list of integer IDs back into a list of tokens.
    std::vector<std::string> decode(const std::vector<int64_t>& ids) const;

    // Getters
    const std::unordered_map<std::string, int64_t>& get_token_to_id_map() const { return token_to_id_; }
    const std::unordered_map<int64_t, std::string>& get_id_to_token_map() const { return id_to_token_; }
    int64_t get_token_id(const std::string& token) const;
    std::string get_token(int64_t id) const;
    size_t get_vocab_size() const { return token_to_id_.size(); }

    // Constants
    static const int64_t ID_OFFSET = 260;
    static const std::string PAD_TOKEN;
    static const std::string UNK_TOKEN;
    static const std::string MASK_TOKEN;
    static const std::string VOID_TOKEN; // Added based on Python version

    const std::vector<std::string>& get_special_tokens() const { return special_tokens_; }


private:
    std::unordered_map<std::string, int64_t> token_to_id_;
    std::unordered_map<int64_t, std::string> id_to_token_;
    std::vector<std::string> special_tokens_;

    int64_t unk_token_id_ = -1; // Cache for <UNK> token ID
};
