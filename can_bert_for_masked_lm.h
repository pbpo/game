#pragma once

#include <torch/torch.h>
#include "bert_model.h"             // For actual BertModel
#include "bert_lm_prediction_head.h" // For actual BertLMPredictionHead

// BertConfigCpp structure definition
// This could be moved to its own file (e.g., bert_config.h) for better organization
struct BertConfigCpp {
    int64_t vocab_size = 30522;
    int64_t hidden_size = 768;
    int64_t num_hidden_layers = 12;
    int64_t num_attention_heads = 12;
    int64_t intermediate_size = 3072;
    double hidden_dropout_prob = 0.1;
    double attention_probs_dropout_prob = 0.1;
    int64_t max_position_embeddings = 512;
    int64_t type_vocab_size = 2;
    double layer_norm_eps = 1e-12;
    bool tie_word_embeddings = true; // Common practice for BERT MLM

    BertConfigCpp(
        int64_t vocab_size_ = 30522,
        int64_t hidden_size_ = 768,
        int64_t num_hidden_layers_ = 12,
        int64_t num_attention_heads_ = 12,
        int64_t intermediate_size_ = 3072,
        double hidden_dropout_prob_ = 0.1,
        double attention_probs_dropout_prob_ = 0.1,
        int64_t max_position_embeddings_ = 512,
        int64_t type_vocab_size_ = 2,
        double layer_norm_eps_ = 1e-12,
        bool tie_word_embeddings_ = true
    ) : vocab_size(vocab_size_),
        hidden_size(hidden_size_),
        num_hidden_layers(num_hidden_layers_),
        num_attention_heads(num_attention_heads_),
        intermediate_size(intermediate_size_),
        hidden_dropout_prob(hidden_dropout_prob_),
        attention_probs_dropout_prob(attention_probs_dropout_prob_),
        max_position_embeddings(max_position_embeddings_),
        type_vocab_size(type_vocab_size_),
        layer_norm_eps(layer_norm_eps_),
        tie_word_embeddings(tie_word_embeddings_)
    {}
};


class CANBertForMaskedLMImpl : public torch::nn::Module {
public:
    CANBertForMaskedLMImpl(const BertConfigCpp& config);

    // Returns tuple: (masked_lm_loss (optional), prediction_scores)
    std::tuple<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& input_ids,
        const torch::Tensor& attention_mask = torch::Tensor(),
        const torch::Tensor& token_type_ids = torch::Tensor(), // Added token_type_ids
        const torch::Tensor& labels = torch::Tensor(),
        bool output_attentions = false,      // Added for BertModel
        bool output_hidden_states = false    // Added for BertModel
    );

    // Helper to get word embedding weights, useful for tying weights with LM head
    torch::Tensor get_word_embedding_weights() const;

    const BertConfigCpp& config() const { return config_; }


private:
    BertConfigCpp config_; // Store a copy of the config
    BertModel bert_{nullptr}; // Using the new fully implemented BertModel
    BertLMPredictionHead cls_{nullptr}; // Using the new fully implemented BertLMPredictionHead
    torch::nn::CrossEntropyLoss loss_fct_{nullptr};
};

TORCH_MODULE(CANBertForMaskedLM);
