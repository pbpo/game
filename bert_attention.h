#pragma once

#include <torch/torch.h>
#include "can_bert_for_masked_lm.h" // For BertConfigCpp
#include <cmath> // For std::sqrt

// Corresponds to BertSelfAttention in Hugging Face
class BertSelfAttentionImpl : public torch::nn::Module {
public:
    BertSelfAttentionImpl(const BertConfigCpp& config);

    // attention_mask: Optional tensor of shape [batch_size, 1, 1, sequence_length] or [batch_size, sequence_length]
    // head_mask: Optional tensor for masking specific heads [num_heads] or [num_layers, num_heads] - not implemented here for simplicity
    // encoder_hidden_states, encoder_attention_mask: for cross-attention - not implemented here (BERT is encoder-only by default)
    // output_attentions: whether to return attention probabilities - not implemented here for simplicity
    std::tuple<torch::Tensor, torch::Tensor> forward( // returns hidden_states, attention_probs (optional)
        const torch::Tensor& hidden_states,                // [batch_size, seq_len, hidden_size]
        const torch::Tensor& attention_mask = torch::Tensor() // [batch_size, 1, 1, to_seq_length] or [batch_size, to_seq_length]
    );

private:
    torch::nn::Linear query_{nullptr};
    torch::nn::Linear key_{nullptr};
    torch::nn::Linear value_{nullptr};
    torch::nn::Dropout dropout_{nullptr};

    int64_t num_attention_heads_;
    int64_t attention_head_size_;
    double attention_probs_dropout_prob_;
    double scale_; // 1.0 / sqrt(attention_head_size_)

    torch::Tensor transpose_for_scores(torch::Tensor x);
};

TORCH_MODULE(BertSelfAttention);


// Corresponds to BertSelfOutput in Hugging Face
class BertSelfOutputImpl : public torch::nn::Module {
public:
    BertSelfOutputImpl(const BertConfigCpp& config);

    torch::Tensor forward(const torch::Tensor& hidden_states, const torch::Tensor& input_tensor);

private:
    torch::nn::Linear dense_{nullptr};
    torch::nn::LayerNorm LayerNorm_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(BertSelfOutput);


// Corresponds to BertAttention in Hugging Face (combines SelfAttention and SelfOutput)
class BertAttentionImpl : public torch::nn::Module {
public:
    BertAttentionImpl(const BertConfigCpp& config);

    // output_attentions: whether to return attention probabilities
    std::tuple<torch::Tensor, torch::Tensor> forward( // hidden_states, attention_probs (optional)
        const torch::Tensor& hidden_states,
        const torch::Tensor& attention_mask = torch::Tensor()
    );

private:
    BertSelfAttention self_attention_{nullptr};
    BertSelfOutput self_output_{nullptr};
};

TORCH_MODULE(BertAttention);
