#include "bert_attention.h"
#include <stdexcept> // For std::runtime_error

// --- BertSelfAttentionImpl ---
BertSelfAttentionImpl::BertSelfAttentionImpl(const BertConfigCpp& config)
    : num_attention_heads_(config.num_attention_heads),
      attention_head_size_(config.hidden_size / config.num_attention_heads),
      attention_probs_dropout_prob_(config.attention_probs_dropout_prob)
{
    if (config.hidden_size % config.num_attention_heads != 0) {
        throw std::runtime_error("The hidden size (" + std::to_string(config.hidden_size) +
                                 ") is not a multiple of the number of attention heads (" +
                                 std::to_string(config.num_attention_heads) + ")");
    }

    query_ = torch::nn::Linear(config.hidden_size, config.hidden_size); // all_head_size is hidden_size
    key_ = torch::nn::Linear(config.hidden_size, config.hidden_size);
    value_ = torch::nn::Linear(config.hidden_size, config.hidden_size);
    dropout_ = torch::nn::Dropout(config.attention_probs_dropout_prob);

    register_module("query", query_);
    register_module("key", key_);
    register_module("value", value_);
    register_module("dropout", dropout_);

    scale_ = 1.0 / std::sqrt(static_cast<double>(attention_head_size_));
}

torch::Tensor BertSelfAttentionImpl::transpose_for_scores(torch::Tensor x) {
    // x shape: [batch_size, seq_len, all_head_size] (all_head_size is hidden_size)
    auto new_x_shape = x.sizes().vec();
    new_x_shape.pop_back(); // Remove last dim (all_head_size)
    new_x_shape.push_back(num_attention_heads_);
    new_x_shape.push_back(attention_head_size_);
    // new_x_shape: [batch_size, seq_len, num_attention_heads, attention_head_size]

    x = x.view(new_x_shape);
    return x.permute({0, 2, 1, 3}); // [batch_size, num_attention_heads, seq_len, attention_head_size]
}

std::tuple<torch::Tensor, torch::Tensor> BertSelfAttentionImpl::forward(
    const torch::Tensor& hidden_states,
    const torch::Tensor& attention_mask_opt
) {
    // hidden_states: [batch_size, seq_len, hidden_size]
    torch::Tensor mixed_query_layer = query_->forward(hidden_states);
    torch::Tensor mixed_key_layer = key_->forward(hidden_states);
    torch::Tensor mixed_value_layer = value_->forward(hidden_states);

    torch::Tensor query_layer = transpose_for_scores(mixed_query_layer);
    torch::Tensor key_layer = transpose_for_scores(mixed_key_layer);
    torch::Tensor value_layer = transpose_for_scores(mixed_value_layer);
    // query_layer, key_layer, value_layer: [batch_size, num_heads, seq_len, head_size]

    // Take the dot product between "query" and "key" to get the raw attention scores.
    // key_layer.transpose(-1, -2) gives [batch_size, num_heads, head_size, seq_len]
    torch::Tensor attention_scores = torch::matmul(query_layer, key_layer.transpose(-1, -2));
    attention_scores = attention_scores * scale_; // Apply scaling factor

    if (attention_mask_opt.defined() && attention_mask_opt.numel() > 0) {
        // attention_mask is expected to be [batch_size, 1, 1, to_seq_length] (for BERT encoder self-attention)
        // or [batch_size, from_seq_length, to_seq_length] for more general cases.
        // The Python code uses: attention_scores = attention_scores + attention_mask
        // where attention_mask has large negative values for masked positions.
        attention_scores = attention_scores + attention_mask_opt;
    }

    // Normalize the attention scores to probabilities.
    torch::Tensor attention_probs = torch::softmax(attention_scores, -1);

    // This is actually dropping out entire tokens to attend to, which might
    // seem unusual, but is taken from the Hugging Face Transformers implementation.
    attention_probs = dropout_->forward(attention_probs);

    // (Optional) head_mask application would go here if implemented.

    torch::Tensor context_layer = torch::matmul(attention_probs, value_layer);
    // context_layer shape: [batch_size, num_heads, seq_len, head_size]

    context_layer = context_layer.permute({0, 2, 1, 3}).contiguous();
    // context_layer shape: [batch_size, seq_len, num_heads, head_size]

    auto context_layer_shape = context_layer.sizes().vec();
    context_layer_shape.pop_back();
    context_layer_shape.pop_back();
    context_layer_shape.push_back(num_attention_heads_ * attention_head_size_); // all_head_size (hidden_size)
    // context_layer_shape: [batch_size, seq_len, all_head_size]
    context_layer = context_layer.view(context_layer_shape);

    // `output_attentions` is true, return attentions, else empty tensor for attentions.
    // For now, always returning attention_probs. Can be made optional.
    return std::make_tuple(context_layer, attention_probs);
}


// --- BertSelfOutputImpl ---
BertSelfOutputImpl::BertSelfOutputImpl(const BertConfigCpp& config) {
    dense_ = torch::nn::Linear(config.hidden_size, config.hidden_size);
    LayerNorm_ = torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({config.hidden_size}).eps(config.layer_norm_eps)
    );
    dropout_ = torch::nn::Dropout(config.hidden_dropout_prob);

    register_module("dense", dense_);
    register_module("LayerNorm", LayerNorm_);
    register_module("dropout", dropout_);
}

torch::Tensor BertSelfOutputImpl::forward(const torch::Tensor& hidden_states, const torch::Tensor& input_tensor) {
    // hidden_states is the output of BertSelfAttention's context_layer
    // input_tensor is the input to BertSelfAttention (i.e., from previous layer or embeddings)
    torch::Tensor dense_output = dense_->forward(hidden_states);
    torch::Tensor dropout_output = dropout_->forward(dense_output);
    torch::Tensor LayerNorm_output = LayerNorm_->forward(dropout_output + input_tensor); // Residual connection
    return LayerNorm_output;
}


// --- BertAttentionImpl ---
BertAttentionImpl::BertAttentionImpl(const BertConfigCpp& config) {
    self_attention_ = BertSelfAttention(config);
    self_output_ = BertSelfOutput(config); // Corrected: SelfOutput, not BertOutput

    register_module("self", self_attention_);
    register_module("output", self_output_);
}

std::tuple<torch::Tensor, torch::Tensor> BertAttentionImpl::forward(
    const torch::Tensor& hidden_states,
    const torch::Tensor& attention_mask_opt
) {
    auto self_attention_outputs = self_attention_->forward(hidden_states, attention_mask_opt);
    torch::Tensor self_attention_context = std::get<0>(self_attention_outputs);
    torch::Tensor self_attention_probs = std::get<1>(self_attention_outputs); // Might be empty if not requested

    torch::Tensor attention_output = self_output_->forward(self_attention_context, hidden_states);

    return std::make_tuple(attention_output, self_attention_probs);
}
