#include "bert_output.h"

BertOutputImpl::BertOutputImpl(const BertConfigCpp& config) {
    dense_ = torch::nn::Linear(config.intermediate_size, config.hidden_size);
    LayerNorm_ = torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({config.hidden_size}).eps(config.layer_norm_eps)
    );
    dropout_ = torch::nn::Dropout(config.hidden_dropout_prob);

    register_module("dense", dense_);
    register_module("LayerNorm", LayerNorm_);
    register_module("dropout", dropout_);
}

torch::Tensor BertOutputImpl::forward(const torch::Tensor& hidden_states, const torch::Tensor& input_tensor) {
    // hidden_states is the output of BertIntermediate (after activation)
    // input_tensor is the output of BertAttention (before BertIntermediate FFN)

    torch::Tensor dense_output = dense_->forward(hidden_states);
    torch::Tensor dropout_output = dropout_->forward(dense_output);

    // Residual connection with the input to the FFN (output of attention block)
    torch::Tensor layer_norm_output = LayerNorm_->forward(dropout_output + input_tensor);

    return layer_norm_output;
}
