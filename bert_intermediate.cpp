#include "bert_intermediate.h"
#include <torch/nn/functional.h> // For torch::gelu if not using nn::GELU module

BertIntermediateImpl::BertIntermediateImpl(const BertConfigCpp& config) {
    dense_ = torch::nn::Linear(config.hidden_size, config.intermediate_size);
    register_module("dense", dense_);

    // If using torch::nn::GELU as a module:
    // gelu_activation_ = torch::nn::GELU();
    // register_module("intermediate_act_fn", gelu_activation_);
}

torch::Tensor BertIntermediateImpl::forward(const torch::Tensor& hidden_states) {
    torch::Tensor dense_output = dense_->forward(hidden_states);

    // Apply GELU activation
    // Using functional torch::gelu.
    // PyTorch's GELU implementation might be an approximation ('none', 'tanh').
    // Hugging Face default is 'torch.nn.functional.gelu' which might map to 'tanh' approx.
    // For exactness, one might need to implement the exact GELU formula if LibTorch's default differs.
    // LibTorch `torch::gelu` or `torch::nn::functional::gelu` should be suitable.
    torch::Tensor activated_output = torch::gelu(dense_output);
    // If using nn::GELU module:
    // torch::Tensor activated_output = gelu_activation_->forward(dense_output);

    return activated_output;
}
