#pragma once

#include <torch/torch.h>
#include "can_bert_for_masked_lm.h" // For BertConfigCpp

// Corresponds to BertIntermediate in Hugging Face
// This is the feed-forward layer (FFN) after self-attention.
class BertIntermediateImpl : public torch::nn::Module {
public:
    BertIntermediateImpl(const BertConfigCpp& config);

    torch::Tensor forward(const torch::Tensor& hidden_states);

private:
    torch::nn::Linear dense_{nullptr};
    // Activation function (e.g., GELU)
    // PyTorch's torch::nn::GELU can be used if available and matches.
    // Otherwise, a custom GELU implementation or approximation might be needed.
    // For simplicity, let's assume torch::gelu or torch::nn::GELU can be used.
    // We'll use the functional version torch::gelu in .cpp for now.
    // If using torch::nn::GELU(), it should be a member:
    // torch::nn::GELU gelu_activation_{nullptr};
};

TORCH_MODULE(BertIntermediate);
