#pragma once

#include <torch/torch.h>
#include "can_bert_for_masked_lm.h" // For BertConfigCpp

// Corresponds to BertOutput in Hugging Face
// This is the output layer of the FFN, including LayerNorm and residual connection.
class BertOutputImpl : public torch::nn::Module {
public:
    BertOutputImpl(const BertConfigCpp& config);

    // hidden_states is the output of BertIntermediate
    // input_tensor is the output of BertAttention (before BertIntermediate)
    torch::Tensor forward(const torch::Tensor& hidden_states, const torch::Tensor& input_tensor);

private:
    torch::nn::Linear dense_{nullptr};
    torch::nn::LayerNorm LayerNorm_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(BertOutput);
