#pragma once

#include <torch/torch.h>
#include "can_bert_for_masked_lm.h" // For BertConfigCpp
#include "bert_attention.h"       // For BertAttention
#include "bert_intermediate.h"    // For BertIntermediate
#include "bert_output.h"          // For BertOutput

// Corresponds to BertLayer in Hugging Face
// A single Transformer block.
class BertLayerImpl : public torch::nn::Module {
public:
    BertLayerImpl(const BertConfigCpp& config);

    // output_attentions: whether to return attention probabilities
    // Other args (head_mask, encoder_hidden_states, etc.) omitted for simplicity now.
    std::tuple<torch::Tensor, torch::Tensor> forward( // layer_output, attention_probs (optional)
        const torch::Tensor& hidden_states,
        const torch::Tensor& attention_mask = torch::Tensor()
    );

private:
    BertAttention attention_{nullptr};
    BertIntermediate intermediate_{nullptr};
    BertOutput output_{nullptr};
    // For cross-attention, would need another BertAttention module here. Not included.
};

TORCH_MODULE(BertLayer);
