#pragma once

#include <torch/torch.h>
#include <vector>
#include "can_bert_for_masked_lm.h" // For BertConfigCpp
#include "bert_layer.h"           // For BertLayer

// Corresponds to BertEncoder in Hugging Face
class BertEncoderImpl : public torch::nn::Module {
public:
    BertEncoderImpl(const BertConfigCpp& config);

    // output_attentions: whether to return all self-attention probabilities
    // output_hidden_states: whether to return all hidden_states of all layers
    // return_dict: (Python specific)
    // Other args omitted for now (head_mask, encoder_hidden_states etc.)
    // Returns a tuple: (last_hidden_state, all_hidden_states (optional), all_attentions (optional))
    // For simplicity, let's start with returning just the last_hidden_state.
    // We can extend it to return more if needed, like the Python model.
    // Let's define the output as:
    // std::tuple<torch::Tensor,                            // last_hidden_state
    //            torch::optional<std::vector<torch::Tensor>>, // all_hidden_states
    //            torch::optional<std::vector<torch::Tensor>>>  // all_attentions
    // For now, only last_hidden_state will be populated.
    std::tuple<torch::Tensor, torch::optional<std::vector<torch::Tensor>>, torch::optional<std::vector<torch::Tensor>>>
    forward(
        const torch::Tensor& hidden_states,                 // Initial embeddings from BertEmbeddings
        const torch::Tensor& attention_mask = torch::Tensor(),
        bool output_attentions = false, // Simplified control
        bool output_hidden_states = false // Simplified control
    );

private:
    torch::nn::ModuleList layer_{nullptr}; // ModuleList of BertLayer
};

TORCH_MODULE(BertEncoder);
