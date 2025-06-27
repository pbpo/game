#pragma once

#include <torch/torch.h>
#include "can_bert_for_masked_lm.h" // For BertConfigCpp
#include "bert_embeddings.h"      // For BertEmbeddings
#include "bert_encoder.h"         // For BertEncoder
#include "bert_pooler.h"          // For BertPooler (optional, but part of standard BertModel)

// Defines the main BertModel structure, similar to Hugging Face's BertModel.
// This will replace the placeholder BertModelCppImpl in can_bert_for_masked_lm.h
class BertModelImpl : public torch::nn::Module {
public:
    BertModelImpl(const BertConfigCpp& config, bool add_pooling_layer = true);

    // Standard BertModel output:
    // Tuple: (sequence_output, pooled_output, (optional) hidden_states, (optional) attentions)
    // sequence_output: [batch_size, seq_len, hidden_size] (last layer hidden states)
    // pooled_output: [batch_size, hidden_size] (output of BertPooler)
    // hidden_states: tuple of Tensors, one for initial embeddings, one for each layer's output
    // attentions: tuple of Tensors, one for each layer's attention probabilities
    using BertModelOutput = std::tuple<
        torch::Tensor,                                  // sequence_output (last_hidden_state)
        torch::Tensor,                                  // pooled_output
        torch::optional<std::vector<torch::Tensor>>,    // all_hidden_states
        torch::optional<std::vector<torch::Tensor>>     // all_attentions
    >;

    BertModelOutput forward(
        const torch::Tensor& input_ids,
        const torch::Tensor& attention_mask = torch::Tensor(),
        const torch::Tensor& token_type_ids = torch::Tensor(),
        bool output_attentions = false,
        bool output_hidden_states = false
    );

    // Getter for the config
    const BertConfigCpp& config() const { return config_; }

    // Getter for word embedding weights (for tying with LM head)
    torch::Tensor get_input_embeddings_weight() const;

private:
    BertConfigCpp config_; // Store the config
    BertEmbeddings embeddings_{nullptr};
    BertEncoder encoder_{nullptr};
    BertPooler pooler_{nullptr}; // Optional, based on add_pooling_layer
    bool add_pooling_layer_;
};

TORCH_MODULE(BertModel); // Changed from BertModelCpp to BertModel for clarity
