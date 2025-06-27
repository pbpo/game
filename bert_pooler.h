#pragma once

#include <torch/torch.h>
#include "can_bert_for_masked_lm.h" // For BertConfigCpp

// Corresponds to BertPooler in Hugging Face
// Takes the first token's hidden state ([CLS] token) and applies a Linear layer and Tanh.
// Often used for sequence classification tasks, might not be strictly necessary for Masked LM
// if only the sequence output is used by the LM head.
// However, CANBertForMaskedLM in Python uses BertModel, which might include a pooler output
// even if it's not directly used by BertLMPredictionHead.
// Let's include it for completeness with the standard BertModel structure.
class BertPoolerImpl : public torch::nn::Module {
public:
    BertPoolerImpl(const BertConfigCpp& config);

    // hidden_states is the sequence output from BertEncoder (last layer)
    // Shape: [batch_size, seq_len, hidden_size]
    torch::Tensor forward(const torch::Tensor& hidden_states);

private:
    torch::nn::Linear dense_{nullptr};
    // Tanh activation will be applied functionally: torch::tanh()
};

TORCH_MODULE(BertPooler);
