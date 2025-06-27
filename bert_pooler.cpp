#include "bert_pooler.h"

BertPoolerImpl::BertPoolerImpl(const BertConfigCpp& config) {
    dense_ = torch::nn::Linear(config.hidden_size, config.hidden_size);
    register_module("dense", dense_);
    // Activation (tanh) is applied functionally in forward pass
}

torch::Tensor BertPoolerImpl::forward(const torch::Tensor& hidden_states) {
    // We "pool" the model by simply taking the hidden state corresponding
    // to the first token. (CLS token)
    // hidden_states shape: [batch_size, seq_len, hidden_size]
    // first_token_tensor shape: [batch_size, hidden_size]
    torch::Tensor first_token_tensor = hidden_states.select(/*dim=*/1, /*index=*/0);
    // Alternative using slice: hidden_states.slice(/*dim=*/1, /*start=*/0, /*end=*/1).squeeze(1);

    torch::Tensor pooled_output = dense_->forward(first_token_tensor);
    pooled_output = torch::tanh(pooled_output);

    return pooled_output;
}
