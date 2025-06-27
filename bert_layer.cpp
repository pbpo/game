#include "bert_layer.h"

BertLayerImpl::BertLayerImpl(const BertConfigCpp& config) {
    attention_ = BertAttention(config);
    // Cross-attention logic would be initialized here if is_decoder is true
    // and config.add_cross_attention is true. Not implemented for standard BERT encoder.

    intermediate_ = BertIntermediate(config);
    output_ = BertOutput(config);

    register_module("attention", attention_);
    register_module("intermediate", intermediate_);
    register_module("output", output_);
}

// Returns: layer_output (Tensor), attention_probs (Tensor, possibly empty)
std::tuple<torch::Tensor, torch::Tensor> BertLayerImpl::forward(
    const torch::Tensor& hidden_states,
    const torch::Tensor& attention_mask_opt // Optional
) {
    // BertAttention's forward returns a tuple (context_layer, attention_probs)
    auto attention_outputs = attention_->forward(hidden_states, attention_mask_opt);
    torch::Tensor attention_output = std::get<0>(attention_outputs); // This is the output of BertSelfOutput
    torch::Tensor self_attention_probs = std::get<1>(attention_outputs); // Attention probabilities

    // Optional: Cross-attention would go here if this layer is part of a decoder.
    // For BERT encoder, self_attention_outputs is what we need.
    // The `attention_output` is already after the first residual connection and LayerNorm (within BertSelfOutput).

    // Feed Forward Network (FFN)
    torch::Tensor intermediate_output = intermediate_->forward(attention_output);

    // Second residual connection and LayerNorm (within BertOutput)
    // BertOutput takes the FFN output and the output of the attention block (before FFN)
    torch::Tensor layer_output = output_->forward(intermediate_output, attention_output);

    // If output_attentions is true, return self_attention_probs.
    // For now, always returning them. Can be made conditional.
    return std::make_tuple(layer_output, self_attention_probs);
}
