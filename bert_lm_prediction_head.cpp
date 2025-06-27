#include "bert_lm_prediction_head.h"
#include <torch/nn/functional.h> // For torch::gelu

// --- BertPredictionHeadTransformImpl ---
BertPredictionHeadTransformImpl::BertPredictionHeadTransformImpl(const BertConfigCpp& config) {
    dense_ = torch::nn::Linear(config.hidden_size, config.hidden_size);
    LayerNorm_ = torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({config.hidden_size}).eps(config.layer_norm_eps)
    );

    register_module("dense", dense_);
    register_module("LayerNorm", LayerNorm_);
    // GELU activation is applied functionally in forward.
}

torch::Tensor BertPredictionHeadTransformImpl::forward(const torch::Tensor& hidden_states) {
    torch::Tensor transformed_states = dense_->forward(hidden_states);
    transformed_states = torch::gelu(transformed_states); // Apply GELU activation
    transformed_states = LayerNorm_->forward(transformed_states);
    return transformed_states;
}


// --- BertLMPredictionHeadImpl ---
BertLMPredictionHeadImpl::BertLMPredictionHeadImpl(const BertConfigCpp& config, const torch::Tensor& word_embedding_weights) {
    transform_ = BertPredictionHeadTransform(config);

    // The decoder layer projects from hidden_size to vocab_size.
    // Its weights are often tied to the word embedding matrix.
    // The bias is a separate parameter.
    decoder_ = torch::nn::Linear(config.hidden_size, config.vocab_size);

    // Tie weights if word_embedding_weights are provided and match dimensions
    if (word_embedding_weights.defined() &&
        word_embedding_weights.size(0) == config.vocab_size &&
        word_embedding_weights.size(1) == config.hidden_size) {
        decoder_->weight = word_embedding_weights; // Share the weight tensor
    } else {
        // Not tying weights or dimensions mismatch. Decoder will have its own initialized weights.
        // This might be an error condition depending on the desired setup.
        // std::cout << "Warning: BertLMPredictionHead decoder weights not tied or dimensions mismatch." << std::endl;
        // Or initialize decoder_->weight with word_embedding_weights if that's the intention for pretraining
        // and the config is such that they should match.
        // If config.tie_word_embeddings is a parameter, it should be checked.
        // For now, if not tied, it uses its own default initialization.
    }
    // The bias of the decoder is typically learned independently.
    // It's already part of torch::nn::Linear. If you wanted a specific bias parameter:
    // bias_ = register_parameter("bias", torch::zeros({config.vocab_size}));
    // And then in forward: return torch::linear(transformed_states, decoder_->weight, bias_);

    register_module("transform", transform_);
    register_module("decoder", decoder_);
}

torch::Tensor BertLMPredictionHeadImpl::forward(const torch::Tensor& hidden_states) {
    torch::Tensor transformed_states = transform_->forward(hidden_states);
    torch::Tensor prediction_scores = decoder_->forward(transformed_states);
    // If using custom bias:
    // torch::Tensor prediction_scores = torch::linear(transformed_states, decoder_->weight, bias_);
    return prediction_scores;
}
