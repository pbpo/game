#pragma once

#include <torch/torch.h>
#include "can_bert_for_masked_lm.h" // For BertConfigCpp

// Corresponds to BertLMPredictionHead in Hugging Face.
// This includes the "transform" (dense layer, activation, LayerNorm) and the decoder layer.

// BertLMPredictionHeadTransform: Dense + Activation + LayerNorm
class BertPredictionHeadTransformImpl : public torch::nn::Module {
public:
    BertPredictionHeadTransformImpl(const BertConfigCpp& config);
    torch::Tensor forward(const torch::Tensor& hidden_states);

private:
    torch::nn::Linear dense_{nullptr};
    // Activation (GELU) - will use functional torch::gelu
    torch::nn::LayerNorm LayerNorm_{nullptr};
};
TORCH_MODULE(BertPredictionHeadTransform);


// BertLMPredictionHead: Transform + Decoder (Linear layer to vocab size)
class BertLMPredictionHeadImpl : public torch::nn::Module {
public:
    BertLMPredictionHeadImpl(const BertConfigCpp& config, const torch::Tensor& word_embedding_weights);

    torch::Tensor forward(const torch::Tensor& hidden_states);

private:
    BertPredictionHeadTransform transform_{nullptr};
    torch::nn::Linear decoder_{nullptr}; // This layer's weights are often tied to word_embeddings
                                       // The bias for the decoder is a separate parameter.
    // torch::Tensor bias_; // Decoder bias parameter
};
TORCH_MODULE(BertLMPredictionHead);
