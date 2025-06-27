#pragma once

#include <torch/torch.h>
#include "can_bert_for_masked_lm.h" // For BertConfigCpp definition

// BertEmbeddings module similar to Hugging Face's BertEmbeddings
class BertEmbeddingsImpl : public torch::nn::Module {
public:
    BertEmbeddingsImpl(const BertConfigCpp& config);

    torch::Tensor forward(
        const torch::Tensor& input_ids,                 // LongTensor of shape [batch_size, sequence_length]
        const torch::Tensor& token_type_ids = torch::Tensor() // Optional LongTensor of shape [batch_size, sequence_length]
        // position_ids and past_key_values_length are not implemented in this version for simplicity,
        // similar to some minimal BERT implementations if not doing generation.
    );

    // Getter for word embedding weights
    torch::Tensor get_word_embedding_weights() const;

private:
    torch::nn::Embedding word_embeddings_{nullptr};
    torch::nn::Embedding position_embeddings_{nullptr};
    torch::nn::Embedding token_type_embeddings_{nullptr};

    torch::nn::LayerNorm LayerNorm_{nullptr};
    torch::nn::Dropout dropout_{nullptr};

    int64_t max_position_embeddings_; // To generate position_ids if not provided
    // BertConfigCpp config_; // Store if needed for other parameters not directly used by layers
};

TORCH_MODULE(BertEmbeddings);
