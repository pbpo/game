#include "bert_embeddings.h"
#include <stdexcept> // For std::runtime_error

BertEmbeddingsImpl::BertEmbeddingsImpl(const BertConfigCpp& config)
    : max_position_embeddings_(config.max_position_embeddings)
    // config_(config) // Store config if more params are needed later
{
    word_embeddings_ = torch::nn::Embedding(
        torch::nn::EmbeddingOptions(config.vocab_size, config.hidden_size)
        // padding_idx can be set if applicable, BERT usually doesn't rely on it here
        // as attention_mask handles padding.
    );
    position_embeddings_ = torch::nn::Embedding(config.max_position_embeddings, config.hidden_size);
    token_type_embeddings_ = torch::nn::Embedding(config.type_vocab_size, config.hidden_size);

    LayerNorm_ = torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({config.hidden_size}).eps(config.layer_norm_eps)
    );
    dropout_ = torch::nn::Dropout(config.hidden_dropout_prob);

    register_module("word_embeddings", word_embeddings_);
    register_module("position_embeddings", position_embeddings_);
    register_module("token_type_embeddings", token_type_embeddings_);
    register_module("LayerNorm", LayerNorm_);
    register_module("dropout", dropout_);

    // Initialize position_ids buffer if needed for some implementations,
    // but Hugging Face calculates it on the fly in forward if not provided.
    // We will do the same.
}

torch::Tensor BertEmbeddingsImpl::forward(
    const torch::Tensor& input_ids,         // LongTensor of shape [batch_size, sequence_length]
    const torch::Tensor& token_type_ids_opt // Optional LongTensor of shape [batch_size, sequence_length]
) {
    if (!input_ids.defined() || input_ids.numel() == 0) {
        throw std::runtime_error("input_ids tensor is undefined or empty in BertEmbeddingsImpl::forward");
    }
    int64_t batch_size = input_ids.size(0);
    int64_t seq_length = input_ids.size(1);

    // Position IDs
    // Create position_ids on the fly: tensor of shape [1, seq_length] from 0 to seq_length - 1
    // This is then broadcasted to [batch_size, seq_length]
    torch::Tensor position_ids = torch::arange(seq_length, torch::TensorOptions().dtype(torch::kLong).device(input_ids.device()));
    position_ids = position_ids.unsqueeze(0).expand({batch_size, seq_length});
    // Ensure position_ids are within max_position_embeddings bounds.
    // This simple arange might exceed if seq_length > max_position_embeddings if not handled by model config.
    // BertConfig typically has max_position_embeddings >= longest sequence.
    // If using learned position embeddings, this must be true.

    // Token Type IDs
    torch::Tensor token_type_ids;
    if (token_type_ids_opt.defined() && token_type_ids_opt.numel() > 0) {
        token_type_ids = token_type_ids_opt;
    } else {
        // Defaults to zeros if not provided
        token_type_ids = torch::zeros_like(input_ids, torch::TensorOptions().dtype(torch::kLong).device(input_ids.device()));
    }

    // Embeddings
    torch::Tensor inputs_embeds = word_embeddings_->forward(input_ids);
    torch::Tensor position_embeds = position_embeddings_->forward(position_ids);
    torch::Tensor token_type_embeds = token_type_embeddings_->forward(token_type_ids);

    torch::Tensor embeddings = inputs_embeds + position_embeds + token_type_embeds;

    embeddings = LayerNorm_->forward(embeddings);
    embeddings = dropout_->forward(embeddings);

    return embeddings;
}

torch::Tensor BertEmbeddingsImpl::get_word_embedding_weights() const {
    if (word_embeddings_) {
        return word_embeddings_->weight;
    }
    // Should not happen if constructor ran correctly
    throw std::runtime_error("word_embeddings_ is not initialized in BertEmbeddingsImpl");
    return torch::Tensor();
}
