#include "bert_model.h"
#include <stdexcept> // For std::runtime_error

BertModelImpl::BertModelImpl(const BertConfigCpp& config, bool add_pooling_layer)
    : config_(config), add_pooling_layer_(add_pooling_layer) {
    embeddings_ = BertEmbeddings(config_);
    encoder_ = BertEncoder(config_);

    if (add_pooling_layer_) {
        pooler_ = BertPooler(config_);
        register_module("pooler", pooler_);
    }

    register_module("embeddings", embeddings_);
    register_module("encoder", encoder_);
}

// Helper to prepare attention mask
torch::Tensor make_extended_attention_mask(const torch::Tensor& attention_mask, int64_t batch_size, int64_t seq_length, torch::Device device) {
    if (!attention_mask.defined() || attention_mask.numel() == 0) {
        // If no mask is provided, create a default mask of all ones.
        // This means no positions are masked.
        // For BERT, the mask should be 0 for unmasked, large negative for masked.
        // So, if no mask, it means all positions are attended.
        // The additive mask should be 0 then.
        return torch::zeros({batch_size, 1, 1, seq_length}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    }

    // Ensure attention_mask is on the correct device
    torch::Tensor extended_attention_mask = attention_mask.to(device);

    // We can broadcast a 1D mask to 3D.
    // If attention_mask is 1D or 2D ([batch_size, seq_length]), expand it
    if (extended_attention_mask.dim() == 2) {
        // Input: [batch_size, seq_length]
        // Output: [batch_size, 1, 1, seq_length] for BERT-style self-attention
        extended_attention_mask = extended_attention_mask.unsqueeze(1).unsqueeze(2);
    } else if (extended_attention_mask.dim() == 3) {
        // Input: [batch_size, from_seq_length, to_seq_length] (e.g. for cross-attention)
        // Output: [batch_size, 1, from_seq_length, to_seq_length]
        // For self-attention, from_seq_length is usually 1.
        extended_attention_mask = extended_attention_mask.unsqueeze(1);
    }
    // If it's already 4D, assume it's correctly shaped.

    // Since attention_mask is 0 or 1, convert to large negative number for masked positions.
    // (0 for unmasked, 1 for masked in some conventions, then invert; or 1 for unmasked, 0 for masked)
    // HuggingFace convention for input mask: 1 for positions to attend, 0 for masked.
    // The additive mask is then (1.0 - mask) * very_negative_number.
    extended_attention_mask = extended_attention_mask.to(torch::kFloat32); // Convert to float for multiplication
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0; // Large negative value for masked positions

    return extended_attention_mask;
}


BertModelImpl::BertModelOutput BertModelImpl::forward(
    const torch::Tensor& input_ids,
    const torch::Tensor& attention_mask_opt, // Optional
    const torch::Tensor& token_type_ids_opt, // Optional
    bool output_attentions,
    bool output_hidden_states
) {
    if (!input_ids.defined() || input_ids.numel() == 0) {
        throw std::runtime_error("input_ids tensor is undefined or empty in BertModelImpl::forward");
    }
    int64_t batch_size = input_ids.size(0);
    int64_t seq_length = input_ids.size(1);
    torch::Device device = input_ids.device();

    // 1. Prepare Attention Mask
    torch::Tensor extended_attention_mask = make_extended_attention_mask(attention_mask_opt, batch_size, seq_length, device);

    // 2. Embeddings
    torch::Tensor embedding_output = embeddings_->forward(input_ids, token_type_ids_opt);

    // 3. Encoder
    // BertEncoder's forward returns: (last_hidden_state, all_hidden_states_opt, all_attentions_opt)
    auto encoder_outputs = encoder_->forward(
        embedding_output,
        extended_attention_mask,
        output_attentions,
        output_hidden_states
    );

    torch::Tensor sequence_output = std::get<0>(encoder_outputs);
    torch::optional<std::vector<torch::Tensor>> all_hidden_states = std::get<1>(encoder_outputs);
    torch::optional<std::vector<torch::Tensor>> all_attentions = std::get<2>(encoder_outputs);

    // If output_hidden_states is true, BertEncoder already prepends the initial embedding_output
    // to all_hidden_states. (Verified in BertEncoderImpl::forward logic).

    // 4. Pooler (Optional)
    torch::Tensor pooled_output = torch::empty({0}, device); // Default empty tensor
    if (add_pooling_layer_ && pooler_) {
        pooled_output = pooler_->forward(sequence_output);
    }

    return std::make_tuple(sequence_output, pooled_output, all_hidden_states, all_attentions);
}

torch::Tensor BertModelImpl::get_input_embeddings_weight() const {
    if (embeddings_) {
        // BertEmbeddingsImpl needs a method to return word_embeddings_->weight
        return embeddings_->get_word_embedding_weights();
    }
    throw std::runtime_error("embeddings_ module is not initialized in BertModelImpl");
    return torch::Tensor();
}
