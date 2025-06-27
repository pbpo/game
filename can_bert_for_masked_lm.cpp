#include "can_bert_for_masked_lm.h"
#include <stdexcept> // For std::runtime_error

// --- CANBertForMaskedLMImpl Implementation ---
CANBertForMaskedLMImpl::CANBertForMaskedLMImpl(const BertConfigCpp& config) : config_(config) {
    bert_ = BertModel(config_);

    torch::Tensor word_embedding_weights;
    if (config_.tie_word_embeddings) {
        // Use the new getter methods to retrieve word embedding weights
        if (bert_) { // Ensure bert_ is initialized
            word_embedding_weights = bert_->get_input_embeddings_weight();
        } else {
            throw std::runtime_error("BertModel is not initialized in CANBertForMaskedLMImpl constructor before accessing weights.");
        }
    }

    cls_ = BertLMPredictionHead(config_, word_embedding_weights);

    register_module("bert", bert_);
    register_module("cls", cls_);

    loss_fct_ = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().ignore_index(-100));
}

// This specific implementation of get_word_embedding_weights is now mostly
// for conceptual alignment with how Python might do it if it were separate.
// The actual weight tying happens by passing the weights to BertLMPredictionHead's constructor.
// If direct access to the weights of the already constructed cls_ or bert_ is needed later,
// similar getter logic would apply to those instances.
torch::Tensor CANBertForMaskedLMImpl::get_word_embedding_weights() const {
    if (bert_) {
        return bert_->get_input_embeddings_weight();
    }
    throw std::runtime_error("CANBertForMaskedLMImpl: bert_ module is not initialized.");
    return torch::Tensor();
}


std::tuple<torch::Tensor, torch::Tensor> CANBertForMaskedLMImpl::forward(
    const torch::Tensor& input_ids,
    const torch::Tensor& attention_mask, // Optional
    const torch::Tensor& token_type_ids, // Optional
    const torch::Tensor& labels,         // Optional
    bool output_attentions,
    bool output_hidden_states
) {
    auto bert_outputs = bert_->forward(
        input_ids,
        attention_mask,
        token_type_ids,
        output_attentions,
        output_hidden_states
    );

    torch::Tensor sequence_output = std::get<0>(bert_outputs);
    torch::Tensor prediction_scores = cls_->forward(sequence_output);

    torch::Tensor masked_lm_loss = torch::empty({0}, input_ids.options().dtype(torch::kFloat32));

    if (labels.defined() && labels.numel() > 0) {
        torch::Tensor flat_prediction_scores = prediction_scores.reshape({-1, config_.vocab_size});
        torch::Tensor flat_labels = labels.reshape({-1});
        masked_lm_loss = loss_fct_(flat_prediction_scores, flat_labels);
    }

    return std::make_tuple(masked_lm_loss, prediction_scores);
}
