#include "bert_encoder.h"

BertEncoderImpl::BertEncoderImpl(const BertConfigCpp& config) {
    layer_ = torch::nn::ModuleList();
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layer_->push_back(BertLayer(config));
    }
    register_module("layer", layer_);
}

std::tuple<torch::Tensor, torch::optional<std::vector<torch::Tensor>>, torch::optional<std::vector<torch::Tensor>>>
BertEncoderImpl::forward(
    const torch::Tensor& initial_hidden_states,
    const torch::Tensor& attention_mask_opt, // Optional
    bool output_attentions, // Simplified control
    bool output_hidden_states // Simplified control
) {
    torch::Tensor current_hidden_states = initial_hidden_states;

    torch::optional<std::vector<torch::Tensor>> all_hidden_states_opt = torch::nullopt;
    if (output_hidden_states) {
        all_hidden_states_opt = std::vector<torch::Tensor>();
        all_hidden_states_opt->reserve(layer_->size() + 1);
        all_hidden_states_opt->push_back(current_hidden_states);
    }

    torch::optional<std::vector<torch::Tensor>> all_self_attentions_opt = torch::nullopt;
    if (output_attentions) {
        all_self_attentions_opt = std::vector<torch::Tensor>();
        all_self_attentions_opt->reserve(layer_->size());
    }

    // (Optional) head_mask processing would go here if implemented.

    for (int i = 0; i < layer_->size(); ++i) {
        auto layer_module = layer_[i]->as<BertLayerImpl>(); // Cast to specific type if needed, or use BertLayer directly
        if (!layer_module) { // Or if using BertLayer: if (!layer_[i])
             throw std::runtime_error("BertEncoder: Layer module is null at index " + std::to_string(i));
        }

        // BertLayer's forward returns: layer_output, attention_probs
        auto layer_outputs = layer_module->forward(current_hidden_states, attention_mask_opt);
        current_hidden_states = std::get<0>(layer_outputs);
        torch::Tensor attention_probs = std::get<1>(layer_outputs);

        if (output_hidden_states) {
            all_hidden_states_opt->push_back(current_hidden_states);
        }
        if (output_attentions) {
            all_self_attentions_opt->push_back(attention_probs);
        }
    }

    // (Optional) apply_chunking_to_forward logic would go here if implemented for memory saving.

    return std::make_tuple(current_hidden_states, all_hidden_states_opt, all_self_attentions_opt);
}
