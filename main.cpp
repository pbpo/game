#include <iostream>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <filesystem> // For creating output directory
#include <map>          // For argument parsing

#include "can_tokenizer.h"
#include "mlm_dataset.h"  // MLMDataset now returns MLMExample = Example<std::vector<torch::Tensor>, torch::Tensor>
#include "can_bert_for_masked_lm.h" // Includes BertConfigCpp


// Basic argument parser
std::map<std::string, std::string> parse_args(int argc, char* argv[]) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg_str = argv[i];
        if (arg_str.rfind("--", 0) == 0) {
            std::string key = arg_str.substr(2);
            std::string value = "true";
            if (i + 1 < argc && argv[i+1][0] != '-') {
                value = argv[i+1];
                i++;
            }
            args[key] = value;
        }
    }
    return args;
}

template<typename T>
T get_arg(const std::map<std::string, std::string>& args, const std::string& key, const T& default_val) {
    auto it = args.find(key);
    if (it != args.end()) {
        std::istringstream iss(it->second);
        T val;
        iss >> val;
        if (iss.fail()) {
             std::cerr << "Warning: Failed to parse argument '" << key << "' with value '" << it->second << "'. Using default: " << default_val << std::endl;
             return default_val;
        }
        return val;
    }
    return default_val;
}
template<>
std::string get_arg<std::string>(const std::map<std::string, std::string>& args, const std::string& key, const std::string& default_val) {
    auto it = args.find(key);
    if (it != args.end()) {
        return it->second;
    }
    return default_val;
}


void train_one_epoch(
    CANBertForMaskedLM model, // Model is passed by value (TORCH_MODULE behavior)
    torch::data::DataLoader<MLMDataset, torch::data::samplers::RandomSampler>& data_loader, // Pass by reference
    torch::optim::Optimizer& optimizer,
    torch::Device device,
    int epoch,
    int total_epochs,
    int report_interval = 100
) {
    model->train();
    size_t batch_idx = 0;
    double running_loss = 0.0;
    size_t total_batches = 0;

    // Calculate total batches for progress reporting
    if (data_loader.dataset().size().has_value() && data_loader.options().batch_size() > 0) {
         total_batches = *data_loader.dataset().size() / data_loader.options().batch_size();
        if (*data_loader.dataset().size() % data_loader.options().batch_size() != 0) {
            total_batches++;
        }
    } else {
        // This case should ideally not happen if dataset size is known and batch_size > 0
        std::cerr << "Warning: Could not determine total batches accurately. Progress reporting might be affected." << std::endl;
    }

    std::cout << "--- Starting Epoch " << epoch + 1 << "/" << total_epochs << " ---" << std::endl;

    for (MLMExample& batch : data_loader) {
        // batch.data is std::vector<torch::Tensor> {input_ids, attention_mask}
        // batch.target is Tensor (labels)
        if (batch.data.size() < 2) {
            std::cerr << "Error: Batch data does not contain enough elements (expected input_ids and attention_mask at batch index " << batch_idx << ")." << std::endl;
            continue; // Skip this malformed batch
        }
        auto input_ids_tensor = batch.data[0].to(device);
        auto attention_mask_tensor = batch.data[1].to(device);

        // Create token_type_ids, usually zeros for BERT MLM if not using sentence pairs.
        // Ensure it's on the same device as input_ids.
        auto token_type_ids_tensor = torch::zeros_like(input_ids_tensor).to(device);
        auto labels_tensor = batch.target.to(device);

        optimizer.zero_grad();

        auto outputs = model->forward(
            input_ids_tensor,
            attention_mask_tensor,
            token_type_ids_tensor,
            labels_tensor
            // output_attentions and output_hidden_states default to false in CANBertForMaskedLMImpl::forward
        );
        torch::Tensor loss = std::get<0>(outputs);

        if (loss.defined() && loss.numel() > 0 && !torch::isnan(loss).all().item<bool>() && !torch::isinf(loss).all().item<bool>()) {
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0); // Gradient clipping
            optimizer.step();
            running_loss += loss.item<float>();
        } else {
             std::cerr << "Warning: Invalid loss encountered at batch " << batch_idx
                       << " (value: " << (loss.defined() && loss.numel() > 0 ? loss.item<float>() : -999.0f)
                       << ", is_nan: " << (loss.defined() ? torch::isnan(loss).all().item<bool>() : true)
                       << ", is_inf: " << (loss.defined() ? torch::isinf(loss).all().item<bool>() : true)
                       << "). Skipping backward pass for this batch." << std::endl;
        }

        if (report_interval > 0 && (batch_idx % report_interval == 0 || (total_batches > 0 && batch_idx == total_batches -1) )) {
            float current_loss_val = (loss.defined() && loss.numel() > 0 && !torch::isnan(loss).all().item<bool>() && !torch::isinf(loss).all().item<bool>()) ? loss.item<float>() : -1.0f;
            double avg_loss_interval = current_loss_val;

            if (batch_idx % report_interval == 0 && batch_idx > 0) {
                 avg_loss_interval = running_loss / report_interval;
            } else if (batch_idx == 0 && report_interval > 1) { // First batch, if interval > 1
                avg_loss_interval = current_loss_val;
            } else if (total_batches > 0 && batch_idx == total_batches -1 && (batch_idx % report_interval != 0) ) { // Last batch, not on interval
                 avg_loss_interval = running_loss / (batch_idx % report_interval +1); // Avg over current partial interval
            }


            std::string total_batches_str = (total_batches > 0) ? std::to_string(total_batches) : "?";
            std::cout << "Epoch: " << epoch + 1 << " | Batch: " << batch_idx +1 << "/" << total_batches_str
                      << " | Avg Loss (last " << ( (batch_idx % report_interval == 0 && batch_idx > 0) ? report_interval : (batch_idx % report_interval +1) ) << " batches): " << avg_loss_interval
                      << " | Current Loss: " << current_loss_val << std::endl;

            if (batch_idx % report_interval == 0 && batch_idx > 0) {
                running_loss = 0.0; // Reset for next interval
            }
        }
        batch_idx++;
    }
    std::cout << "--- Epoch " << epoch + 1 << " finished ---" << std::endl;
}


int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);

    std::string data_path = get_arg<std::string>(args, "data_path", "");
    std::string vocab_path = get_arg<std::string>(args, "vocab_path", "");
    std::string output_dir_str = get_arg<std::string>(args, "output_dir", "checkpoints_cpp/");
    std::string dataset_type = get_arg<std::string>(args, "dataset_type", "candump");
    int64_t seq_len = get_arg<int64_t>(args, "seq_len", 126);
    int64_t batch_size = get_arg<int64_t>(args, "batch_size", 32);
    int64_t epochs = get_arg<int64_t>(args, "epochs", 5);
    double learning_rate = get_arg<double>(args, "learning_rate", 5e-5);
    int num_workers = get_arg<int>(args, "num_workers", 0); // DataLoader workers. Often 0 for C++ unless GIL is not an issue.

    if (data_path.empty() || vocab_path.empty()) {
        std::cerr << "Error: --data_path and --vocab_path are required." << std::endl;
        std::cerr << "Usage: ./cpp_train --data_path /path/to/data.log --vocab_path /path/to/vocab.json [...]" << std::endl;
        return 1;
    }

    std::filesystem::path output_dir(output_dir_str);
    try {
        if (!std::filesystem::exists(output_dir)) {
            if (std::filesystem::create_directories(output_dir)) {
                std::cout << "Created output directory: " << output_dir.string() << std::endl;
            } else {
                std::cerr << "Error: Could not create output directory: " << output_dir.string() << std::endl;
                return 1; // Exit if directory creation fails
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error when creating output directory: " << e.what() << std::endl;
        return 1;
    }

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    CANTokenizer tokenizer; // Tokenizer is on stack, its lifetime must cover its use in MLMDataset
    if (!tokenizer.load_vocab(vocab_path)) {
        std::cerr << "Error: Could not load vocabulary from " << vocab_path << std::endl;
        return 1;
    }
    std::cout << "Vocabulary loaded. Vocab size: " << tokenizer.get_vocab_size() << std::endl;

    BertConfigCpp model_config(
        tokenizer.get_vocab_size(),
        256,                        // hidden_size from Python script
        4,                          // num_hidden_layers from Python script
        1,                          // num_attention_heads from Python script
        512,                        // intermediate_size from Python script
        0.1,                        // hidden_dropout_prob
        0.1,                        // attention_probs_dropout_prob
        seq_len                     // max_position_embeddings
    );
    std::cout << "BertConfigCpp created." << std::endl;
    std::cout << "  Vocab Size: " << model_config.vocab_size << std::endl;
    std::cout << "  Hidden Size: " << model_config.hidden_size << std::endl;
    std::cout << "  Num Hidden Layers: " << model_config.num_hidden_layers << std::endl;
    std::cout << "  Num Attention Heads: " << model_config.num_attention_heads << std::endl;
    std::cout << "  Max Position Embeddings: " << model_config.max_position_embeddings << std::endl;

    CANBertForMaskedLM model(model_config);
    model->to(device);
    std::cout << "CANBertForMaskedLM model created and moved to device." << std::endl;

    long total_params = 0;
    for (const auto& p : model->parameters()) {
        total_params += p.numel();
    }
    std::cout << "Total model parameters: " << total_params << std::endl;

    // Create MLMDataset instance. Tokenizer is passed by const reference.
    MLMDataset train_dataset(data_path, tokenizer, seq_len, dataset_type);

    if (!train_dataset.size().has_value() || train_dataset.size().value() == 0) {
         std::cerr << "Error: MLM dataset is empty. Check data file or tokenizer." << std::endl;
         return 1;
    }
    std::cout << "MLMDataset created. Total trainable sequences: " << train_dataset.size().value() << std::endl;

    auto data_loader_options = torch::data::DataLoaderOptions()
                                   .batch_size(batch_size)
                                   .workers(num_workers) // Set number of workers
                                   .enforce_ordering(false); // For RandomSampler, ordering is not enforced.

    // Create DataLoader with RandomSampler (shuffles data each epoch)
    // MLMDataset (train_dataset) is moved into the DataLoader.
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        data_loader_options);

    std::cout << "DataLoader created with RandomSampler." << std::endl;

    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(learning_rate));
    std::cout << "Optimizer AdamW created." << std::endl;

    std::cout << "Starting training..." << std::endl;
    for (int64_t epoch = 0; epoch < epochs; ++epoch) {
        train_one_epoch(model, *data_loader, optimizer, device, epoch, epochs);

        // Save checkpoint
        std::string checkpoint_filename = "can-bert-cpp-epoch-" + std::to_string(epoch + 1) + ".pt";
        std::filesystem::path checkpoint_path = output_dir / checkpoint_filename;
        try {
            torch::save(model, checkpoint_path.string());
            std::cout << "Epoch " << epoch + 1 << " | Checkpoint saved to " << checkpoint_path.string() << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error saving checkpoint (LibTorch error): " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error saving checkpoint (Std error): " << e.what() << std::endl;
        }
    }

    std::cout << "Training finished." << std::endl;

    return 0;
}

// To compile (example with g++ and CMake for LibTorch):
// CMakeLists.txt needs to include all .cpp files:
// add_executable(cpp_train
//     main.cpp
//     can_tokenizer.cpp
//     mlm_dataset.cpp
//     can_bert_for_masked_lm.cpp
//     bert_embeddings.cpp
//     bert_attention.cpp
//     bert_intermediate.cpp
//     bert_output.cpp
//     bert_layer.cpp
//     bert_encoder.cpp
//     bert_pooler.cpp
//     bert_model.cpp
//     bert_lm_prediction_head.cpp
// )
// target_link_libraries(cpp_train PRIVATE ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json) // If using nlohmann for json in tokenizer
// Ensure CMAKE_CXX_STANDARD 17 or higher for std::filesystem
// For nlohmann/json, you might need to add it via FetchContent or include its path.
// Example for nlohmann/json with FetchContent in CMake:
// include(FetchContent)
// FetchContent_Declare(
//   nlohmann_json
//   GIT_REPOSITORY https://github.com/nlohmann/json.git
//   GIT_TAG v3.10.5 # Or a specific version
// )
// FetchContent_MakeAvailable(nlohmann_json)
// And then link nlohmann_json::nlohmann_json
// For <filesystem>, ensure your compiler supports C++17 and link against stdc++fs if needed on older GCC.
// For GCC 9+ or Clang 7+, usually -std=c++17 is enough.
// For GCC < 9, you might need to explicitly link: target_link_libraries(cpp_train PRIVATE stdc++fs)
```
