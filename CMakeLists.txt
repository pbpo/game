cmake_minimum_required(VERSION 3.16 FATAL_ERROR) # Increased version for modern CMake features like FetchContent if used extensively
project(CANBertCPP LANGUAGES CXX)

# --- Standard and Compiler ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Prefer not to use compiler-specific extensions

# --- LibTorch ---
# User needs to ensure LibTorch is installed and CMAKE_PREFIX_PATH is set correctly.
# Example: export CMAKE_PREFIX_PATH=/path/to/libtorch
# Or pass it during CMake configuration: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)

if(NOT Torch_FOUND)
    message(FATAL_ERROR "LibTorch not found. Please set CMAKE_PREFIX_PATH to your LibTorch installation directory.")
endif()
message(STATUS "Found LibTorch version: ${TORCH_VERSION}")
message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}") # Debug: show found libraries

# --- nlohmann/json (for Tokenizer vocab save/load) ---
# Option 1: Using FetchContent (CMake downloads it)
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3 # Specify a stable version
)
FetchContent_MakeAvailable(nlohmann_json)
message(STATUS "nlohmann/json fetched via FetchContent.")

# Option 2: Find installed nlohmann/json (if user installed it separately and it provides CMake config)
# find_package(nlohmann_json 3.2.0 QUIET) # Check for a specific version
# if (NOT nlohmann_json_FOUND)
#     message(STATUS "nlohmann_json package not found via find_package. Using FetchContent as fallback or direct include.")
#     # Fallback to FetchContent or require user to provide path
# else()
#    message(STATUS "Found nlohmann_json version ${nlohmann_json_VERSION}")
# endif()

# Option 3: User provides path to nlohmann/json include directory (simplest if header-only)
# set(NLOHMANN_JSON_INCLUDE_DIR "/path/to/nlohmann_json/single_include" CACHE PATH "Path to nlohmann/json include directory")
# if (NOT EXISTS "${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
#    message(FATAL_ERROR "nlohmann/json.hpp not found at specified NLOHMANN_JSON_INCLUDE_DIR: ${NLOHMANN_JSON_INCLUDE_DIR}")
# endif()
# include_directories(${NLOHMANN_JSON_INCLUDE_DIR})
# message(STATUS "Using nlohmann/json from user-provided path: ${NLOHMANN_JSON_INCLUDE_DIR}")


# --- Source Files ---
# Ensure all .cpp files are listed here
set(SOURCES
    main.cpp
    can_tokenizer.cpp
    mlm_dataset.cpp
    can_bert_for_masked_lm.cpp
    bert_embeddings.cpp
    bert_attention.cpp
    bert_intermediate.cpp
    bert_output.cpp
    bert_layer.cpp
    bert_encoder.cpp
    bert_pooler.cpp
    bert_model.cpp
    bert_lm_prediction_head.cpp
)

# --- Include Directories ---
# Add current source directory to include paths for our own headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# If nlohmann_json is fetched via FetchContent, its include directory is usually handled by its target.
# If using a local nlohmann/json.hpp, ensure its directory is in include_directories.

# --- Executable ---
add_executable(cpp_train ${SOURCES})

# --- Link Libraries ---
target_link_libraries(cpp_train PRIVATE Torch::torch) # Modern way to link LibTorch (if LibTorch provides Torch::torch target)
                                                      # Older LibTorch might just use ${TORCH_LIBRARIES}
                                                      # Using Torch::torch is generally preferred.

# Link nlohmann_json
# If using FetchContent and it creates a target like nlohmann_json::nlohmann_json:
target_link_libraries(cpp_train PRIVATE nlohmann_json::nlohmann_json)
# If nlohmann_json is header-only and just included, no separate linking needed for it.

# --- Platform Specifics & Compiler Flags (Optional) ---
if (MSVC)
    # Add MSVC specific flags if needed
    # target_compile_options(cpp_train PRIVATE /MP) # Example: multi-processor compilation
else()
    # Add GCC/Clang specific flags if needed
    # target_compile_options(cpp_train PRIVATE -Wall -Wextra -pedantic)
endif()

# For C++17 filesystem library:
# Modern compilers (GCC 9+, Clang 7+) usually link it automatically with -std=c++17.
# For older GCC (e.g., 7 or 8), you might need to explicitly link stdc++fs:
# if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
#    target_link_libraries(cpp_train PRIVATE stdc++fs)
# endif()
# However, CMAKE_CXX_STANDARD 17 should ideally handle this.

message(STATUS "CMake configuration finished. Executable target: cpp_train")
message(STATUS "To build, run: cd build && cmake .. && make") # Or your preferred generator
message(STATUS "Ensure CMAKE_PREFIX_PATH is set to your LibTorch directory if not found automatically.")
