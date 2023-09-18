#include "model_args.h"

#include <torch/torch.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>

namespace llm {
bool ModelArgs::load_from_file(const std::string& file_path) {
  using json = nlohmann::json;
  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    LOG(ERROR) << "failed to open model args file: " << file_path;
    return false;
  }

  json data = json::parse(ifs);
  if (data.contains("dim")) {
    dim_ = data["dim"].get<int64_t>();
  }
  if (data.contains("n_layers")) {
    n_layers_ = data["n_layers"].get<int64_t>();
  }
  if (data.contains("n_heads")) {
    n_heads_ = data["n_heads"].get<int64_t>();
  }
  if (data.contains("n_kv_heads")) {
    n_kv_heads_ = data["n_kv_heads"].get<int64_t>();
  }
  if (data.contains("vocab_size")) {
    vocab_size_ = data["vocab_size"].get<int64_t>();
  }
  if (data.contains("multiple_of")) {
    multiple_of_ = data["multiple_of"].get<int64_t>();
  }
  if (data.contains("ffn_dim_multiplier")) {
    ffn_dim_multiplier_ = data["ffn_dim_multiplier"].get<float>();
  }
  if (data.contains("norm_eps")) {
    norm_eps_ = data["norm_eps"].get<float>();
  }

  // TODO: add more args
  return true;
}

bool ModelArgs::load_from_file_for_hf(const std::string& file_path) {
  using json = nlohmann::json;
  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    LOG(ERROR) << "failed to open model args file: " << file_path;
    return false;
  }

  json data = json::parse(ifs);
  if (data.contains("hidden_size")) {
    dim_ = data["hidden_size"].get<int64_t>();
  }
  if (data.contains("num_hidden_layers")) {
    n_layers_ = data["num_hidden_layers"].get<int64_t>();
  }
  if (data.contains("num_attention_heads")) {
    n_heads_ = data["num_attention_heads"].get<int64_t>();
  }
  if (data.contains("num_key_value_heads")) {
    n_kv_heads_ = data["num_key_value_heads"].get<int64_t>();
  }
  if (data.contains("vocab_size")) {
    vocab_size_ = data["vocab_size"].get<int64_t>();
  }
  if (data.contains("multiple_of")) {
    multiple_of_ = data["multiple_of"].get<int64_t>();
  }
  if (data.contains("ffn_dim_multiplier")) {
    ffn_dim_multiplier_ = data["ffn_dim_multiplier"].get<float>();
  }
  if (data.contains("rms_norm_eps")) {
    norm_eps_ = data["rms_norm_eps"].get<float>();
  }

  // TODO: add more args
  // rope_scaling
  return true;
}

}  // namespace llm
