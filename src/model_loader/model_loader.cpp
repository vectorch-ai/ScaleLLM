#include "model_loader.h"

#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "model_loader/state_dict.h"
#include "models/model_args.h"

namespace llm {

StateDictIterator::StateDictIterator(
    const std::vector<std::string>& model_weights_files,
    size_t index,
    bool is_pickle,
    bool is_sharded)
    : model_weights_files_(model_weights_files),
      index_(index),
      is_pickle_(is_pickle),
      is_sharded_(is_sharded) {}

const StateDict* StateDictIterator::get_state_dict() const {
  const size_t num_weight_files = model_weights_files_.size();
  CHECK(index_ < num_weight_files);
  // lazy loading
  if (!state_dict_) {
    const int shard_id = is_sharded_ ? static_cast<int>(index_) : 0;
    const int num_shards = is_sharded_ ? static_cast<int>(num_weight_files) : 1;
    if (is_pickle_) {
      state_dict_ = StateDict::load_pickle_file(
          model_weights_files_[index_], shard_id, num_shards);
    } else {
      state_dict_ = StateDict::load_safetensors(
          model_weights_files_[index_], shard_id, num_shards);
    }

    LOG(INFO) << "Loaded model weights from " << model_weights_files_[index_];
  }
  return state_dict_.get();
}

std::unique_ptr<ModelLoader> ModelLoader::create(
    const std::string& model_weights_path) {
  bool has_pytorch_weight_files = false;
  for (const auto& entry :
       std::filesystem::directory_iterator(model_weights_path)) {
    if (entry.path().extension() == ".pth") {
      has_pytorch_weight_files = true;
      break;
    }
  }
  if (has_pytorch_weight_files) {
    return std::make_unique<PTModelLoader>(model_weights_path);
  }
  return std::make_unique<HFModelLoader>(model_weights_path);
}

PTModelLoader::PTModelLoader(const std::string& model_weights_path) {
  const std::string args_file_path = model_weights_path + "/params.json";
  CHECK(load_model_args(args_file_path))
      << "Failed to load model args from " << args_file_path;
  for (const auto& entry :
       std::filesystem::directory_iterator(model_weights_path)) {
    if (entry.path().extension() == ".pth") {
      model_weights_files_.push_back(entry.path().string());
    }
  }
  CHECK(!model_weights_files_.empty())
      << "Failed to find model weights files in " << model_weights_path;
  // sort the model weights files by name
  std::sort(model_weights_files_.begin(), model_weights_files_.end());
}

bool PTModelLoader::load_model_args(const std::string& args_file_path) {
  using json = nlohmann::json;
  std::ifstream ifs(args_file_path);
  if (!ifs.is_open()) {
    LOG(ERROR) << "failed to open model args file: " << args_file_path;
    return false;
  }

  json data = json::parse(ifs);
  if (data.contains("dim")) {
    args_.dim() = data["dim"].get<int64_t>();
  }
  if (data.contains("n_layers")) {
    args_.n_layers() = data["n_layers"].get<int64_t>();
  }
  if (data.contains("n_heads")) {
    args_.n_heads() = data["n_heads"].get<int64_t>();
  }
  if (data.contains("n_kv_heads")) {
    args_.n_kv_heads() = data["n_kv_heads"].get<int64_t>();
  }
  if (data.contains("vocab_size")) {
    args_.vocab_size() = data["vocab_size"].get<int64_t>();
  }
  if (data.contains("multiple_of")) {
    args_.multiple_of() = data["multiple_of"].get<int64_t>();
  }
  if (data.contains("ffn_dim_multiplier")) {
    args_.ffn_dim_multiplier() = data["ffn_dim_multiplier"].get<float>();
  }
  if (data.contains("norm_eps")) {
    args_.norm_eps() = data["norm_eps"].get<float>();
  }
  if (data.contains("rope_theta")) {
    args_.rope_theta() = data["rope_theta"].get<float>();
  }
  if (data.contains("rope_scaling") && data["rope_scaling"].is_number_float()) {
    args_.rope_scaling() = data["rope_scaling"].get<float>();
  }
  // TODO: read from gflags
  args_.architectures().emplace_back("llama2");

  if (data.contains("hidden_dim")) {
    args_.hidden_dim() = data["hidden_dim"].get<int64_t>();
  } else {
    // calculate hidden_dim from dim
    const int64_t dim = args_.dim();
    const int64_t multiple_of = args_.multiple_of();
    const float ffn_dim_multiplier = args_.ffn_dim_multiplier().value_or(1.0f);
    int64_t hidden_dim = 4 * dim;
    hidden_dim = 2 * hidden_dim / 3;
    // custom dim factor multiplier
    hidden_dim *= ffn_dim_multiplier;
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);
    args_.hidden_dim() = hidden_dim;
  }

  // TODO: add more args
  return true;
}

HFModelLoader::HFModelLoader(const std::string& model_weights_path) {
  const std::string args_file_path = model_weights_path + "/config.json";
  CHECK(load_model_args(args_file_path))
      << "Failed to load model args from " << args_file_path;
  // try to load safetensors first
  for (const auto& entry :
       std::filesystem::directory_iterator(model_weights_path)) {
    // load bin or safe tensors
    if (entry.path().extension() == ".safetensors") {
      model_weights_files_.push_back(entry.path().string());
    }
  }
  // then load pickle files
  if (model_weights_files_.empty()) {
    // load pickle files
    for (const auto& entry :
         std::filesystem::directory_iterator(model_weights_path)) {
      if (entry.path().extension() == ".bin") {
        model_weights_files_.push_back(entry.path().string());
      }
    }
    is_pickle_ = true;
  }
  CHECK(!model_weights_files_.empty())
      << "Failed to find model weights files in " << model_weights_path;
  // sort the model weights files by name
  std::sort(model_weights_files_.begin(), model_weights_files_.end());
}

bool HFModelLoader::load_model_args(const std::string& args_file_path) {
  using json = nlohmann::json;
  std::ifstream ifs(args_file_path);
  if (!ifs.is_open()) {
    LOG(ERROR) << "failed to open model args file: " << args_file_path;
    return false;
  }

  json data = json::parse(ifs);
  if (data.contains("hidden_size")) {
    args_.dim() = data["hidden_size"].get<int64_t>();
  }
  if (data.contains("num_hidden_layers")) {
    args_.n_layers() = data["num_hidden_layers"].get<int64_t>();
  }
  if (data.contains("num_attention_heads")) {
    args_.n_heads() = data["num_attention_heads"].get<int64_t>();
  }
  if (data.contains("num_key_value_heads")) {
    args_.n_kv_heads() = data["num_key_value_heads"].get<int64_t>();
  }
  if (data.contains("vocab_size")) {
    args_.vocab_size() = data["vocab_size"].get<int64_t>();
  }
  if (data.contains("multiple_of")) {
    args_.multiple_of() = data["multiple_of"].get<int64_t>();
  }
  if (data.contains("ffn_dim_multiplier")) {
    args_.ffn_dim_multiplier() = data["ffn_dim_multiplier"].get<float>();
  }
  if (data.contains("rms_norm_eps")) {
    args_.norm_eps() = data["rms_norm_eps"].get<float>();
  }
  if (data.contains("architectures")) {
    CHECK(data["architectures"].is_array());
    for (const auto& str : data["architectures"]) {
      args_.architectures().push_back(str.get<std::string>());
    }
  }
  if (data.contains("rope_theta")) {
    args_.rope_theta() = data["rope_theta"].get<float>();
  }
  if (data.contains("rope_scaling") && data["rope_scaling"].is_number_float()) {
    args_.rope_scaling() = data["rope_scaling"].get<float>();
  }
  if (data.contains("intermediate_size")) {
    args_.hidden_dim() = data["intermediate_size"].get<int64_t>();
  } else {
    // calculate hidden_dim from dim
    const int64_t dim = args_.dim();
    const int64_t multiple_of = args_.multiple_of();
    const float ffn_dim_multiplier = args_.ffn_dim_multiplier().value_or(1.0f);
    int64_t hidden_dim = 4 * dim;
    hidden_dim = 2 * hidden_dim / 3;
    // custom dim factor multiplier
    hidden_dim *= ffn_dim_multiplier;
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);
    args_.hidden_dim() = hidden_dim;
  }

  // TODO: add more args
  // rope_scaling
  return true;
}

}  // namespace llm
