#include "model_loader.h"

#include <gflags/gflags.h>
#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "model_loader/state_dict.h"
#include "models/args.h"
#include "tokenizer/hf_tokenizer.h"
#include "tokenizer/sentencepiece_tokenizer.h"

DEFINE_int64(max_position_embeddings, 0, "Maximum position embeddings.");
DEFINE_string(model_type, "", "model type, e.g. llama2, llama, gpt_neox");

DEFINE_string(tokenizer_path, "", "Path to the tokenizer file.");

namespace llm {
using json = nlohmann::json;

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

std::unique_ptr<Tokenizer> PTModelLoader::tokenizer() const {
  // use the tokenizer path specified by the user if exists
  const std::string tokenizer_path =
      !FLAGS_tokenizer_path.empty() ? FLAGS_tokenizer_path
                                    : model_weights_path_ + "/tokenizer.model";
  if (!std::filesystem::exists(tokenizer_path)) {
    LOG(ERROR) << "Failed to find tokenizer file: " << tokenizer_path;
    return nullptr;
  }
  return std::make_unique<SentencePieceTokenizer>(tokenizer_path);
}

PTModelLoader::PTModelLoader(const std::string& model_weights_path)
    : model_weights_path_(model_weights_path) {
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
  std::ifstream ifs(args_file_path);
  if (!ifs.is_open()) {
    LOG(ERROR) << "failed to open model args file: " << args_file_path;
    return false;
  }

  // hardcode bos and eos for llama2
  args_.bos_token_id() = 1;
  args_.eos_token_id() = 2;

  json data = json::parse(ifs);
  if (data.contains("dim")) {
    args_.hidden_size() = data["dim"].get<int64_t>();
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
  if (FLAGS_max_position_embeddings > 0) {
    args_.max_position_embeddings() = FLAGS_max_position_embeddings;
  }
  if (data.contains("norm_eps")) {
    args_.rms_norm_eps() = data["norm_eps"].get<float>();
  }
  if (data.contains("rope_theta")) {
    args_.rope_theta() = data["rope_theta"].get<float>();
  }
  if (data.contains("rope_scaling") && data["rope_scaling"].is_number_float()) {
    args_.rope_scaling() = data["rope_scaling"].get<float>();
  }

  if (!FLAGS_model_type.empty()) {
    args_.model_type() = FLAGS_model_type;
  } else if (data.contains("model_type")) {
    args_.model_type() = data["model_type"].get<std::string>();
  } else {
    // Hardcode model type to llama2 if not specified
    args_.model_type() = "llama2";
  }

  if (data.contains("intermediate_size")) {
    args_.intermediate_size() = data["intermediate_size"].get<int64_t>();
  } else {
    int64_t multiple_of = 256;
    float ffn_dim_multiplier = 1.0f;
    if (data.contains("multiple_of")) {
      multiple_of = data["multiple_of"].get<int64_t>();
    }
    if (data.contains("ffn_dim_multiplier")) {
      ffn_dim_multiplier = data["ffn_dim_multiplier"].get<float>();
    }

    // calculate hidden_dim from dim
    int64_t intermediate_size = args_.hidden_size() * 4;
    intermediate_size = 2 * intermediate_size / 3;
    // custom dim factor multiplier
    intermediate_size *= ffn_dim_multiplier;
    // round up to make hidden layer size multiple of large power of 2
    intermediate_size =
        multiple_of * ((intermediate_size + multiple_of - 1) / multiple_of);
    args_.intermediate_size() = intermediate_size;
  }

  // TODO: add more args
  return true;
}

HFModelLoader::HFModelLoader(const std::string& model_weights_path): model_weights_path_(model_weights_path) {
  CHECK(load_model_args(model_weights_path));
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

std::unique_ptr<Tokenizer> HFModelLoader::tokenizer() const {
  // use the tokenizer path specified by the user if exists
  const std::string tokenizer_path = model_weights_path_ + "/tokenizer.json";
  if (!std::filesystem::exists(tokenizer_path)) {
    LOG(ERROR) << "Failed to find tokenizer file: " << tokenizer_path;
    return nullptr;
  }
  return HFTokenizer::from_file(tokenizer_path);
}

bool HFModelLoader::load_model_args(const std::string& model_weights_path) {
  const std::string args_file_path = model_weights_path + "/config.json";
  std::ifstream ifs(args_file_path);
  if (!ifs.is_open()) {
    LOG(ERROR) << "failed to open model args file: " << args_file_path;
    return false;
  }

  json data = json::parse(ifs);
  if (data.contains("bos_token_id")) {
    args_.bos_token_id() = data["bos_token_id"].get<int32_t>();
  }
  if (data.contains("eos_token_id")) {
    args_.eos_token_id() = data["eos_token_id"].get<int32_t>();
  }
  if (data.contains("hidden_size")) {
    args_.hidden_size() = data["hidden_size"].get<int64_t>();
  }
  if (data.contains("hidden_act")) {
    args_.hidden_act() = data["hidden_act"].get<std::string>();
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

  if (FLAGS_max_position_embeddings > 0) {
    args_.max_position_embeddings() = FLAGS_max_position_embeddings;
  } else if (data.contains("max_position_embeddings")) {
    args_.max_position_embeddings() =
        data["max_position_embeddings"].get<int64_t>();
  }
  if (data.contains("rms_norm_eps")) {
    args_.rms_norm_eps() = data["rms_norm_eps"].get<float>();
  }
  if (data.contains("layer_norm_eps")) {
    args_.layer_norm_eps() = data["layer_norm_eps"].get<float>();
  }
  // TODO: load args based on model type
  if (data.contains("model_type")) {
    args_.model_type() = data["model_type"].get<std::string>();
  }
  if (data.contains("rotary_emb_base")) {
    args_.rope_theta() = data["rotary_emb_base"].get<float>();
  }
  if (data.contains("rope_scaling") && data["rope_scaling"].is_number_float()) {
    args_.rope_scaling() = data["rope_scaling"].get<float>();
  }
  if (data.contains("rotary_pct")) {
    args_.rotary_pct() = data["rotary_pct"].get<float>();
  }
  if (data.contains("intermediate_size")) {
    args_.intermediate_size() = data["intermediate_size"].get<int64_t>();
  } else {
    LOG(ERROR) << "Failed to find intermediate_size in config.json";
    return false;
  }
  if (data.contains("use_parallel_residual")) {
    args_.use_parallel_residual() =
        data["use_parallel_residual"].get<bool>();
  }

  // load quantization args if exists
  if (data.contains("quantization_config")) {
    const auto quantization_config = data["quantization_config"];
    if (quantization_config.contains("quant_method")) {
      quant_args_.quant_method() =
          quantization_config["quant_method"].get<std::string>();
    }
    if (quantization_config.contains("bits")) {
      quant_args_.bits() = quantization_config["bits"].get<int64_t>();
    }
    if (quantization_config.contains("group_size")) {
      quant_args_.group_size() =
          quantization_config["group_size"].get<int64_t>();
    }
    if (quantization_config.contains("desc_act")) {
      quant_args_.desc_act() = quantization_config["desc_act"].get<bool>();
    }
    if (quantization_config.contains("true_sequential")) {
      quant_args_.true_sequential() =
          quantization_config["true_sequential"].get<bool>();
    }
  }

  // load quantization args for awq if exists
  const std::string quant_args_file_path =
      model_weights_path + "/quant_config.json";
  if (std::filesystem::exists(quant_args_file_path)) {
    std::ifstream ifs(quant_args_file_path);
    if (!ifs.is_open()) {
      LOG(ERROR) << "failed to open model args file: " << quant_args_file_path;
      return false;
    }

    json data = json::parse(ifs);
    if (data.contains("version")) {
      quant_args_.quant_method() = data["version"].get<std::string>();
    }
    if (data.contains("w_bit")) {
      quant_args_.bits() = data["w_bit"].get<int64_t>();
    }
    if (data.contains("q_group_size")) {
      quant_args_.group_size() = data["q_group_size"].get<int64_t>();
    }
  }

  // load quantization args for gptq if exists
  const std::string gptq_args_file_path =
      model_weights_path + "/quantize_config.json";
  if (std::filesystem::exists(gptq_args_file_path)) {
    std::ifstream ifs(gptq_args_file_path);
    if (!ifs.is_open()) {
      LOG(ERROR) << "failed to open model args file: " << gptq_args_file_path;
      return false;
    }

    json data = json::parse(ifs);
    if (data.contains("version")) {
      quant_args_.quant_method() = data["version"].get<std::string>();
    }
    if (data.contains("bits")) {
      quant_args_.bits() = data["bits"].get<int64_t>();
    }
    if (data.contains("group_size")) {
      quant_args_.group_size() = data["group_size"].get<int64_t>();
    }
    if (data.contains("desc_act")) {
      quant_args_.desc_act() = data["desc_act"].get<bool>();
    }
    if (data.contains("true_sequential")) {
      quant_args_.true_sequential() = data["true_sequential"].get<bool>();
    }
  }

  return true;
}

}  // namespace llm
