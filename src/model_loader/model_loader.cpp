#include "model_loader.h"

#include <gflags/gflags.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <filesystem>
#include <fstream>
#include <vector>

#include "common/json_reader.h"
#include "common/logging.h"
#include "model_loader/state_dict.h"
#include "models/args.h"
#include "models/model_registry.h"
#include "tokenizer/hf_tokenizer.h"
#include "tokenizer/sentencepiece_tokenizer.h"

DEFINE_string(model_type, "", "model type, e.g. llama2, llama, gpt_neox");
DEFINE_string(quant_method, "", "quantization method, e.g. awq, gptq");

DEFINE_string(tokenizer_path, "", "Path to the tokenizer file.");

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
  GCHECK(index_ < num_weight_files);
  // lazy loading
  if (!state_dict_) {
    GLOG(INFO) << "Loading model weights from " << model_weights_files_[index_];

    const int shard_id = is_sharded_ ? static_cast<int>(index_) : 0;
    const int num_shards = is_sharded_ ? static_cast<int>(num_weight_files) : 1;
    if (is_pickle_) {
      state_dict_ = StateDict::load_pickle_file(
          model_weights_files_[index_], shard_id, num_shards);
    } else {
      state_dict_ = StateDict::load_safetensors(
          model_weights_files_[index_], shard_id, num_shards);
    }
  }
  return state_dict_.get();
}

std::unique_ptr<ModelLoader> ModelLoader::create(
    const std::string& model_weights_path) {
  bool has_hf_weight_files = false;
  for (const auto& entry :
       std::filesystem::directory_iterator(model_weights_path)) {
    if (entry.path().extension() == ".safetensors" ||
        entry.path().extension() == ".bin") {
      has_hf_weight_files = true;
      break;
    }
  }
  if (has_hf_weight_files) {
    return std::make_unique<HFModelLoader>(model_weights_path);
  }
  return std::make_unique<PTModelLoader>(model_weights_path);
}

std::unique_ptr<Tokenizer> PTModelLoader::tokenizer() const {
  // use the tokenizer path specified by the user if exists
  const std::string tokenizer_path =
      !FLAGS_tokenizer_path.empty() ? FLAGS_tokenizer_path
                                    : model_weights_path_ + "/tokenizer.model";
  if (!std::filesystem::exists(tokenizer_path)) {
    GLOG(ERROR) << "Failed to find tokenizer file: " << tokenizer_path;
    return nullptr;
  }
  return std::make_unique<SentencePieceTokenizer>(tokenizer_path);
}

PTModelLoader::PTModelLoader(const std::string& model_weights_path)
    : model_weights_path_(model_weights_path) {
  const std::string args_file_path = model_weights_path + "/params.json";
  GCHECK(load_model_args(args_file_path))
      << "Failed to load model args from " << args_file_path;
  for (const auto& entry :
       std::filesystem::directory_iterator(model_weights_path)) {
    if (entry.path().extension() == ".pth") {
      model_weights_files_.push_back(entry.path().string());
    }
  }
  GCHECK(!model_weights_files_.empty())
      << "Failed to find model weights files in " << model_weights_path;
  // sort the model weights files by name
  std::sort(model_weights_files_.begin(), model_weights_files_.end());
}

bool PTModelLoader::load_model_args(const std::string& args_file_path) {
  JsonReader reader;
  if (!reader.parse(args_file_path)) {
    GLOG(ERROR) << "Failed to parse model args file: " << args_file_path;
    return false;
  }
  // hardcode the model type to llama2 for now.
  args_.model_type() = "llama2";
  auto args_loader = ModelRegistry::get_model_args_loader(args_.model_type());
  if (args_loader == nullptr) {
    GLOG(ERROR) << "Failed to find model args loader for model type "
                << args_.model_type();
    return false;
  }
  return args_loader(reader, &args_);
}

HFModelLoader::HFModelLoader(const std::string& model_weights_path)
    : model_weights_path_(model_weights_path) {
  GCHECK(load_model_args(model_weights_path));
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
  GCHECK(!model_weights_files_.empty())
      << "Failed to find model weights files in " << model_weights_path;
  // sort the model weights files by name
  std::sort(model_weights_files_.begin(), model_weights_files_.end());
}

std::unique_ptr<Tokenizer> HFModelLoader::tokenizer() const {
  // use the tokenizer path specified by the user if exists
  const std::string tokenizer_path = model_weights_path_ + "/tokenizer.json";
  if (std::filesystem::exists(tokenizer_path)) {
    return HFTokenizer::from_file(tokenizer_path);
  }

  const std::string vocab_path = model_weights_path_ + "/tokenizer.model";
  if (std::filesystem::exists(vocab_path)) {
    GLOG(WARNING) << "Failed to find tokenizer.json, use tokenizer.model "
                     "instead. Please consider to convert the tokenizer.model "
                     "to tokenizer.json for better performance.";
    return std::make_unique<SentencePieceTokenizer>(vocab_path);
  }

  GLOG(ERROR)
      << "Failed to find tokenizer file tokenizer.json or tokenizer.model from "
      << model_weights_path_;
  return nullptr;
}

bool HFModelLoader::load_model_args(const std::string& model_weights_path) {
  JsonReader reader;
  const std::string args_file_path = model_weights_path + "/config.json";
  if (!reader.parse(args_file_path)) {
    GLOG(ERROR) << "Failed to parse model args file: " << args_file_path;
    return false;
  }

  if (auto data = reader.value<std::string>("model_type")) {
    args_.model_type() = data.value();
  } else {
    GLOG(ERROR) << "Failed to find model_type in " << args_file_path;
    return false;
  }

  auto args_loader = ModelRegistry::get_model_args_loader(args_.model_type());
  if (args_loader == nullptr) {
    GLOG(ERROR) << "Failed to find model args loader for model type "
                << args_.model_type();
    return false;
  }
  args_loader(reader, &args_);

  // load quantization args if exists
  if (reader.contains("quantization_config")) {
    if (auto v =
            reader.value<std::string>("quantization_config.quant_method")) {
      quant_args_.quant_method() = v.value();
    }
    if (auto v = reader.value<int64_t>("quantization_config.bits")) {
      quant_args_.bits() = v.value();
    }
    if (auto v = reader.value<int64_t>("quantization_config.group_size")) {
      quant_args_.group_size() = v.value();
    }
    if (auto v = reader.value<bool>("quantization_config.desc_act")) {
      quant_args_.desc_act() = v.value();
    }
    if (auto v = reader.value<bool>("quantization_config.true_sequential")) {
      quant_args_.true_sequential() = v.value();
    }
  }

  // load quantization args for awq if exists
  JsonReader awq_reader;
  const std::string quant_args_file_path =
      model_weights_path + "/quant_config.json";
  if (awq_reader.parse(quant_args_file_path)) {
    // hardcode the quant_method to awq if not exists
    quant_args_.quant_method() =
        awq_reader.value_or<std::string>("quant_method", "awq");

    if (auto v = awq_reader.value<int64_t>("w_bit")) {
      quant_args_.bits() = v.value();
    }
    if (auto v = awq_reader.value<int64_t>("q_group_size")) {
      quant_args_.group_size() = v.value();
    }
  }

  // load quantization args for gptq if exists
  const std::string gptq_args_file_path =
      model_weights_path + "/quantize_config.json";
  JsonReader gptq_reader;
  if (gptq_reader.parse(gptq_args_file_path)) {
    // hardcode the quant_method to gptq if not exists
    quant_args_.quant_method() =
        gptq_reader.value_or<std::string>("quant_method", "gptq");

    if (auto v = gptq_reader.value<int64_t>("bits")) {
      quant_args_.bits() = v.value();
    }
    if (auto v = gptq_reader.value<int64_t>("group_size")) {
      quant_args_.group_size() = v.value();
    }
    if (auto v = gptq_reader.value<bool>("desc_act")) {
      quant_args_.desc_act() = v.value();
    }
    if (auto v = gptq_reader.value<bool>("true_sequential")) {
      quant_args_.true_sequential() = v.value();
    }
  }

  if (!FLAGS_quant_method.empty() &&
      quant_args_.quant_method() != FLAGS_quant_method) {
    GLOG(WARNING) << "Overwriting quant_method to " << FLAGS_quant_method;
    quant_args_.quant_method() = FLAGS_quant_method;
  }

  return true;
}

}  // namespace llm
