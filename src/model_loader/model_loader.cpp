#include "model_loader.h"

#include <absl/strings/match.h>
#include <absl/strings/str_replace.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <filesystem>
#include <vector>

#include "args_overrider.h"
#include "common/json_reader.h"
#include "model_loader/state_dict.h"
#include "models/model_args.h"
#include "models/model_registry.h"
#include "tokenizer/hf_tokenizer.h"
#include "tokenizer/sentencepiece_tokenizer.h"
#include "tokenizer/tiktoken_tokenizer.h"

namespace llm {
StateDictIterator::StateDictIterator(
    const std::vector<std::string>& model_weights_files,
    size_t index,
    bool is_pickle,
    bool is_sharded)
    : model_weights_files_(model_weights_files),
      index_(index),
      is_pickle_(is_pickle) {}

const StateDict* StateDictIterator::get_state_dict() const {
  const size_t num_weight_files = model_weights_files_.size();
  CHECK(index_ < num_weight_files);
  // lazy loading
  if (!state_dict_) {
    LOG(INFO) << "Loading model weights from " << model_weights_files_[index_];
    state_dict_ = StateDict::load(model_weights_files_[index_], is_pickle_);
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
  CHECK(has_hf_weight_files)
      << "Failed to find model weights files (*.safetensors, *.bin) in "
      << model_weights_path;
  return std::make_unique<HFModelLoader>(model_weights_path);
}

HFModelLoader::HFModelLoader(const std::string& model_weights_path)
    : model_weights_path_(model_weights_path) {
  CHECK(load_model_args(model_weights_path))
      << "Failed to load model args from " << model_weights_path;
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
  // check if fast tokenizer exists
  const std::string tokenizer_path = model_weights_path_ + "/tokenizer.json";
  if (std::filesystem::exists(tokenizer_path)) {
    LOG(INFO) << "Using fast tokenizer.";
    // load fast tokenizer
    return HFTokenizer::from_file(tokenizer_path);
  }

  // fallback to sentencepiece/tiktoken tokenizer if no fast tokenizer exists
  if (tokenizer_args_.tokenizer_type() == "tiktoken") {
    LOG(INFO) << "Using Tiktoken tokenizer.";
    return std::make_unique<TiktokenTokenizer>(model_weights_path_,
                                               tokenizer_args_);
  }

  LOG(INFO) << "Using SentencePiece tokenizer.";
  return std::make_unique<SentencePieceTokenizer>(model_weights_path_,
                                                  tokenizer_args_);
}

bool HFModelLoader::load_model_args(const std::string& model_weights_path) {
  JsonReader reader;
  const std::string args_file_path = model_weights_path + "/config.json";
  if (!reader.parse(args_file_path)) {
    LOG(ERROR) << "Failed to parse model args file: " << args_file_path;
    return false;
  }

  std::string model_type;
  if (auto data = reader.value<std::string>("model_type")) {
    model_type = data.value();
  } else {
    LOG(ERROR) << "Failed to find model_type in " << args_file_path;
    return false;
  }

  auto args_loader = ModelRegistry::get_model_args_loader(model_type);
  if (args_loader == nullptr) {
    LOG(ERROR) << "Failed to find model args loader for model type "
               << model_type;
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

  // load tokenizer args from tokenizer_config.json if exists
  JsonReader tokenizer_reader;
  const std::string tokenizer_args_file_path =
      model_weights_path_ + "/tokenizer_config.json";
  if (tokenizer_reader.parse(tokenizer_args_file_path)) {
    // read chat template if exists
    if (auto v = tokenizer_reader.value<std::string>("chat_template")) {
      tokenizer_args_.chat_template() = v.value();
    }
  }

  auto tokenizer_args_loader =
      ModelRegistry::get_tokenizer_args_loader(args_.model_type());
  if (tokenizer_args_loader != nullptr) {
    if (!tokenizer_args_loader(tokenizer_reader, &tokenizer_args_)) {
      LOG(ERROR) << "Failed to load tokenizer args from "
                 << tokenizer_args_file_path;
      return false;
    }
  }

  // apply args override from gflag if exists
  override_args_from_gflag(args_, quant_args_, tokenizer_args_);

  // Some hacky logics to support loading of old models
  // always use float16 for quantization
  // TODO: support quantization for other data types
  if (!quant_args_.quant_method().empty() && args_.dtype() != "float16") {
    LOG(WARNING) << "Overwriting dtype from " << args_.dtype()
                 << " to float16 for quantization";
    args_.dtype() = "float16";
  }

  // fix chat template
  if (!tokenizer_args_.chat_template().empty()) {
    std::string chat_template = tokenizer_args_.chat_template();
    // replace "if not add_generation_prompt is defined" since the predence of
    // "not" and "defined" is different across implementations
    tokenizer_args_.chat_template() =
        absl::StrReplaceAll(chat_template,
                            {{"if not add_generation_prompt is defined",
                              "if add_generation_prompt is undefined"}});
  }
  return true;
}

}  // namespace llm
