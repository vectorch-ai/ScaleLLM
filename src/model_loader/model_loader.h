#pragma once

#include <torch/torch.h>

#include <vector>

#include "layers/quantization/quant_args.h"
#include "model_loader/state_dict.h"
#include "models/model_args.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_args.h"

namespace llm {

// Iterator for StateDict, load the model weights file one by one to save
// memory
class StateDictIterator {
 public:
  StateDictIterator(const std::vector<std::string>& model_weights_files,
                    size_t index,
                    bool is_pickle,
                    bool is_sharded);

  // load the next model weights file
  void operator++() {
    // reset the state dict
    state_dict_.reset();
    // move to the next model weights file
    index_++;
  }

  // return the current model weights file
  const StateDict& operator*() const { return *get_state_dict(); }

  // return the current model weights file
  const StateDict* operator->() const { return get_state_dict(); }

  // return true if the iterator reaches the end
  bool operator==(const StateDictIterator& other) const {
    return model_weights_files_ == other.model_weights_files_ &&
           index_ == other.index_;
  }

  // return true if the iterator does not reach the end
  bool operator!=(const StateDictIterator& other) const {
    return !(*this == other);
  }

 private:
  const StateDict* get_state_dict() const;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::vector<std::string>& model_weights_files_;
  // index of the current model weights file
  size_t index_ = 0;
  // whether the model weights file is a pickle file
  bool is_pickle_ = false;

  // pointer to the current state dict, lazy loaded
  mutable std::unique_ptr<StateDict> state_dict_;
};

class ModelLoader {
 public:
  virtual ~ModelLoader() = default;

  virtual const ModelArgs& model_args() const = 0;
  virtual const QuantArgs& quant_args() const = 0;
  virtual const TokenizerArgs& tokenizer_args() const = 0;

  virtual std::unique_ptr<Tokenizer> tokenizer() const = 0;

  virtual size_t weights_files_count() const = 0;
  virtual StateDictIterator begin() const = 0;
  virtual StateDictIterator end() const = 0;

  // create a model loader from the given path
  static std::unique_ptr<ModelLoader> create(
      const std::string& model_weights_path);
};

// A model loader for huggingface model files.
class HFModelLoader : public ModelLoader {
 public:
  HFModelLoader(const std::string& model_weights_path);

  const ModelArgs& model_args() const override { return args_; }

  const QuantArgs& quant_args() const override { return quant_args_; }

  const TokenizerArgs& tokenizer_args() const override {
    return tokenizer_args_;
  }

  std::unique_ptr<Tokenizer> tokenizer() const override;

  size_t weights_files_count() const override {
    return model_weights_files_.size();
  }

  // support range-based for loop
  StateDictIterator begin() const override {
    return {model_weights_files_, 0, is_pickle_, false};
  }
  StateDictIterator end() const override {
    return {model_weights_files_, weights_files_count(), true, false};
  }

 private:
  bool load_model_args(const std::string& args_file_path);

  std::string model_weights_path_;

  // loaded model args
  ModelArgs args_;

  // quantization args
  QuantArgs quant_args_;

  TokenizerArgs tokenizer_args_;

  // sorted model weights files
  std::vector<std::string> model_weights_files_;
  // is pickle or safetensors
  bool is_pickle_ = false;
};
}  // namespace llm
