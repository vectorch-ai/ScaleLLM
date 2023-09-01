#pragma once

#include <torch/torch.h>

#include <vector>

#include "models/model_args.h"
#include "torch_utils/state_dict.h"

namespace llm {

// A class to load model weights from a given path and iterate over the model
// weights file one by one.
// for now only support loading pytorch model files.
// TODO: support HF safetensors: https://github.com/huggingface/safetensors
class ModelLoader {
 public:
  ModelLoader(std::string model_weights_path);

  const ModelArgs& model_args() const { return args_; }

  size_t weights_files_count() const { return model_weights_files_.size(); }

  // support range-based for loop
  class Iterator;
  Iterator begin() const { return {this, 0}; }
  Iterator end() const { return {this, weights_files_count()}; }

  // Iterator for StateDict, load the model weights file one by one to save
  // memory
  class Iterator {
   public:
    Iterator(const ModelLoader* loader, size_t index)
        : loader_(loader), index_(index) {}

    // load the next model weights file
    void operator++() { index_++; }

    // return the current model weights file
    const StateDict& operator*() const { return *get_state_dict(); }

    // return the current model weights file
    const StateDict* operator->() const { return get_state_dict(); }

    // return true if the iterator reaches the end
    bool operator==(const Iterator& other) const {
      return loader_ == other.loader_ && index_ == other.index_;
    }

    // return true if the iterator does not reach the end
    bool operator!=(const Iterator& other) const { return !(*this == other); }

   private:
    const StateDict* get_state_dict() const;

    const ModelLoader* loader_ = nullptr;
    size_t index_ = 0;
    mutable std::unique_ptr<StateDict> state_dict_;
  };

 private:
  // loaded model args
  ModelArgs args_;
  // path to the model weights
  std::string model_weights_path_;
  // sorted model weights files
  std::vector<std::string> model_weights_files_;
};
}  // namespace llm
