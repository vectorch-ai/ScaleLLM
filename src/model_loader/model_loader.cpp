#include "model_loader.h"

#include <torch/torch.h>

#include <filesystem>
#include <vector>

#include "model_loader/state_dict.h"
#include "models/model_args.h"

namespace llm {

ModelLoader::ModelLoader(std::string model_weights_path)
    : model_weights_path_(std::move(model_weights_path)) {
  const std::string args_file_path = model_weights_path_ + "/params.json";
  CHECK(args_.load_from_file(args_file_path))
      << "Failed to load model args from " << args_file_path;
  for (const auto& entry :
       std::filesystem::directory_iterator(model_weights_path_)) {
    if (entry.path().extension() == ".pth") {
      model_weights_files_.push_back(entry.path().string());
    }
  }
  // sort the model weights files by name
  std::sort(model_weights_files_.begin(), model_weights_files_.end());
}

const StateDict* ModelLoader::Iterator::get_state_dict() const {
  CHECK(index_ < loader_->model_weights_files_.size());
  // lazy loading
  if (!state_dict_) {
    state_dict_ = StateDict::load_pickle_file(
        loader_->model_weights_files_[index_], torch::kCPU);
    LOG(INFO) << "Loaded model weights from "
              << loader_->model_weights_files_[index_];
  }
  return state_dict_.get();
}
}  // namespace llm
