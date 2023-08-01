#
#pragma once
#include <torch/torch.h>
#include <unordered_map>

namespace llm {

// A wrapper around torch::Dict<torch::IValue, torch::IValue>.
class StateDict final {
 public:
  static StateDict load_from_file(const std::string& model_path,
                                  torch::DeviceType device_type = torch::kCPU);

  explicit StateDict(std::unordered_map<std::string, torch::Tensor> dict)
      : dict_(std::move(dict)) {}

  // get the tensor with the given name. return nullptr if not found.
  torch::Tensor get_tensor(const std::string_view& tensor_name) const;

  // select all the tensors whose name starts with prefix.
  // the returned tensor name will be the suffix of the original name.
  StateDict select(const std::string_view& prefix) const;

  // support range-based for loop
  auto begin() const { return dict_.begin(); }
  auto end() const { return dict_.end(); }

 private:
  std::unordered_map<std::string, torch::Tensor> dict_;
};
}  // namespace llm
