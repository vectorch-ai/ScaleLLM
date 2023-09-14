#pragma once
#include <c10/core/DeviceType.h>
#include <folly/system/MemoryMapping.h>
#include <torch/torch.h>

#include <memory>
#include <unordered_map>

namespace llm {

class StateDict final {
 public:
  static std::unique_ptr<StateDict> load_pickle_file(
      const std::string& weights_file,
      const torch::Device& device = torch::kCPU);

  static std::unique_ptr<StateDict> load_safetensors(
      const std::string& weights_file,
      const torch::Device& device = torch::kCPU);

  explicit StateDict(std::unordered_map<std::string, torch::Tensor> dict)
      : dict_(std::move(dict)) {}

  StateDict(std::unique_ptr<folly::MemoryMapping> mem_map,
            std::unordered_map<std::string, torch::Tensor> dict)
      : mem_map_(std::move(mem_map)), dict_(std::move(dict)) {}

  // get the tensor with the given name. return nullptr if not found.
  torch::Tensor get_tensor(const std::string_view& tensor_name) const;

  // select all the tensors whose name starts with prefix.
  // the returned tensor name will be the suffix of the original name.
  StateDict select(const std::string_view& prefix) const;

  // support range-based for loop
  auto begin() const { return dict_.begin(); }
  auto end() const { return dict_.end(); }

 private:
  // memory mapping for safetensors
  std::unique_ptr<folly::MemoryMapping> mem_map_;

  std::unordered_map<std::string, torch::Tensor> dict_;
};
}  // namespace llm
