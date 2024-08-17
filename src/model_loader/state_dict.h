#pragma once
#include <c10/core/DeviceType.h>
#include <folly/system/MemoryMapping.h>
#include <torch/torch.h>

#include <memory>
#include <string_view>
#include <unordered_map>

namespace llm {

class StateDict final {
 public:
  static std::unique_ptr<StateDict> load(const std::string& weights_file,
                                         bool is_pickle);

  static std::unique_ptr<StateDict> load_pickle_file(
      const std::string& weights_file);

  static std::unique_ptr<StateDict> load_safetensors(
      const std::string& weights_file);

  StateDict(std::unordered_map<std::string, torch::Tensor> dict,
            const std::string& prefix = "");

  StateDict(std::unique_ptr<folly::MemoryMapping> mem_map,
            std::unordered_map<std::string, torch::Tensor> dict);

  // get the tensor with the given name. return nullptr if not found.
  torch::Tensor get_tensor(const std::string& tensor_name) const;

  // get the sharded tensor with the given name for the given rank.
  torch::Tensor get_sharded_tensor(const std::string& tensor_name,
                                   int64_t dim,
                                   int rank,
                                   int world_size) const;

  // select all the tensors whose name starts with prefix.
  // the returned tensor name will be the suffix of the original name.
  StateDict select(const std::string& prefix) const;

  // select all tensors whose name starts with prefix and apply the transform
  // for each tensor.
  using TensorTransform =
      std::function<torch::Tensor(const std::string&, const torch::Tensor&)>;
  StateDict select_with_transform(const std::string& prefix,
                                  TensorTransform transform_func) const;

  size_t size() const { return dict_.size(); }

  std::string_view prefix() const { return prefix_; }

  // support range-based for loop
  auto begin() const { return dict_.begin(); }
  auto end() const { return dict_.end(); }

 private:
  // memory mapping for safetensors
  std::unique_ptr<folly::MemoryMapping> mem_map_;

  std::unordered_map<std::string, torch::Tensor> dict_;

  TensorTransform transform_func_ = nullptr;

  // prefix for debug purpose
  std::string prefix_;
};
}  // namespace llm
