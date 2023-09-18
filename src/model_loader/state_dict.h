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
      int shard_id,
      int num_shards);

  static std::unique_ptr<StateDict> load_safetensors(
      const std::string& weights_file,
      int shard_id,
      int num_shards);

  StateDict(std::unordered_map<std::string, torch::Tensor> dict,
            int shard_id,
            int num_shards);

  StateDict(std::unique_ptr<folly::MemoryMapping> mem_map,
            std::unordered_map<std::string, torch::Tensor> dict,
            int shard_id,
            int num_shards);

  // get the tensor with the given name. return nullptr if not found.
  torch::Tensor get_tensor(const std::string_view& tensor_name) const;

  // get the sharded tensor with the given prefix+name for the given rank.
  // N.B. An assumption is that those tensors can be loaded from one file.
  torch::Tensor get_sharded_tensor(
      const std::vector<std::string_view>& prefixes,
      const std::string_view& tensor_name,
      int64_t dim,
      int rank,
      int world_size) const;

  // get the sharded tensor with the given name for the given rank.
  torch::Tensor get_sharded_tensor(const std::string_view& tensor_name,
                                   int64_t dim,
                                   int rank,
                                   int world_size) const;

  // select all the tensors whose name starts with prefix.
  // the returned tensor name will be the suffix of the original name.
  StateDict select(const std::string_view& prefix) const;

  size_t size() const { return dict_.size(); }

  // support range-based for loop
  auto begin() const { return dict_.begin(); }
  auto end() const { return dict_.end(); }

 private:
  // memory mapping for safetensors
  std::unique_ptr<folly::MemoryMapping> mem_map_;

  std::unordered_map<std::string, torch::Tensor> dict_;

  // configs for data shards
  // data shard id of this weight file, valid range [0, num_shards)
  int shard_id_ = 0;
  // total number of data shards
  int num_shards_ = 1;
};
}  // namespace llm
