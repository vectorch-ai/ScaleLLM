#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "common/slice.h"

namespace llm {
// Physical memory used for key and value cache in attention layers
// the fixed memory is allocated in the constructor for each attention layer.
class KVCache final {
 public:
  KVCache() = default;

  KVCache(torch::Tensor key_cache, torch::Tensor value_cache);

  // check if the key and value cache is empty
  bool empty() const {
    return !key_cache_.defined() || !value_cache_.defined();
  }

  // get key and value cache tensors
  std::tuple<torch::Tensor, torch::Tensor> get_kv_cache() const {
    return {key_cache_, value_cache_};
  }

  int64_t block_size() const { return block_size_; }

  // set key and value cache for the given slot_ids
  // the slot_ids are the indices of the key/value cache, [num_slots] IntTensor
  // keys/values: [num_slots, num_heads, head_dim]
  void set_kv_cache(const torch::Tensor& slot_ids,
                    const torch::Tensor& keys,
                    const torch::Tensor& values);

  // put following functions as public for testing/benchmarking
  void set_kv_cache_slow(const torch::Tensor& slot_ids,
                         const torch::Tensor& keys,
                         const torch::Tensor& values);

  void set_kv_cache_cuda(const torch::Tensor& slot_ids,
                         const torch::Tensor& keys,
                         const torch::Tensor& values);

  void set_kv_cache(const Slice<int32_t>& slot_ids,
                    const torch::Tensor& keys,
                    const torch::Tensor& values);

  std::tuple<torch::Tensor, torch::Tensor> get_kv_cache(
      const Slice<int32_t>& slot_ids) const;

  std::tuple<torch::Tensor, torch::Tensor> get_kv_cache(
      const torch::Tensor& slot_ids) const;

 private:
  int64_t block_size_ = 0;

  // TODO: allocate cache with shape: [n_slots, num_heads, 2, head_dim]
  // the contunuous memory region for key and value cache would be splited into
  // fixed size blocks. the blocks allocation would be managed by the
  // blockallocator.
  // [n_slots, num_heads, head_dim]
  torch::Tensor key_cache_;
  // [n_slots, num_heads, head_dim]
  torch::Tensor value_cache_;
};

}  // namespace llm
