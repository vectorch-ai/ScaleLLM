#pragma once
#include <torch/torch.h>

#include <cstdint>

#include "common/slice.h"

namespace llm {
// Physical memory used for key and value cache in attention layers
// the fixed memory is allocated in the constructor for each attention layer.
class KVCache final {
 public:
  KVCache() = default;

  KVCache(int64_t n_blocks,
          int64_t block_size,
          int64_t n_kv_heads,
          int64_t head_dim,
          const torch::TensorOptions& options);

  // check if the key and value cache is empty
  bool empty() const { return block_size_ == 0; }

  int64_t block_size() const { return block_size_; }

  // get key and value cache tensors
  std::tuple<torch::Tensor, torch::Tensor> get_kv_cache() const {
    return {key_cache_, value_cache_};
  }

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

  // the contunuous memory region for key and value cache would be splited into
  // fixed size blocks. the blocks allocation would be managed by the
  // blockallocator.
  // [n_slots, num_heads, head_dim]
  torch::Tensor key_cache_;
  // [n_slots, num_heads, head_dim]
  torch::Tensor value_cache_;
};

}  // namespace llm
