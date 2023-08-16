#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace llm {
// GPU physical memory used for key and value cache in attention layers
// the fixed memory is allocated in the constructor for each attention layer
// and is never released.
class KVCache final {
 public:
  KVCache(torch::Tensor key_cache, torch::Tensor value_cache);

  // set key and value cache for the given slot_ids
  // the slot_ids are the indices of the key/value cache, [num_slots] IntTensor
  // keys/values: [num_slots, num_heads, head_dim]
  void set_kv_cache(const torch::Tensor& slot_ids,
                    const torch::Tensor& keys,
                    const torch::Tensor& values);

  // get key and value cache for the given slot_ids
  // the slot_ids are the indices of the key/value cache
  // returns keys/values: [num_slots, num_heads, head_dim]
  std::tuple<torch::Tensor, torch::Tensor> get_kv_cache(
      const std::vector<int>& slot_ids) const;

  // get key and value cache for a sequence based on physical memory blocks
  // block_table: [num_blocks] IntTensor
  // context_len: the length of the sequence
  // returns keys/values: [context_len, num_heads, head_dim]
  std::tuple<torch::Tensor, torch::Tensor> get_kv_cache(
      const torch::Tensor& block_table,
      int64_t context_len) const;

  // get key and value cache tensors
  torch::Tensor get_key_cache() const { return key_cache_; }
  torch::Tensor get_value_cache() const { return value_cache_; }

 private:
  int64_t num_heads_;
  int64_t head_size_;
  int64_t block_size_;
  int64_t x_;

  // the contunuous memory region for key and value cache would be splited into
  // fixed size blocks. the blocks allocation would be managed by the
  // blockallocator.
  // TODO: follow vllm key/value cache layout for now, refactor later
  // [num_blocks, num_heads, head_dim/x, block_size, x]
  torch::Tensor key_cache_;
  // [num_blocks, num_heads, head_dim, block_size]
  torch::Tensor value_cache_;
};

}  // namespace llm
