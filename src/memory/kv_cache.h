#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace llm {
// Physical memory used for key and value cache in attention layers
// the fixed memory is allocated in the constructor for each attention layer.
class KVCache final {
 public:
  KVCache() = default;

  // TODO: pass in kv_shape and options instead
  KVCache(torch::Tensor key_cache, torch::Tensor value_cache);

  // check if the key and value cache is empty
  bool empty() const {
    return !key_cache_.defined() || !value_cache_.defined();
  }

  // get key and value cache tensors
  std::tuple<torch::Tensor, torch::Tensor> get_kv_cache() const {
    return {key_cache_, value_cache_};
  }

  std::tuple<torch::Tensor, torch::Tensor, int32_t> get_kv_cache_slot_view()
      const {
    return {key_cache_.view({-1, num_kv_heads_, head_size_}),
            value_cache_.view({-1, num_kv_heads_, head_size_}),
            block_size_};
  }

  // set key and value cache for the given slot_ids
  // the slot_ids are the indices of the key/value cache, [num_slots] IntTensor
  // keys/values: [num_slots, num_heads, head_dim]
  void set_kv_cache(const torch::Tensor& slot_ids,
                    const torch::Tensor& keys,
                    const torch::Tensor& values);

  // get key and value cache for a sequence based on physical memory blocks
  // block_table: [num_blocks] IntTensor
  // context_len: the length of the sequence
  // returns keys/values: [context_len, num_heads, head_dim]
  std::tuple<torch::Tensor, torch::Tensor> get_kv_cache(
      const torch::Tensor& block_table,
      int64_t context_len) const;

  // put following functions as public for testing/benchmarking
  void set_kv_cache_slow(const torch::Tensor& slot_ids,
                         const torch::Tensor& keys,
                         const torch::Tensor& values);

  void set_kv_cache_cuda(const torch::Tensor& slot_ids,
                         const torch::Tensor& keys,
                         const torch::Tensor& values);

  std::tuple<torch::Tensor, torch::Tensor> get_kv_cache(
      const torch::Tensor& slot_ids) const;

  std::tuple<torch::Tensor, torch::Tensor> get_kv_cache(
      const torch::Tensor& block_tables,
      const torch::Tensor& kv_cu_seq_lens) const;

 private:
  std::tuple<torch::Tensor, torch::Tensor> get_kv_cache(
      const std::vector<int>& slot_ids) const;

  int64_t num_kv_heads_ = 0;
  int64_t head_size_ = 0;
  int64_t block_size_ = 0;

  // the contunuous memory region for key and value cache would be splited into
  // fixed size blocks. the blocks allocation would be managed by the
  // blockallocator.
  // [num_blocks, block_size, num_heads, head_dim]
  torch::Tensor key_cache_;
  // [num_blocks, block_size, num_heads, head_dim]
  torch::Tensor value_cache_;
};

}  // namespace llm
