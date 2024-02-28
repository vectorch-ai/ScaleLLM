#include "kv_cache.h"

#include <ATen/core/TensorBody.h>
#include <c10/core/TensorImpl.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "common/logging.h"
#include "kernels/kv_cache_kernels.h"

namespace llm {

// [num_blocks, block_size, num_kv_heads, head_dim]
KVCache::KVCache(torch::Tensor key_cache, torch::Tensor value_cache)
    : num_kv_heads_(value_cache.size(-2)),
      head_size_(value_cache.size(-1)),
      block_size_(value_cache.size(-3)),
      key_cache_(std::move(key_cache)),
      value_cache_(std::move(value_cache)) {}

void KVCache::set_kv_cache(const torch::Tensor& slot_ids,
                           const torch::Tensor& keys,
                           const torch::Tensor& values) {
  DCHECK_EQ(slot_ids.size(0), keys.size(0));
  DCHECK_EQ(slot_ids.size(0), values.size(0));
  DCHECK_EQ(slot_ids.device(), keys.device());
  DCHECK_EQ(slot_ids.device(), values.device());

  if (keys.is_cuda()) {
    // use cuda kernel
    return set_kv_cache_cuda(slot_ids, keys, values);
  }
  return set_kv_cache_slow(slot_ids, keys, values);
}

void KVCache::set_kv_cache_slow(const torch::Tensor& slot_ids,
                                const torch::Tensor& keys,
                                const torch::Tensor& values) {
  auto slot_ids_cpu = slot_ids.cpu();
  const int32_t* ids = slot_ids_cpu.data_ptr<int32_t>();
  const auto num_tokens = keys.size(0);

  using torch::indexing::Slice;
  for (int64_t i = 0; i < num_tokens; ++i) {
    const int32_t slot_id = ids[i];
    const auto block_id = slot_id / block_size_;
    const auto block_offset = slot_id % block_size_;

    // key_cache_[block_id, block_offset, :, :] = key
    key_cache_.index_put_({block_id, block_offset, Slice(), Slice()}, keys[i]);
    // value_cache_[block_id, block_offset, :, :] = value
    value_cache_.index_put_({block_id, block_offset, Slice(), Slice()},
                            values[i]);
  }
}

void KVCache::set_kv_cache_cuda(const torch::Tensor& slot_ids,
                                const torch::Tensor& keys,
                                const torch::Tensor& values) {
  kernel::set_kv_cache(slot_ids, keys, values, key_cache_, value_cache_);
}

std::tuple<torch::Tensor, torch::Tensor> KVCache::get_kv_cache(
    const torch::Tensor& slot_ids) const {
  DCHECK_EQ(slot_ids.dtype(), torch::kInt);

  const torch::Tensor slot_ids_cpu = slot_ids.cpu();
  const int32_t* ids = slot_ids_cpu.data_ptr<int32_t>();
  const auto num_slots = slot_ids_cpu.numel();
  // construct slot ids for the sequence
  std::vector<int32_t> slot_ids_vec;
  slot_ids_vec.reserve(num_slots);
  for (int64_t i = 0; i < num_slots; ++i) {
    slot_ids_vec.push_back(ids[i]);
  }
  return get_kv_cache(slot_ids_vec);
}

std::tuple<torch::Tensor, torch::Tensor> KVCache::get_kv_cache(
    const std::vector<int>& slot_ids) const {
  std::vector<torch::Tensor> keys;
  keys.reserve(slot_ids.size());
  std::vector<torch::Tensor> values;
  values.reserve(slot_ids.size());

  using torch::indexing::Slice;
  for (int slot_id : slot_ids) {
    const auto block_id = slot_id / block_size_;
    const auto block_offset = slot_id % block_size_;
    // key = key_cache_[block_id, block_offset, :, :]
    const auto key =
        key_cache_.index({block_id, block_offset, Slice(), Slice()});
    keys.push_back(key.reshape({num_kv_heads_, head_size_}));
    // value = value_cache_[block_id, block_offset, :, :]
    const auto value =
        value_cache_.index({block_id, block_offset, Slice(), Slice()});
    values.push_back(value);
  }
  return std::make_tuple(torch::stack(keys), torch::stack(values));
}

std::tuple<torch::Tensor, torch::Tensor> KVCache::get_kv_cache(
    const torch::Tensor& block_table,
    int64_t context_len) const {
  const torch::Tensor block_table_cpu = block_table.cpu();
  const int32_t* block_ids = block_table_cpu.data_ptr<int32_t>();
  // construct slot ids for the sequence
  std::vector<int32_t> slot_ids;
  slot_ids.reserve(context_len);
  for (int64_t i = 0; i < context_len; ++i) {
    const int32_t block_id = block_ids[i / block_size_];
    const int32_t block_offset = i % block_size_;
    const int32_t slot_id = block_id * block_size_ + block_offset;
    slot_ids.push_back(slot_id);
  }
  return get_kv_cache(slot_ids);
}

}  // namespace llm
