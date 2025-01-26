#include "kv_cache.h"

#include <ATen/core/TensorBody.h>
#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "kernels/kv_cache_kernels.h"

namespace llm {
using ISlice = torch::indexing::Slice;

// [num_blocks, block_size, num_kv_heads, head_dim]
KVCache::KVCache(torch::Tensor key_cache, torch::Tensor value_cache)
    : n_kv_heads_(value_cache.size(-2)),
      head_dim_(value_cache.size(-1)),
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
  const torch::Tensor slot_ids_cpu = slot_ids.cpu();
  const int32_t* ids = slot_ids_cpu.data_ptr<int32_t>();
  const auto n_slots = slot_ids_cpu.numel();
  set_kv_cache(Slice<int32_t>(ids, n_slots), keys, values);
}

void KVCache::set_kv_cache_cuda(const torch::Tensor& slot_ids,
                                const torch::Tensor& keys,
                                const torch::Tensor& values) {
  kernel::set_kv_cache(slot_ids, keys, values, key_cache_, value_cache_);
}

// keys/values: [n_tokens, n_kv_heads, head_dim]
void KVCache::set_kv_cache(const Slice<int32_t>& slot_ids,
                           const torch::Tensor& keys,
                           const torch::Tensor& values) {
  const auto n_tokens = keys.size(0);
  CHECK(slot_ids.size() == n_tokens);

  // set key and value into cache one by one
  for (int64_t i = 0; i < n_tokens; ++i) {
    const int32_t slot_id = slot_ids[i];
    const auto block_id = slot_id / block_size_;
    const auto block_offset = slot_id % block_size_;

    // [block_id, block_offset, n_kv_heads, head_dim]
    key_cache_.index_put_({block_id, block_offset, ISlice(), ISlice()},
                          keys[i]);
    value_cache_.index_put_({block_id, block_offset, ISlice(), ISlice()},
                            values[i]);
  }
}

std::tuple<torch::Tensor, torch::Tensor> KVCache::get_kv_cache(
    const torch::Tensor& slot_ids) const {
  DCHECK_EQ(slot_ids.dtype(), torch::kInt);

  const torch::Tensor slot_ids_cpu = slot_ids.cpu();
  const int32_t* ids = slot_ids_cpu.data_ptr<int32_t>();
  const auto n_slots = slot_ids_cpu.numel();
  return get_kv_cache(Slice<int32_t>(ids, n_slots));
}

std::tuple<torch::Tensor, torch::Tensor> KVCache::get_kv_cache(
    const Slice<int32_t>& slot_ids) const {
  std::vector<torch::Tensor> keys;
  keys.reserve(slot_ids.size());
  std::vector<torch::Tensor> values;
  values.reserve(slot_ids.size());

  for (int slot_id : slot_ids) {
    const int64_t block_id = slot_id / block_size_;
    const int64_t block_offset = slot_id % block_size_;
    // key = key_cache_[block_id, block_offset, :, :]
    const auto key =
        key_cache_.index({block_id, block_offset, ISlice(), ISlice()});
    keys.push_back(key);
    // value = value_cache_[block_id, block_offset, :, :]
    const auto value =
        value_cache_.index({block_id, block_offset, ISlice(), ISlice()});
    values.push_back(value);
  }
  return std::make_tuple(torch::stack(keys), torch::stack(values));
}

}  // namespace llm
