#include "kv_cache.h"

#include <ATen/core/TensorBody.h>
#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "kernels/kv_cache_kernels.h"

namespace llm {

KVCache::KVCache(int64_t n_blocks,
                 int64_t block_size,
                 int64_t n_kv_heads,
                 int64_t head_dim,
                 const torch::TensorOptions& options)
    : block_size_(block_size) {
  // TODO: allocate cache with shape: [n_slots, num_heads, 2, head_dim]
  // [n_slots, n_kv_heads, head_dim]
  key_cache_ =
      torch::empty({n_blocks * block_size, n_kv_heads, head_dim}, options);
  value_cache_ =
      torch::empty({n_blocks * block_size, n_kv_heads, head_dim}, options);
}

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
    // [n_slots, n_kv_heads, head_dim]
    key_cache_[slot_id] = keys[i];
    value_cache_[slot_id] = values[i];
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
    // key/value_cache_[slot_id, :, :]
    keys.push_back(key_cache_[slot_id]);
    values.push_back(value_cache_[slot_id]);
  }
  return std::make_tuple(torch::stack(keys), torch::stack(values));
}

}  // namespace llm
