#include "kv_cache.h"

#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

extern void reshape_and_cache(torch::Tensor& key,
                              torch::Tensor& value,
                              torch::Tensor& key_cache,
                              torch::Tensor& value_cache,
                              torch::Tensor& slot_mapping);

namespace llm {

KVCache::KVCache(torch::Tensor key_cache, torch::Tensor value_cache)
    : num_kv_heads_(value_cache.size(1)),
      head_size_(value_cache.size(2)),
      block_size_(value_cache.size(3)),
      x_(key_cache.size(-1)),
      key_cache_(std::move(key_cache)),
      value_cache_(std::move(value_cache)) {}

void KVCache::set_kv_cache(torch::Tensor slot_ids,
                           torch::Tensor keys,
                           torch::Tensor values) {
  DCHECK_EQ(slot_ids.size(0), keys.size(0));
  DCHECK_EQ(slot_ids.size(0), values.size(0));
  DCHECK_EQ(slot_ids.device(), keys.device());
  DCHECK_EQ(slot_ids.device(), values.device());

  auto slot_ids_cpu = slot_ids.cpu();
  const int32_t* ids = slot_ids_cpu.data_ptr<int32_t>();
  const auto num_tokens = keys.size(0);

  using torch::indexing::Slice;
  for (int64_t i = 0; i < num_tokens; ++i) {
    const int32_t slot_id = ids[i];
    const auto block_id = slot_id / block_size_;
    const auto block_offset = slot_id % block_size_;

    const auto key = keys[i];
    const auto value = values[i];

    // key_cache_[block_id, :, :, block_offset, :] =
    // key.reshape(-1, head_size_/ x_, x_)
    key_cache_.index_put_({block_id, Slice(), Slice(), block_offset, Slice()},
                          key.reshape({-1, head_size_ / x_, x_}));
    // value_cache_[block_id, :, :, block_offset] = value
    value_cache_.index_put_({block_id, Slice(), Slice(), block_offset}, value);
  }
}

void KVCache::set_kv_cache_cuda(torch::Tensor slot_ids,
                                torch::Tensor keys,
                                torch::Tensor values) {
  DCHECK_EQ(slot_ids.dtype(), torch::kInt);
  DCHECK_EQ(slot_ids.size(0), keys.size(0));
  DCHECK_EQ(slot_ids.size(0), values.size(0));
  DCHECK_EQ(slot_ids.device(), keys.device());
  DCHECK_EQ(slot_ids.device(), values.device());

  reshape_and_cache(keys, values, key_cache_, value_cache_, slot_ids);
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
    // key = key_cache_[block_id, :, :, block_offset, :]
    const auto key =
        key_cache_.index({block_id, Slice(), Slice(), block_offset, Slice()});
    keys.push_back(key.reshape({num_kv_heads_, head_size_}));
    // value = value_cache_[block_id, :, :, block_offset]
    const auto value =
        value_cache_.index({block_id, Slice(), Slice(), block_offset});
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
