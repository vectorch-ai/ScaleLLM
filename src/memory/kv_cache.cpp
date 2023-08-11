#include "kv_cache.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace llm {

KVCache::KVCache(torch::Tensor key_cache, torch::Tensor value_cache)
    : num_heads_(value_cache.size(1)),
      head_size_(value_cache.size(2)),
      block_size_(value_cache.size(3)),
      x_(key_cache.size(-1)),
      key_cache_(std::move(key_cache)),
      value_cache_(std::move(value_cache)) {}

void KVCache::set_kv_cache(const std::vector<uint32_t>& slot_ids,
                           const torch::Tensor& keys,
                           const torch::Tensor& values) {
  DCHECK_EQ(slot_ids.size(), keys.size(0));
  DCHECK_EQ(slot_ids.size(), values.size(0));

  using torch::indexing::Slice;
  for (int64_t i = 0; i < slot_ids.size(); ++i) {
    const uint32_t slot_id = slot_ids[i];
    const auto block_id = slot_id / block_size_;
    const auto block_offset = slot_id % block_size_;

    const auto key = keys[i];
    const auto value = values[i];

    // key_cache_[block_id, :, :, block_offset, :] = key.reshape(-1, head_size_ / x_, x_)
    key_cache_.index_put_({block_id, Slice(), Slice(), block_offset, Slice()},
                          key.reshape({-1, head_size_ / x_, x_}));
    // value_cache_[block_id, :, :, block_offset] = value
    value_cache_.index_put_({block_id, Slice(), Slice(), block_offset}, value);
  }
}

std::tuple<torch::Tensor, torch::Tensor> KVCache::get_kv_cache(
    const std::vector<int64_t>& slot_ids) const {
  std::vector<torch::Tensor> keys;
  keys.reserve(slot_ids.size());
  std::vector<torch::Tensor> values;
  values.reserve(slot_ids.size());

  using torch::indexing::Slice;
  for (int64_t slot_id : slot_ids) {
    const auto block_id = slot_id / block_size_;
    const auto block_offset = slot_id % block_size_;
    // key = key_cache_[block_id, :, :, block_offset, :]
    const auto key =
        key_cache_.index({block_id, Slice(), Slice(), block_offset, Slice()});
    keys.push_back(key.reshape({num_heads_, head_size_}));
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
  // construct slot ids for the sequence
  std::vector<int64_t> slot_ids;
  slot_ids.reserve(context_len);
  for (int64_t i = 0; i < context_len; ++i) {
    const int64_t block_id = block_table[i / block_size_].item<int64_t>();
    const int64_t block_offset = i % block_size_;
    const int64_t slot_id = block_id * block_size_ + block_offset;
    slot_ids.push_back(slot_id);
  }
  return get_kv_cache(slot_ids);
}

}  // namespace llm
