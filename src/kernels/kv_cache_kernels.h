#pragma once
#include <torch/torch.h>

namespace llm::kernel {

void set_kv_cache(const torch::Tensor& slot_ids,
                  const torch::Tensor& keys,
                  const torch::Tensor& values,
                  torch::Tensor& key_cache,
                  torch::Tensor& value_cache);

}  // namespace llm::kernel
