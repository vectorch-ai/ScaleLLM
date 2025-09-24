#include "qkv_linear.h"

#include <absl/strings/match.h>
#include <glog/logging.h>
#include <torch/torch.h>

namespace llm {
QKVColumnParallelLinearImpl::QKVColumnParallelLinearImpl(
    int64_t hidden_size,
    int64_t n_heads,
    int64_t n_kv_heads,
    int64_t head_dim,
    const std::vector<std::string>& prefixes,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  CHECK_EQ(prefixes.size(), 3)
      << "prefixes size must be 3 for q, k, v projections";

  // calculate logical kv heads with support of MQA/GQA
  const int32_t world_size = parallel_args.world_size();
  int64_t effective_kv_heads = n_kv_heads;
  // replication ratio of kv heads for MQA/GQA cases
  int64_t kv_replication_ratio = 0;

  if (n_kv_heads >= world_size) {
    // partition kv heads evenly across world_size for MHA
    CHECK_EQ(n_kv_heads % world_size, 0)
        << "kv_heads can't be partitioned evenly across world_size";
  } else {
    // replicate kv heads evenly across world_size for GQA/MQA
    CHECK_EQ(world_size % n_kv_heads, 0)
        << "kv heads can't be replicated evenly across world_size";
    kv_replication_ratio = world_size / n_kv_heads;
    effective_kv_heads = world_size;
  }

  // output features for Q, K, V
  std::vector<int64_t> out_features = {n_heads * head_dim,
                                       effective_kv_heads * head_dim,
                                       effective_kv_heads * head_dim};

  // create state_dict selector to handle MQA/GQA cases
  // for MQA/GQA cases, we need to replicate the weights of kv heads.
  auto state_dict_selector = [=](const StateDict& sd, const std::string&) {
    if (kv_replication_ratio <= 1) {
      return sd.select("");
    }
    // replicate kv heads for MQA/GQA cases
    return sd.select_with_transform(
        "", [=](const std::string& tensor_name, const torch::Tensor& tensor) {
          // skip query weights
          for (size_t i = 1; i < prefixes.size(); ++i) {
            const auto& kv_prefix = prefixes[i];
            if (absl::StartsWith(tensor_name, kv_prefix)) {
              // reshape to [n_kv_heads, head_dim, ...]
              auto reshaped_tensor = tensor.reshape({n_kv_heads, head_dim, -1});
              // interleave repeat kv heads along kv_head dim
              reshaped_tensor = reshaped_tensor.repeat_interleave(
                  kv_replication_ratio, /*dim=*/0);
              // reshape to [n_kv_heads * kv_replication_ratio * head_dim, ...]
              return reshaped_tensor.reshape(
                  {n_kv_heads * kv_replication_ratio * head_dim, -1});
            }
          }
          return tensor;
        });
  };

  parallel_linear_ = register_module("qkv_parallel_linear",
                                     FusedColumnParallelLinear(hidden_size,
                                                               out_features,
                                                               prefixes,
                                                               bias,
                                                               gather_output,
                                                               quant_args,
                                                               parallel_args,
                                                               options),
                                     state_dict_selector);
}

}  // namespace llm
