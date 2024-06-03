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
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : n_kv_heads_(n_kv_heads), head_dim_(head_dim) {
  // calculate logical kv heads with support of MQA/GQA
  const int32_t world_size = parallel_args.world_size();
  int64_t effective_out_features = (n_heads + 2 * n_kv_heads) * head_dim;
  if (n_kv_heads >= world_size) {
    // partition kv heads evenly across world_size for MHA
    CHECK_EQ(n_kv_heads % world_size, 0)
        << "kv_heads can't be partitioned evenly across world_size";
  } else {
    // replicate kv heads evenly across world_size for GQA/MQA
    CHECK_EQ(world_size % n_kv_heads, 0)
        << "kv heads can't be replicated evenly across world_size";
    kv_replication_ratio_ = world_size / n_kv_heads;
    effective_out_features = (n_heads + 2 * world_size) * head_dim;
  }

  parallel_linear_ =
      register_module("parallel_linear",
                      ColumnParallelLinear(hidden_size,
                                           effective_out_features,
                                           bias,
                                           gather_output,
                                           quant_args,
                                           parallel_args,
                                           options));
}

// special load_state_dict for fused cases
void QKVColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes,
    const std::vector<std::string>& kv_prefixes) {
  if (kv_replication_ratio_ > 1) {
    // replicate kv heads
    auto kv_replicated_state_dict = state_dict.select_with_transform(
        "", [&](const std::string& name, const torch::Tensor& tensor) {
          for (const auto& kv_prefix : kv_prefixes) {
            if (absl::StartsWith(name, kv_prefix)) {
              // reshape to [n_kv_heads, head_dim, ...]
              auto reshaped_tensor =
                  tensor.reshape({n_kv_heads_, head_dim_, -1});
              // interleave repeat kv heads along kv_head dim
              reshaped_tensor = reshaped_tensor.repeat_interleave(
                  kv_replication_ratio_, /*dim=*/0);
              // reshape to [n_kv_heads * kv_replication_ratio * head_dim, ...]
              return reshaped_tensor.reshape(
                  {n_kv_heads_ * kv_replication_ratio_ * head_dim_, -1});
            }
          }
          return tensor;
        });
    parallel_linear_->load_state_dict(kv_replicated_state_dict, prefixes);
  } else {
    parallel_linear_->load_state_dict(state_dict, prefixes);
  }
}

}  // namespace llm
