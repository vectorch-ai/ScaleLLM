#include "qlinear_gptq_marlin_impl.h"

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "layers/weight_utils.h"
#include "model_loader/state_dict.h"

namespace llm {
namespace {
int64_t round_up(int64_t num, int64_t multiple) {
  return ((num + multiple - 1) / multiple);
}

}  // namespace

ColumnParallelQLinearGPTQMarlinImpl::ColumnParallelQLinearGPTQMarlinImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantArgs& quant_args,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : bits_(quant_args.bits()),
      gather_output_(gather_output),
      parallel_args_(parallel_args) {
  const auto bits = quant_args.bits();
  const auto group_size =
      quant_args.group_size() > 0 ? quant_args.group_size() : in_features;
  const int64_t world_size = parallel_args.world_size();
  CHECK(out_features % world_size == 0)
      << "out_features " << out_features << " not divisible by world_size "
      << world_size;
  const int64_t out_features_per_partition = out_features / world_size;
  const int64_t pack_factor = 32 / bits;

  // gptq pack weight on dim 0
  qweight_ =
      torch::empty({in_features / pack_factor, out_features_per_partition},
                   options.dtype(torch::kInt32));

  // TODO: support tensor parallelism for gptq
  CHECK(out_features_per_partition % pack_factor == 0)
      << "out_features_per_partition " << out_features_per_partition
      << " not divisible by pack_factor " << pack_factor;
  const auto n_groups = round_up(in_features, group_size);
  qzeros_ = torch::empty({n_groups, out_features_per_partition / pack_factor},
                         options.dtype(torch::kInt32));

  scales_ = torch::empty({n_groups, out_features_per_partition}, options);

  // default g_idx values
  if (quant_args.desc_act()) {
    g_idx_ = torch::empty({in_features}, options.dtype(torch::kInt32));
  }

  if (bias) {
    bias_ = torch::empty({out_features_per_partition}, options);
  }
}

// load the weight from the checkpoint
void ColumnParallelQLinearGPTQMarlinImpl::load_state_dict(
    const StateDict& state_dict) {
  const auto rank = parallel_args_.rank();
  const auto world_size = parallel_args_.world_size();

  // load sharded weights on dim 1
  LOAD_SHARDED_WEIGHT(qweight, 1);
  LOAD_SHARDED_WEIGHT(qzeros, 1);
  LOAD_SHARDED_WEIGHT(scales, 1);

  if (g_idx_.defined()) {
    LOAD_WEIGHT(g_idx);
  }

  // load bias if defined
  if (bias_.defined()) {
    // load sharded bias on dim 0
    LOAD_SHARDED_WEIGHT(bias, 0);
  }
}

// special load_state_dict for fused cases
void ColumnParallelQLinearGPTQMarlinImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  const auto rank = parallel_args_.rank();
  const auto world_size = parallel_args_.world_size();

  // load and merge weights on dim 1
  LOAD_FUSED_WEIGHT(qweight, 1);
  LOAD_FUSED_WEIGHT(qzeros, 1);
  LOAD_FUSED_WEIGHT(scales, 1);

  if (g_idx_.defined()) {
    LOG(FATAL) << "fused weight does not support desc_act";
    // load and merge g_idx
    // LOAD_FUSED_WEIGHT(g_idx, 0);
  }

  // load bias if defined
  if (bias_.defined()) {
    // load and merge bias on dim 0
    LOAD_FUSED_WEIGHT(bias, 0);
  }
}

void ColumnParallelQLinearGPTQMarlinImpl::verify_loaded_weights(
    const std::string& prefix) const {
  CHECK(qweight_is_loaded_)
      << "qweight is not loaded for " << prefix + "qweight";
  CHECK(qzeros_is_loaded_) << "qzeros is not loaded for " << prefix + "qzeros";
  CHECK(scales_is_loaded_) << "scales is not loaded for " << prefix + "scales";
  CHECK(!g_idx_.defined() || g_idx_is_loaded_)
      << "g_idx is not loaded for " << prefix + "g_idx";
  CHECK(!bias_.defined() || bias_is_loaded_)
      << "bias is not loaded for " << prefix + "bias";
}

torch::Tensor ColumnParallelQLinearGPTQMarlinImpl::forward(
    torch::Tensor input) const {
  return input;
}

// RowParallelQLinearGPTQMarlinImpl
RowParallelQLinearGPTQMarlinImpl::RowParallelQLinearGPTQMarlinImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantArgs& quant_args,
    bool input_is_parallelized,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : bits_(quant_args.bits()),
      input_is_parallelized_(input_is_parallelized),
      parallel_args_(parallel_args) {
  const auto bits = quant_args.bits();
  const int64_t world_size = parallel_args.world_size();
  CHECK(in_features % world_size == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size;
  const int64_t in_features_per_partition = in_features / world_size;
  const int64_t pack_factor = 32 / bits;
  const auto group_size =
      quant_args.group_size() > 0 ? quant_args.group_size() : in_features;

  qweight_ =
      torch::empty({in_features_per_partition / pack_factor, out_features},
                   options.dtype(torch::kInt32));

  // load full qzeros and scales
  const auto n_groups = round_up(in_features, group_size);
  qzeros_ = torch::empty({n_groups, out_features / pack_factor},
                         options.dtype(torch::kInt32));
  scales_ = torch::empty({n_groups, out_features}, options);

  // load sharded g_idx on dim 0
  if (quant_args.desc_act()) {
    g_idx_ =
        torch::empty({in_features_per_partition}, options.dtype(torch::kInt32));
  }

  if (bias) {
    bias_ = torch::empty({out_features}, options);
  }
}

// load the weight from the checkpoint
void RowParallelQLinearGPTQMarlinImpl::load_state_dict(
    const StateDict& state_dict) {
  const auto rank = parallel_args_.rank();
  const auto world_size = parallel_args_.world_size();

  // load sharded weights on dim 0
  LOAD_SHARDED_WEIGHT(qweight, 0);
  LOAD_SHARDED_WEIGHT(qzeros, 0);
  LOAD_SHARDED_WEIGHT(scales, 0);

  if (g_idx_.defined()) {
    LOAD_SHARDED_WEIGHT(g_idx, 0);
  }

  if (bias_.defined()) {
    // load bias
    LOAD_WEIGHT(bias);
  }
}

void RowParallelQLinearGPTQMarlinImpl::verify_loaded_weights(
    const std::string& prefix) const {
  CHECK(qweight_is_loaded_)
      << "qweight is not loaded for " << prefix + "qweight";
  CHECK(qzeros_is_loaded_) << "qzeros is not loaded for " << prefix + "qzeros";
  CHECK(scales_is_loaded_) << "scales is not loaded for " << prefix + "scales";
  CHECK(!bias_.defined() || bias_is_loaded_)
      << "bias is not loaded for " << prefix + "bias";
}

torch::Tensor RowParallelQLinearGPTQMarlinImpl::forward(
    torch::Tensor input) const {
  return input;
}

}  // namespace llm
