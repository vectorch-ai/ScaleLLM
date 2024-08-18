#include "qlinear_gptq_marlin_impl.h"

#include <glog/logging.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "handlers/llm_handler.h"
#include "kernels/quantization/marlin/marlin.h"
#include "layers/weight_utils.h"
#include "model_loader/state_dict.h"

// void gptq_repack(const torch::Tensor& b_q_weight,  // (k/pack_factor, n)
//                  const torch::Tensor& perm,        // ?
//                  torch::Tensor& out,               // (k/16,
//                  n*16/pack_factor) int64_t num_bits);

namespace llm {
namespace {
int64_t round_up(int64_t num, int64_t multiple) {
  return ((num + multiple - 1) / multiple);
}

const std::vector<int64_t> scale_perm = {
    0, 8,  16, 24, 32, 40, 48, 56, 1, 9,  17, 25, 33, 41, 49, 57,
    2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59,
    4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61,
    6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63};
const std::vector<int64_t> scale_perm_single = {
    0, 1, 8,  9,  16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27,
    4, 5, 12, 13, 20, 21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31,
};

torch::Tensor repack_weight(torch::Tensor& qweight,
                            torch::Tensor& scales,
                            torch::Tensor& g_idx,
                            int64_t num_bits) {
  torch::Tensor perm;
  if (g_idx.defined()) {
    // sort g_idx in ascending order
    perm = torch::argsort(g_idx).to(torch::kInt32);
    auto p_g_idx = g_idx.index_select(/*dim=*/0, perm);
    g_idx.set_data(p_g_idx);
  }

  // permute and repack qweight to marlin compatible format
  auto out = torch::empty_like(qweight);
  marlin::gptq_repack(qweight, perm, out, num_bits);
  qweight.set_data(out);

  // // permute scales
  // const int64_t perm_len = scale_perm.size();
  // auto perm_scale =
  //     scales.reshape({-1, perm_len}).index_select(/*dim=*/1,
  //     perm).contiguous();
  // scales.set_data(perm_scale.view_as(scales));
  return perm;
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

  const int64_t max_workspace_size = out_features_per_partition / 64 * 16;
  workspace_ = torch::zeros({max_workspace_size}, options);
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
    torch::Tensor input) {
  // repack qweight and scales to marlin compatible format at the first call
  if (!perm_.defined()) {
    perm_ = repack_weight(qweight_, scales_, g_idx_, bits_);
  }

  auto output =
      torch::empty({input.size(0), qweight_.size(1)}, input.options());
  marlin::gptq_gemm(input,
                    qweight_,
                    output,
                    scales_,
                    qzeros_,
                    g_idx_,
                    perm_,
                    workspace_,
                    bits_,
                    /*is_k_full=*/true,
                    /*has_zp=*/false,
                    /*use_fp32_reduce=*/true);
  if (bias_.defined()) {
    output.add_(bias_);
  }
  if (parallel_args_.world_size() > 1 && gather_output_) {
    output = gather_from_model_parallel_region(output, parallel_args_);
  }
  return output;
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

  const int64_t max_workspace_size = out_features / 64 * 16;
  workspace_ = torch::zeros({max_workspace_size}, options);
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

torch::Tensor RowParallelQLinearGPTQMarlinImpl::forward(torch::Tensor input) {
  // repack qweight and scales to marlin compatible format at the first call
  if (!perm_.defined()) {
    perm_ = repack_weight(qweight_, scales_, g_idx_, bits_);
  }

  if (!input_is_parallelized_) {
    input = scatter_to_model_parallel_region(input, parallel_args_);
  }

  auto output =
      torch::empty({input.size(0), qweight_.size(1)}, input.options());
  marlin::gptq_gemm(input,
                    qweight_,
                    output,
                    scales_,
                    qzeros_,
                    g_idx_,
                    perm_,
                    workspace_,
                    bits_,
                    /*is_k_full=*/!act_order_,
                    /*has_zp=*/false,
                    /*use_fp32_reduce=*/true);

  if (parallel_args_.world_size() > 1) {
    output = reduce_from_model_parallel_region(output, parallel_args_);
  }
  // N.B. need to apply bias after the reduce
  if (bias_.defined()) {
    output.add_(bias_);
  }
  return output;
}

}  // namespace llm
