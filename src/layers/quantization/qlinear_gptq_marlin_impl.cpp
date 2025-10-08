#include "qlinear_gptq_marlin_impl.h"

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "kernels/quantization/marlin.h"
#include "layers/linear/weight_utils.h"
#include "model_loader/state_dict.h"
#include "model_parallel/model_parallel.h"

namespace llm {
namespace {
int64_t round_up(int64_t num, int64_t multiple) {
  return ((num + multiple - 1) / multiple);
}

void check_gptq_quant_args(const QuantArgs& quant_args) {
  CHECK(quant_args.is_sym())
      << "Only symmetric quantization is supported for GPTQ";

  const auto bits = quant_args.bits();
  CHECK(bits == 4 || bits == 8) << "Only 4 and 8 bits are supported for GPTQ";

  const auto group_size = quant_args.group_size();
  CHECK(group_size == -1 || group_size == 32 || group_size == 64 ||
        group_size == 128)
      << "Only group_size of -1, 32, 64, 128 are supported for GPTQ";
}

const std::vector<int64_t> kScalesPerm = {
    0, 8,  16, 24, 32, 40, 48, 56, 1, 9,  17, 25, 33, 41, 49, 57,
    2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59,
    4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61,
    6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63};
const std::vector<int64_t> kScalesPermSingle = {
    0, 1, 8,  9,  16, 17, 24, 25, 2, 3, 10, 11, 18, 19, 26, 27,
    4, 5, 12, 13, 20, 21, 28, 29, 6, 7, 14, 15, 22, 23, 30, 31,
};

torch::Tensor repack_weight(torch::Tensor& qweight,
                            torch::Tensor& scales,
                            torch::Tensor& g_idx,
                            int64_t num_bits,
                            bool act_order) {
  torch::Tensor perm;
  if (act_order) {
    // sort g_idx in ascending order
    perm = torch::argsort(g_idx).to(torch::kInt32);
    auto p_g_idx = g_idx.index_select(/*dim=*/0, perm);
    g_idx.set_data(p_g_idx);
  } else {
    perm = torch::empty({0}, g_idx.options());
  }

  // permute and repack qweight to marlin compatible format
  auto marlin_qweights = torch::empty_like(qweight);
  marlin::gptq_repack(qweight, perm, marlin_qweights, num_bits);
  qweight.set_data(marlin_qweights);

  // permute scales
  const int64_t n_groups = scales.size(0);
  const auto& scale_perm = n_groups == 1 ? kScalesPermSingle : kScalesPerm;
  const int64_t perm_len = scale_perm.size();
  auto marlin_scales =
      scales.reshape({-1, perm_len})
          .index_select(/*dim=*/1, torch::tensor(scale_perm, scales.device()));
  marlin_scales = marlin_scales.reshape(scales.sizes()).contiguous();
  scales.set_data(marlin_scales);
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
      act_order_(quant_args.desc_act()),
      gather_output_(gather_output),
      parallel_args_(parallel_args) {
  check_gptq_quant_args(quant_args);

  const int64_t world_size = parallel_args.world_size();
  CHECK(out_features % world_size == 0)
      << "out_features " << out_features << " not divisible by world_size "
      << world_size;
  const int64_t out_features_per_partition = out_features / world_size;
  const int64_t pack_factor = 32 / bits_;

  // gptq pack weight on dim 0
  qweight_ =
      torch::empty({in_features / pack_factor, out_features_per_partition},
                   options.dtype(torch::kInt32));

  CHECK(out_features_per_partition % pack_factor == 0)
      << "out_features_per_partition " << out_features_per_partition
      << " not divisible by pack_factor " << pack_factor;

  int64_t n_groups = 1;
  if (quant_args.group_size() > 0) {
    n_groups = round_up(in_features, quant_args.group_size());
  }
  scales_ = torch::empty({n_groups, out_features_per_partition}, options);
  qzeros_ = torch::empty({0}, options);

  if (act_order_) {
    g_idx_ = torch::empty({in_features}, options.dtype(torch::kInt32));
  } else {
    g_idx_ = torch::empty({0}, options.dtype(torch::kInt32));
  }

  if (bias) {
    bias_ = torch::empty({out_features_per_partition}, options);
  }

  const int64_t max_workspace_size = out_features_per_partition / 64 * 16;
  workspace_ = torch::zeros({max_workspace_size}, options.dtype(torch::kInt32));
}

torch::Tensor ColumnParallelQLinearGPTQMarlinImpl::forward(
    torch::Tensor input) {
  // repack qweight and scales to marlin compatible format at the first call
  if (!perm_.defined()) {
    perm_ = repack_weight(qweight_, scales_, g_idx_, bits_, act_order_);
    CHECK(perm_.defined());
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
      act_order_(quant_args.desc_act()),
      input_is_parallelized_(input_is_parallelized),
      parallel_args_(parallel_args) {
  check_gptq_quant_args(quant_args);

  const int64_t world_size = parallel_args.world_size();
  CHECK(in_features % world_size == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size;
  const int64_t in_features_per_partition = in_features / world_size;
  const int64_t pack_factor = 32 / bits_;

  qweight_ =
      torch::empty({in_features_per_partition / pack_factor, out_features},
                   options.dtype(torch::kInt32));

  // load full scales for act_order and channelwise quant
  load_full_scales_ = act_order_ || quant_args.group_size() == -1;
  int64_t n_groups = 1;
  if (quant_args.group_size() > 0) {
    n_groups =
        round_up(load_full_scales_ ? in_features : in_features_per_partition,
                 quant_args.group_size());
  }
  scales_ = torch::empty({n_groups, out_features}, options);
  qzeros_ = torch::empty({0}, options);

  if (act_order_) {
    // load sharded g_idx on dim 0
    g_idx_ =
        torch::empty({in_features_per_partition}, options.dtype(torch::kInt32));
  } else {
    g_idx_ = torch::empty({0}, options.dtype(torch::kInt32));
  }

  if (bias) {
    bias_ = torch::empty({out_features}, options);
  }

  const int64_t max_workspace_size = out_features / 64 * 16;
  workspace_ = torch::zeros({max_workspace_size}, options.dtype(torch::kInt32));
}

torch::Tensor RowParallelQLinearGPTQMarlinImpl::forward(torch::Tensor input) {
  // repack qweight and scales to marlin compatible format at the first call
  if (!perm_.defined()) {
    perm_ = repack_weight(qweight_, scales_, g_idx_, bits_, act_order_);
    CHECK(perm_.defined());
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
