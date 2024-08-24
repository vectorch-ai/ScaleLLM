#include "qlinear_awq_marlin_impl.h"

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <cstdint>

#include "kernels/quantization/marlin/marlin.h"
#include "layers/weight_utils.h"
#include "model_loader/state_dict.h"
#include "model_parallel/model_parallel.h"
#include "pack_utils.h"

namespace llm {
namespace {
int64_t round_up(int64_t num, int64_t multiple) {
  return ((num + multiple - 1) / multiple);
}

void check_awq_quant_args(const QuantArgs& quant_args) {
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

// clang-format off
const std::vector<int64_t> kInterleavingBits4 = {
    0, 2, 4, 6, 1, 3, 5, 7,
};
const std::vector<int64_t> kInterleavingBits8 = {
    0, 2, 1, 3,
};

// argsort([0, 2, 4, 6, 1, 3, 5, 7])
const std::vector<int64_t> kUndoInterleavingBits4 = {
    0, 4, 1, 5, 2, 6, 3, 7,
};
// argsort([0, 2, 1, 3])
const std::vector<int64_t> kUndoInterleavingBits8 = {
    0, 2, 1, 3,
};
// clang-format on

torch::Tensor repack_qzeros(const torch::Tensor& qzeros, int64_t num_bits) {
  const int64_t n_groups = qzeros.size(0);
  // unpack qzeros:
  // (n_groups, out_features/pack_factor) -> (n_groups, out_features)
  auto unpacked_qzeros = pack_utils::unpack_cols(qzeros, num_bits);

  // undo interleaving
  const auto& undo_interleaving =
      num_bits == 4 ? kUndoInterleavingBits4 : kUndoInterleavingBits8;
  const int64_t undo_interleaving_len = undo_interleaving.size();
  unpacked_qzeros = unpacked_qzeros.reshape({-1, undo_interleaving_len});
  unpacked_qzeros = unpacked_qzeros.index_select(
      /*dim=*/1, torch::tensor(undo_interleaving, qzeros.device()));
  unpacked_qzeros =
      unpacked_qzeros.ravel().reshape({n_groups, -1}).contiguous();

  // pack qzeros to marlin compatible format
  // permute qzeros in the same way as scales
  const int64_t perm_len = kScalesPerm.size();
  auto marlin_qzeros =
      unpacked_qzeros.reshape({-1, perm_len})
          .index_select(/*dim=*/1, torch::tensor(kScalesPerm, qzeros.device()));

  // interleaving columns
  const auto& interleaving =
      num_bits == 4 ? kInterleavingBits4 : kInterleavingBits8;
  const int64_t interleaving_len = interleaving.size();
  marlin_qzeros =
      marlin_qzeros.reshape({-1, interleaving_len})
          .index_select(/*dim=*/1,
                        torch::tensor(interleaving, qzeros.device()));
  marlin_qzeros = marlin_qzeros.reshape(unpacked_qzeros.sizes()).contiguous();
  // pack qzeros on columns
  auto packed_marlin_qzeros = pack_utils::pack_cols(marlin_qzeros, num_bits);
  return packed_marlin_qzeros.to(qzeros);
}

void repack_weight(torch::Tensor& qweight,
                   torch::Tensor& qzeros,
                   torch::Tensor& scales,
                   int64_t num_bits) {
  // permute and repack qweight to marlin compatible format
  auto marlin_qweights = torch::empty(
      {qweight.size(0) / 16, qweight.size(1) * 16}, qweight.options());
  marlin::awq_repack(qweight, marlin_qweights, num_bits);
  qweight.set_data(marlin_qweights);

  // permute and repack qzeros
  auto marlin_qzeros = repack_qzeros(qzeros, num_bits);
  qzeros.set_data(marlin_qzeros);

  // permute scales
  const int64_t n_groups = scales.size(0);
  const auto& scale_perm = n_groups == 1 ? kScalesPermSingle : kScalesPerm;
  const int64_t perm_len = scale_perm.size();
  auto marlin_scales =
      scales.reshape({-1, perm_len})
          .index_select(/*dim=*/1, torch::tensor(scale_perm, scales.device()));
  marlin_scales = marlin_scales.reshape(scales.sizes()).contiguous();
  scales.set_data(marlin_scales);
}

}  // namespace

ColumnParallelQLinearAWQMarlinImpl::ColumnParallelQLinearAWQMarlinImpl(
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
  check_awq_quant_args(quant_args);

  const int64_t world_size = parallel_args.world_size();
  CHECK(out_features % world_size == 0)
      << "out_features " << out_features << " not divisible by world_size "
      << world_size;
  const int64_t out_features_per_partition = out_features / world_size;
  const int64_t pack_factor = 32 / bits_;

  // verify shapes
  CHECK(out_features_per_partition % 64 == 0);
  CHECK(in_features % 128 == 0);

  // awq pack weight on dim 1
  CHECK(out_features_per_partition % pack_factor == 0)
      << "out_features_per_partition " << out_features_per_partition
      << " not divisible by pack_factor " << pack_factor;

  qweight_ =
      torch::empty({in_features, out_features_per_partition / pack_factor},
                   options.dtype(torch::kInt32));

  int64_t n_groups = 1;
  if (quant_args.group_size() > 0) {
    CHECK(in_features % quant_args.group_size() == 0);
    n_groups = in_features / quant_args.group_size();
  }
  qzeros_ = torch::empty({n_groups, out_features_per_partition / pack_factor},
                         options.dtype(torch::kInt32));

  scales_ = torch::empty({n_groups, out_features_per_partition}, options);

  if (bias) {
    bias_ = torch::empty({out_features_per_partition}, options);
  }

  const int64_t max_workspace_size = out_features_per_partition / 64 * 16;
  workspace_ = torch::zeros({max_workspace_size}, options.dtype(torch::kInt32));

  g_idx_ = torch::empty({0}, options.dtype(torch::kInt32));
  perm_ = torch::empty({0}, options.dtype(torch::kInt32));
}

// load the weight from the checkpoint
void ColumnParallelQLinearAWQMarlinImpl::load_state_dict(
    const StateDict& state_dict) {
  const auto rank = parallel_args_.rank();
  const auto world_size = parallel_args_.world_size();

  // load sharded weights on dim 1
  LOAD_SHARDED_WEIGHT(qweight, 1);
  LOAD_SHARDED_WEIGHT(qzeros, 1);
  LOAD_SHARDED_WEIGHT(scales, 1);

  // load bias if defined
  if (bias_.defined()) {
    // load sharded bias on dim 0
    LOAD_SHARDED_WEIGHT(bias, 0);
  }
}

// special load_state_dict for fused cases
void ColumnParallelQLinearAWQMarlinImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  const auto rank = parallel_args_.rank();
  const auto world_size = parallel_args_.world_size();

  // load and merge weights on dim 1
  LOAD_FUSED_WEIGHT(qweight, 1);
  LOAD_FUSED_WEIGHT(qzeros, 1);
  LOAD_FUSED_WEIGHT(scales, 1);

  // load bias if defined
  if (bias_.defined()) {
    // load and merge bias on dim 0
    LOAD_FUSED_WEIGHT(bias, 0);
  }
}

void ColumnParallelQLinearAWQMarlinImpl::verify_loaded_weights(
    const std::string& prefix) const {
  CHECK(qweight_is_loaded_)
      << "qweight is not loaded for " << prefix + "qweight";
  CHECK(qzeros_is_loaded_) << "qzeros is not loaded for " << prefix + "qzeros";
  CHECK(scales_is_loaded_) << "scales is not loaded for " << prefix + "scales";
  CHECK(!bias_.defined() || bias_is_loaded_)
      << "bias is not loaded for " << prefix + "bias";
}

torch::Tensor ColumnParallelQLinearAWQMarlinImpl::forward(torch::Tensor input) {
  // repack qweight and scales to marlin compatible format at the first call
  if (!weight_repacked_) {
    repack_weight(qweight_, qzeros_, scales_, bits_);
    weight_repacked_ = true;
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
                    /*has_zp=*/true,
                    /*use_fp32_reduce=*/true);
  if (bias_.defined()) {
    output.add_(bias_);
  }
  if (parallel_args_.world_size() > 1 && gather_output_) {
    output = gather_from_model_parallel_region(output, parallel_args_);
  }
  return output;
}

// RowParallelQLinearAWQMarlinImpl
RowParallelQLinearAWQMarlinImpl::RowParallelQLinearAWQMarlinImpl(
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
  check_awq_quant_args(quant_args);

  const int64_t world_size = parallel_args.world_size();
  CHECK(in_features % world_size == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size;
  const int64_t in_features_per_partition = in_features / world_size;
  const int64_t pack_factor = 32 / bits_;

  qweight_ =
      torch::empty({in_features_per_partition, out_features / pack_factor},
                   options.dtype(torch::kInt32));

  // load full scales for act_order and channelwise quant
  int64_t n_groups = 1;
  if (quant_args.group_size() > 0) {
    CHECK(in_features_per_partition % quant_args.group_size() == 0);
    n_groups = in_features_per_partition / quant_args.group_size();
  }
  qzeros_ = torch::empty({n_groups, out_features / pack_factor},
                         options.dtype(torch::kInt32));
  scales_ = torch::empty({n_groups, out_features}, options);

  if (bias) {
    bias_ = torch::empty({out_features}, options);
  }

  const int64_t max_workspace_size = out_features / 64 * 16;
  workspace_ = torch::zeros({max_workspace_size}, options);

  g_idx_ = torch::empty({0}, options.dtype(torch::kInt32));
  perm_ = torch::empty({0}, options.dtype(torch::kInt32));
}

// load the weight from the checkpoint
void RowParallelQLinearAWQMarlinImpl::load_state_dict(
    const StateDict& state_dict) {
  const auto rank = parallel_args_.rank();
  const auto world_size = parallel_args_.world_size();

  // load sharded weights on dim 0
  LOAD_SHARDED_WEIGHT(qweight, 0);
  LOAD_SHARDED_WEIGHT(qzeros, 0);
  LOAD_SHARDED_WEIGHT(scales, 0);

  if (bias_.defined()) {
    // load bias
    LOAD_WEIGHT(bias);
  }
}

void RowParallelQLinearAWQMarlinImpl::verify_loaded_weights(
    const std::string& prefix) const {
  CHECK(qweight_is_loaded_)
      << "qweight is not loaded for " << prefix + "qweight";
  CHECK(qzeros_is_loaded_) << "qzeros is not loaded for " << prefix + "qzeros";
  CHECK(scales_is_loaded_) << "scales is not loaded for " << prefix + "scales";
  CHECK(!bias_.defined() || bias_is_loaded_)
      << "bias is not loaded for " << prefix + "bias";
}

torch::Tensor RowParallelQLinearAWQMarlinImpl::forward(torch::Tensor input) {
  // repack qweight and scales to marlin compatible format at the first call
  if (!weight_repacked_) {
    repack_weight(qweight_, qzeros_, scales_, bits_);
    weight_repacked_ = true;
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
                    /*is_k_full=*/true,
                    /*has_zp=*/true,
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
