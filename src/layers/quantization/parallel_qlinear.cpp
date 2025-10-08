#include "parallel_qlinear.h"

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "model_loader/state_dict.h"

namespace llm {
namespace {
int64_t round_up(int64_t num, int64_t multiple) {
  return ((num + multiple - 1) / multiple);
}

}  // namespace

namespace detail {
// construct weights matrix for gptq from quantized weights
// return the weights matrix [in_features, out_features] with following formula:
// weights = scales * (qweights - qzeros)
//    pack_factor = 32 / bits
//    n_in_ints = in_features / pack_factor
//    n_out_ints = out_features / pack_factor
//    n_groups = ceil(in_features / group_size)
torch::Tensor construct_weights(
    const torch::Tensor& qweights,  // [n_in_ints, out_features] IntTensor
    const torch::Tensor& qzeros,    // [n_groups, n_out_ints] IntTensor
    const torch::Tensor& scales,    // [n_groups, out_features] HalfTensor
    const torch::Tensor& g_idx,     // [in_features] IntTensor
    int64_t bits) {
  CHECK(bits == 2 || bits == 4 || bits == 8) << "Only 2,4,8 bits are supported";
  const int64_t pack_factor = 32 / bits;
  const int64_t n_groups = scales.size(0);
  const int64_t out_features = scales.size(1);

  std::vector<int32_t> bits_to_shift;
  for (int32_t i = 0; i < 32; i += bits) {
    bits_to_shift.push_back(i);
  }
  // [1, 32/bits]
  const auto shift_bits =
      torch::tensor(bits_to_shift, qweights.options()).unsqueeze(0);
  const auto dtype = (bits == 8) ? torch::kInt16 : torch::kInt8;
  const uint16_t mask = static_cast<uint16_t>(std::pow(2, bits) - 1);
  // [n_groups, out_features/pack_factor, pack_factor]
  auto zeros = torch::bitwise_right_shift(
                   qzeros.unsqueeze(2).expand({-1, -1, pack_factor}),
                   shift_bits.unsqueeze(0))
                   .to(dtype);
  zeros.bitwise_and_(mask);
  zeros.add_(1);
  // [n_groups, out_features]
  zeros = zeros.reshape(scales.sizes());

  // [in_features/pack_factor, pack_factor, out_features]
  auto weights = torch::bitwise_right_shift(
                     qweights.unsqueeze(1).expand({-1, pack_factor, -1}),
                     shift_bits.unsqueeze(-1))
                     .to(dtype);
  weights.bitwise_and_(mask);
  // [in_features, out_features]
  weights = weights.reshape({-1, out_features});
  // auto gathered_scales = scales.gather(/*dim=*/0, /*index=*/g_idx);
  // auto gathered_zeros = zeros.gather(/*dim=*/0, /*index=*/g_idx);
  // return gathered_scales * (weights - gathered_zeros);
  return scales.index({g_idx}) * (weights - zeros.index({g_idx}));
}

// construct weights matrix for gptq from quantized weights without using g_idx
// slower than construct_weights with g_idx
// return the weights matrix [in_features, out_features] with following formula:
// weights = scales * (qweights - qzeros)
torch::Tensor construct_weights(
    const torch::Tensor& qweights,  // [n_in_ints, out_features] IntTensor
    const torch::Tensor& qzeros,    // [n_groups, n_out_ints] IntTensor
    const torch::Tensor& scales,    // [n_groups, out_features] HalfTensor
    int64_t bits) {
  CHECK(bits == 2 || bits == 4 || bits == 8) << "Only 2,4,8 bits are supported";
  const int64_t pack_factor = 32 / bits;
  const int64_t n_groups = scales.size(0);
  const int64_t out_features = scales.size(1);

  std::vector<int32_t> bits_to_shift;
  for (int32_t i = 0; i < 32; i += bits) {
    bits_to_shift.push_back(i);
  }

  // [1, n_ints=32/bits]
  const auto shift_bits =
      torch::tensor(bits_to_shift, qweights.options()).unsqueeze(0);
  const auto dtype = (bits == 8) ? torch::kInt16 : torch::kInt8;
  const uint16_t mask = static_cast<uint16_t>(std::pow(2, bits) - 1);
  // [n_groups, out_features/pack_factor, pack_factor]
  auto zeros = torch::bitwise_right_shift(
                   qzeros.unsqueeze(2).expand({-1, -1, pack_factor}),
                   shift_bits.unsqueeze(0))
                   .to(dtype);
  zeros.bitwise_and_(mask);
  zeros.add_(1);
  // [n_groups, 1, out_features]
  zeros = zeros.reshape({n_groups, 1, out_features});

  // [in_features/pack_factor, pack_factor, out_features]
  auto weights = torch::bitwise_right_shift(
                     qweights.unsqueeze(1).expand({-1, pack_factor, -1}),
                     shift_bits.unsqueeze(-1))
                     .to(dtype);
  weights.bitwise_and_(mask);
  // [in_features, out_features] => [n_groups, group_size, out_features]
  weights = weights.reshape({n_groups, -1, out_features});
  // [n_groups, 1, out_features] * [n_groups, group_size, out_features]
  weights = scales.unsqueeze(1) * (weights - zeros);
  // [n_groups, group_size, out_features] => [in_features, out_features]
  return weights.reshape({-1, out_features});
}
}  // namespace detail

ColumnParallelQLinearImpl::ColumnParallelQLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantArgs& quant_args,
    int64_t qweight_pack_dim,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : bits_(quant_args.bits()),
      gather_output_(gather_output),
      parallel_args_(parallel_args) {
  const auto bits = quant_args.bits();
  const auto group_size =
      quant_args.group_size() > 0 ? quant_args.group_size() : in_features;
  CHECK(qweight_pack_dim == 0 || qweight_pack_dim == 1)
      << "qweight_pack_dim must be 0 or 1";
  const auto rank = parallel_args_.rank();
  const auto world_size = parallel_args_.world_size();
  CHECK(out_features % world_size == 0)
      << "out_features " << out_features << " not divisible by world_size "
      << world_size;
  const int64_t out_features_per_partition = out_features / world_size;
  const int64_t pack_factor = 32 / bits;

  if (qweight_pack_dim == 0) {
    qweight_ = register_sharded_parameter(
        "qweight",
        /*dim=*/1,
        rank,
        world_size,
        torch::empty({in_features / pack_factor, out_features_per_partition},
                     options.dtype(torch::kInt32)));
  } else {
    qweight_ = register_sharded_parameter(
        "qweight",
        /*dim=*/1,
        rank,
        world_size,
        torch::empty({in_features, out_features_per_partition / pack_factor},
                     options.dtype(torch::kInt32)));
  }
  const int64_t n_groups = round_up(in_features, group_size);
  qzeros_ = register_sharded_parameter(
      "qzeros",
      /*dim=*/1,
      rank,
      world_size,
      torch::empty({n_groups, out_features_per_partition / pack_factor},
                   options.dtype(torch::kInt32)));

  scales_ = register_sharded_parameter(
      "scales",
      /*dim=*/1,
      rank,
      world_size,
      torch::empty({n_groups, out_features_per_partition}, options));

  if (bias) {
    bias_ = register_sharded_parameter(
        "bias",
        /*dim=*/0,
        rank,
        world_size,
        torch::empty({out_features_per_partition}, options));
  }
}

torch::Tensor ColumnParallelQLinearImpl::quant_matmul(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& qzeros,
    const torch::Tensor& scales) const {
  const int64_t out_features = qweight.size(-1);
  // scales * (qweights - qzeros): [in_features, out_features]
  const auto weights =
      detail::construct_weights(qweight, qzeros, scales, bits_);
  // output: [batch, out_features]
  return torch::matmul(input, weights);
}

RowParallelQLinearImpl::RowParallelQLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantArgs& quant_args,
    int64_t qweight_pack_dim,
    bool input_is_parallelized,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : bits_(quant_args.bits()),
      input_is_parallelized_(input_is_parallelized),
      parallel_args_(parallel_args) {
  const auto bits = quant_args.bits();
  CHECK(qweight_pack_dim == 0 || qweight_pack_dim == 1)
      << "qweight_pack_dim must be 0 or 1";
  const auto rank = parallel_args_.rank();
  const auto world_size = parallel_args_.world_size();
  CHECK(in_features % world_size == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size;
  const int64_t in_features_per_partition = in_features / world_size;
  const int64_t pack_factor = 32 / bits;
  const auto group_size =
      quant_args.group_size() > 0 ? quant_args.group_size() : in_features;

  if (qweight_pack_dim == 0) {
    qweight_ = register_sharded_parameter(
        "qweight",
        /*dim=*/0,
        rank,
        world_size,
        torch::empty({in_features_per_partition / pack_factor, out_features},
                     options.dtype(torch::kInt32)));
  } else {
    qweight_ = register_sharded_parameter(
        "qweight",
        /*dim=*/0,
        rank,
        world_size,
        torch::empty({in_features_per_partition, out_features / pack_factor},
                     options.dtype(torch::kInt32)));
  }
  qzeros_ = register_sharded_parameter(
      "qzeros",
      /*dim=*/0,
      rank,
      world_size,
      torch::empty({round_up(in_features_per_partition, group_size),
                    out_features / pack_factor},
                   options.dtype(torch::kInt32)));

  scales_ = register_sharded_parameter(
      "scales",
      /*dim=*/0,
      rank,
      world_size,
      torch::empty(
          {round_up(in_features_per_partition, group_size), out_features},
          options));

  if (bias) {
    bias_ = register_parameter("bias", torch::empty({out_features}, options));
  }
}

torch::Tensor RowParallelQLinearImpl::quant_matmul(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& qzeros,
    const torch::Tensor& scales) const {
  const int64_t out_features = qweight.size(-1);
  // scales * (qweights - qzeros): [in_features, out_features]
  const auto weights =
      detail::construct_weights(qweight, qzeros, scales, bits_);
  // output: [batch, out_features]
  return torch::matmul(input, weights);
}

}  // namespace llm
