#include "qlinear_gptq_impl.h"

#include <c10/core/DeviceType.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "../model_parallel.h"
#include "model_loader/state_dict.h"
#include "models/args.h"

DEFINE_string(qlinear_gptq_impl,
              "",
              "type of qlinear gptq impl, slow, cuda, or empty for auto");

extern void vecquant2matmul_cuda(torch::Tensor vec,
                                 torch::Tensor mat,
                                 torch::Tensor mul,
                                 torch::Tensor scales,
                                 torch::Tensor zeros,
                                 torch::Tensor g_idx);

extern void vecquant3matmul_cuda(torch::Tensor vec,
                                 torch::Tensor mat,
                                 torch::Tensor mul,
                                 torch::Tensor scales,
                                 torch::Tensor zeros,
                                 torch::Tensor g_idx);

extern void vecquant4matmul_cuda(torch::Tensor vec,
                                 torch::Tensor mat,
                                 torch::Tensor mul,
                                 torch::Tensor scales,
                                 torch::Tensor zeros,
                                 torch::Tensor g_idx);

extern void vecquant8matmul_cuda(torch::Tensor vec,
                                 torch::Tensor mat,
                                 torch::Tensor mul,
                                 torch::Tensor scales,
                                 torch::Tensor zeros,
                                 torch::Tensor g_idx);

// Create Q4Matrix, return handle
extern uintptr_t make_q4(torch::Tensor qweight,
                         torch::Tensor qzeros,
                         torch::Tensor scales,
                         torch::Tensor g_idx,
                         int device);

// Matmul half @ quant -> half
extern void q4_matmul(torch::Tensor x, uintptr_t w, torch::Tensor out);

namespace llm {
namespace {
void vec_quant_matmul_cuda(torch::Tensor vec,
                           torch::Tensor mat,
                           torch::Tensor mul,
                           torch::Tensor scales,
                           torch::Tensor zeros,
                           torch::Tensor g_idx,
                           int64_t bits) {
  switch (bits) {
    case 2:
      return vecquant2matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
    case 3:
      return vecquant3matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
    case 4:
      return vecquant4matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
    case 8:
      return vecquant8matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
    default:
      LOG(FATAL) << "Unsupported bits " << bits;
  }
  __builtin_unreachable();
}

}  // namespace

namespace details {
// construct weights matrix for gptq from quantized weights
// return the weights matrix [in_features, out_features] with following formula:
// weights = scales * (qweights - qzeros)
torch::Tensor construct_weights(
    const torch::Tensor& qweights,  // [n_ints, out_features] IntTensor
    const torch::Tensor& qzeros,    // [n_groups, n_ints] IntTensor
    const torch::Tensor& scales,    // [n_groups, out_features] HalfTensor
    const torch::Tensor& g_idx,     // [in_features] IntTensor
    int64_t bits) {
  CHECK(bits == 2 || bits == 4 || bits == 8) << "Only 2,4,8 bits are supported";

  std::vector<int32_t> bits_to_shift;
  for (int32_t i = 0; i < 32; i += bits) {
    bits_to_shift.push_back(i);
  }
  // [1, 32/bits]
  const auto shift_bits =
      torch::tensor(bits_to_shift, qweights.options()).unsqueeze(0);
  const auto dtype = (bits == 8) ? torch::kInt16 : torch::kInt8;
  const uint16_t mask = static_cast<uint16_t>(std::pow(2, bits) - 1);
  // [n_groups, out_features/n_bits, n_ints]
  auto zeros = torch::bitwise_right_shift(
                   qzeros.unsqueeze(2).expand({-1, -1, 32 / bits}),
                   shift_bits.unsqueeze(0))
                   .to(dtype);
  zeros.bitwise_and_(mask);
  zeros.add_(1);
  // [n_groups, out_features]
  zeros = zeros.reshape(scales.sizes());

  auto weights = torch::bitwise_right_shift(
                     qweights.unsqueeze(1).expand({-1, 32 / bits, -1}),
                     shift_bits.unsqueeze(-1))
                     .to(dtype);
  weights.bitwise_and_(mask);
  weights = weights.reshape({-1, qweights.size(1)});
  // auto gathered_scales = scales.gather(/*dim=*/0, /*index=*/g_idx);
  // auto gathered_zeros = zeros.gather(/*dim=*/0, /*index=*/g_idx);
  return scales.index({g_idx}) * (weights - zeros.index({g_idx}));
}

// construct weights matrix for gptq from quantized weights without using g_idx
// slower than construct_weights with g_idx
// return the weights matrix [in_features, out_features] with following formula:
// weights = scales * (qweights - qzeros)
torch::Tensor construct_weights(
    const torch::Tensor& qweights,  // [n_ints, out_features] IntTensor
    const torch::Tensor& qzeros,    // [n_groups, n_ints] IntTensor
    const torch::Tensor& scales,    // [n_groups, out_features] HalfTensor
    int64_t bits) {
  CHECK(bits == 2 || bits == 4 || bits == 8) << "Only 2,4,8 bits are supported";

  std::vector<int32_t> bits_to_shift;
  for (int32_t i = 0; i < 32; i += bits) {
    bits_to_shift.push_back(i);
  }

  // [1, n_ints=32/bits]
  const auto shift_bits =
      torch::tensor(bits_to_shift, qweights.options()).unsqueeze(0);
  const auto dtype = (bits == 8) ? torch::kInt16 : torch::kInt8;
  const uint16_t mask = static_cast<uint16_t>(std::pow(2, bits) - 1);
  // [n_groups, out_features/n_bits, n_ints]
  auto zeros = torch::bitwise_right_shift(
                   qzeros.unsqueeze(2).expand({-1, -1, 32 / bits}),
                   shift_bits.unsqueeze(0))
                   .to(dtype);
  zeros.bitwise_and_(mask);
  zeros.add_(1);
  // [n_groups, 1, out_features]
  zeros = zeros.reshape({scales.size(0), 1, scales.size(1)});

  auto weights = torch::bitwise_right_shift(
                     qweights.unsqueeze(1).expand({-1, 32 / bits, -1}),
                     shift_bits.unsqueeze(-1))
                     .to(dtype);
  weights.bitwise_and_(mask);
  // [n_groups, group_size, out_features]
  weights = weights.reshape({scales.size(0), -1, scales.size(1)});
  weights = scales.unsqueeze(1) * (weights - zeros);
  return weights.reshape({-1, scales.size(1)});
}
}  // namespace details

ColumnParallelQLinearGPTQImpl::ColumnParallelQLinearGPTQImpl(
    int64_t in_features,
    int64_t out_features,
    int64_t bits,
    int64_t group_size,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::ScalarType& dtype,
    const torch::Device& device)
    : ColumnParallelQLinearImpl(in_features,
                                out_features,
                                bits,
                                group_size,
                                /*qweight_pack_dim=*/0,
                                parallel_args.rank(),
                                parallel_args.world_size(),
                                dtype,
                                device),
      bits_(bits),
      parallel_args_(parallel_args),
      gather_output_(gather_output) {
  CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 8)
      << "Only 2,3,4,8 bits are supported";
  CHECK(group_size > 0) << "group_size must be positive";

  std::vector<int32_t> g_idx_data;
  g_idx_data.reserve(in_features);
  for (int32_t i = 0; i < in_features; ++i) {
    g_idx_data.push_back(i / group_size);
  }
  g_idx_ = register_buffer(
      "g_idx",
      torch::tensor(g_idx_data, torch::dtype(torch::kInt32).device(device)));
}

torch::Tensor ColumnParallelQLinearGPTQImpl::forward(
    torch::Tensor input) const {
  const int64_t out_features = qweight_.size(-1);
  torch::Tensor output;
  if (FLAGS_qlinear_gptq_impl == "slow") {
    output = torch::zeros({input.size(0), out_features}, input.options());
    const auto weights =
        details::construct_weights(qweight_, qzeros_, scales_, bits_);
    torch::matmul_out(/*out=*/output, /*self=*/input, /*other=*/weights);
  } else if (bits_ == 4) {
    // use exllama for 4 bits, which is faster
    if (q4_ == 0) {
      auto none_tensor = torch::empty({1, 1}, torch::kMeta);
      q4_ = make_q4(
          qweight_, qzeros_, scales_, none_tensor, qweight_.device().index());
    }
    output = torch::zeros({input.size(0), out_features}, input.options());
    q4_matmul(input, q4_, output);
  } else {
    auto input_float = input.to(torch::kFloat32);
    auto scales_float = scales_.to(torch::kFloat32);
    auto output_float = torch::zeros({input_float.size(0), out_features},
                                     input_float.options());

    vec_quant_matmul_cuda(input_float,
                          qweight_,
                          output_float,
                          scales_float,
                          qzeros_,
                          g_idx_,
                          bits_);
    output = output_float.to(input);
  }

  if (parallel_args_.world_size() > 1 && gather_output_) {
    output = gather_from_model_parallel_region(output, parallel_args_);
  }
  return output;
}

RowParallelQLinearGPTQImpl::RowParallelQLinearGPTQImpl(
    int64_t in_features,
    int64_t out_features,
    int64_t bits,
    int64_t group_size,
    bool input_is_parallelized,
    const ParallelArgs& parallel_args,
    const torch::ScalarType& dtype,
    const torch::Device& device)
    : RowParallelQLinearImpl(in_features,
                             out_features,
                             bits,
                             group_size,
                             /*qweight_pack_dim=*/0,
                             parallel_args.rank(),
                             parallel_args.world_size(),
                             dtype,
                             device),
      bits_(bits),
      parallel_args_(parallel_args),
      input_is_parallelized_(input_is_parallelized) {
  CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 8)
      << "Only 2,3,4,8 bits are supported";
  CHECK(group_size > 0) << "group_size must be positive";

  std::vector<int32_t> g_idx_data;
  g_idx_data.reserve(in_features);
  for (int32_t i = 0; i < in_features; ++i) {
    g_idx_data.push_back(i / group_size);
  }
  g_idx_ = register_buffer(
      "g_idx",
      torch::tensor(g_idx_data, torch::dtype(torch::kInt32).device(device)));
}

torch::Tensor RowParallelQLinearGPTQImpl::forward(torch::Tensor input) const {
  if (!input_is_parallelized_) {
    input = scatter_to_model_parallel_region(input, parallel_args_);
  }

  const int64_t out_features = qweight_.size(-1);
  torch::Tensor output;
  if (FLAGS_qlinear_gptq_impl == "slow") {
    output = torch::zeros({input.size(0), out_features}, input.options());
    const auto weights =
        details::construct_weights(qweight_, qzeros_, scales_, bits_);
    torch::matmul_out(/*out=*/output, /*self=*/input, /*other=*/weights);
  } else if (bits_ == 4) {
    // use exllama for 4 bits, which is faster
    if (q4_ == 0) {
      auto none_tensor = torch::empty({1, 1}, torch::kMeta);
      q4_ = make_q4(
          qweight_, qzeros_, scales_, none_tensor, qweight_.device().index());
    }
    output = torch::zeros({input.size(0), out_features}, input.options());
    q4_matmul(input, q4_, output);
  } else {
    auto input_float = input.to(torch::kFloat32);
    auto scales_float = scales_.to(torch::kFloat32);
    auto output_float = torch::zeros({input_float.size(0), out_features},
                                     input_float.options());
    vec_quant_matmul_cuda(input_float,
                          qweight_,
                          output_float,
                          scales_float,
                          qzeros_,
                          g_idx_,
                          bits_);
    output = output_float.to(input);
  }
  if (parallel_args_.world_size() > 1) {
    output = reduce_from_model_parallel_region(output, parallel_args_);
  }
  return output;
}

}  // namespace llm
