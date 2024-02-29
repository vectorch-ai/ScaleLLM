#include "qlinear_gptq_impl.h"

#include <c10/core/DeviceType.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "model_loader/state_dict.h"
#include "model_parallel/model_parallel.h"
#include "models/model_args.h"

namespace llm {

extern void vec_quant_matmul_64(torch::Tensor vec,
                                torch::Tensor mat,
                                torch::Tensor mul,
                                torch::Tensor scales,
                                torch::Tensor zeros,
                                torch::Tensor g_idx,
                                int64_t bits);
extern void vec_quant_matmul_256(torch::Tensor vec,
                                 torch::Tensor mat,
                                 torch::Tensor mul,
                                 torch::Tensor scales,
                                 torch::Tensor zeros,
                                 torch::Tensor g_idx,
                                 int64_t bits);

ColumnParallelQLinearGPTQImpl::ColumnParallelQLinearGPTQImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantArgs& quant_args,
    bool gather_output,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    const torch::Device& device)
    : ColumnParallelQLinearImpl(in_features,
                                out_features,
                                bias,
                                quant_args,
                                /*qweight_pack_dim=*/0,
                                gather_output,
                                parallel_args,
                                dtype,
                                device),
      bits_(quant_args.bits()) {
  const auto bits = quant_args.bits();
  CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 8)
      << "Only 2,3,4,8 bits are supported";

  const auto group_size =
      quant_args.group_size() > 0 ? quant_args.group_size() : in_features;

  std::vector<int32_t> g_idx_data;
  g_idx_data.reserve(in_features);
  for (int32_t i = 0; i < in_features; ++i) {
    g_idx_data.push_back(i / group_size);
  }
  g_idx_ = register_buffer(
      "g_idx",
      torch::tensor(g_idx_data, torch::dtype(torch::kInt32).device(device)));

  vec_quant_matmul_func_ = vec_quant_matmul_256;
  if (in_features % 256 != 0 || out_features % 256 != 0) {
    vec_quant_matmul_func_ = vec_quant_matmul_64;
  }
  if (in_features % 64 != 0 || out_features % 64 != 0) {
    LOG(FATAL) << "in_features and out_features size is not supported: ["
               << in_features << ", " << out_features << "]";
  }
}

ColumnParallelQLinearGPTQImpl::~ColumnParallelQLinearGPTQImpl() {}

torch::Tensor ColumnParallelQLinearGPTQImpl::quant_matmul(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& qzeros,
    const torch::Tensor& scales) const {
  const int64_t out_features = qweight.size(-1);
  // convert to float
  auto input_float = input.to(torch::kFloat32);
  auto scales_float = scales.to(torch::kFloat32);
  auto output_float =
      torch::zeros({input_float.size(0), out_features}, input_float.options());
  vec_quant_matmul_func_(
      input_float, qweight, output_float, scales_float, qzeros, g_idx_, bits_);
  // convert back to input type
  return output_float.to(input);
}

RowParallelQLinearGPTQImpl::RowParallelQLinearGPTQImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantArgs& quant_args,
    bool input_is_parallelized,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    const torch::Device& device)
    : RowParallelQLinearImpl(in_features,
                             out_features,
                             bias,
                             quant_args,
                             /*qweight_pack_dim=*/0,
                             input_is_parallelized,
                             parallel_args,
                             dtype,
                             device),
      bits_(quant_args.bits()) {
  const auto bits = quant_args.bits();
  CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 8)
      << "Only 2,3,4,8 bits are supported";

  const auto group_size =
      quant_args.group_size() > 0 ? quant_args.group_size() : in_features;

  std::vector<int32_t> g_idx_data;
  g_idx_data.reserve(in_features);
  for (int32_t i = 0; i < in_features; ++i) {
    g_idx_data.push_back(i / group_size);
  }
  g_idx_ = register_buffer(
      "g_idx",
      torch::tensor(g_idx_data, torch::dtype(torch::kInt32).device(device)));

  vec_quant_matmul_func_ = vec_quant_matmul_256;
  if (in_features % 256 != 0 || out_features % 256 != 0) {
    vec_quant_matmul_func_ = vec_quant_matmul_64;
  }
  if (in_features % 64 != 0 || out_features % 64 != 0) {
    LOG(FATAL) << "in_features and out_features size is not supported: ["
               << in_features << ", " << out_features << "]";
  }
}

RowParallelQLinearGPTQImpl::~RowParallelQLinearGPTQImpl() {}

torch::Tensor RowParallelQLinearGPTQImpl::quant_matmul(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& qzeros,
    const torch::Tensor& scales) const {
  const int64_t out_features = qweight.size(-1);
  // convert to float
  auto input_float = input.to(torch::kFloat32);
  auto scales_float = scales.to(torch::kFloat32);
  auto output_float =
      torch::zeros({input_float.size(0), out_features}, input_float.options());
  vec_quant_matmul_func_(
      input_float, qweight, output_float, scales_float, qzeros, g_idx_, bits_);
  // convert back to input type
  return output_float.to(input);
}

}  // namespace llm
