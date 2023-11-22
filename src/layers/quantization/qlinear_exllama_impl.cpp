#include "qlinear_exllama_impl.h"

#include <c10/core/DeviceType.h>
#include <gflags/gflags.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "../model_parallel.h"
#include "common/logging.h"
#include "model_loader/state_dict.h"
#include "models/args.h"

// Create Q4Matrix, return handle
extern uintptr_t make_q4(torch::Tensor qweight,
                         torch::Tensor qzeros,
                         torch::Tensor scales,
                         torch::Tensor g_idx,
                         int device);

// Matmul half @ quant -> half
extern void q4_matmul(torch::Tensor x, uintptr_t w, torch::Tensor out);

// Free Q4Matrix handle
extern void free_q4(uintptr_t w);

namespace llm {

ColumnParallelQLinearExllamaImpl::ColumnParallelQLinearExllamaImpl(
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
                                device) {
  const auto bits = quant_args.bits();
  GCHECK(bits == 4) << "Only 4 bits are supported";
  const auto group_size =
      quant_args.group_size() > 0 ? quant_args.group_size() : in_features;

  // using torch::aarange to create g_idx
  std::vector<int32_t> g_idx_data;
  g_idx_data.reserve(in_features);
  for (int32_t i = 0; i < in_features; ++i) {
    g_idx_data.push_back(i / group_size);
  }
  g_idx_ = register_buffer(
      "g_idx",
      torch::tensor(g_idx_data, torch::dtype(torch::kInt32).device(device)));
}

ColumnParallelQLinearExllamaImpl::~ColumnParallelQLinearExllamaImpl() {
  if (q4_ != 0) {
    free_q4(q4_);
  }
}

torch::Tensor ColumnParallelQLinearExllamaImpl::quant_matmul(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& qzeros,
    const torch::Tensor& scales) const {
  // lazy initialization
  if (q4_ == 0) {
    auto none_tensor = torch::empty({1, 1}, torch::kMeta);
    q4_ =
        make_q4(qweight, qzeros, scales, none_tensor, qweight.device().index());
  }
  const int64_t out_features = qweight.size(-1);
  auto output = torch::zeros({input.size(0), out_features}, input.options());
  q4_matmul(input, q4_, output);
  return output;
}

RowParallelQLinearExllamaImpl::RowParallelQLinearExllamaImpl(
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
                             device) {
  const auto bits = quant_args.bits();
  GCHECK(bits == 4) << "Only 4 bits are supported";
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
}

RowParallelQLinearExllamaImpl::~RowParallelQLinearExllamaImpl() {
  if (q4_ != 0) {
    free_q4(q4_);
  }
}

torch::Tensor RowParallelQLinearExllamaImpl::quant_matmul(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& qzeros,
    const torch::Tensor& scales) const {
  // lazy initialization
  if (q4_ == 0) {
    auto none_tensor = torch::empty({1, 1}, torch::kMeta);
    q4_ =
        make_q4(qweight, qzeros, scales, none_tensor, qweight.device().index());
  }
  const int64_t out_features = qweight.size(-1);
  auto output = torch::zeros({input.size(0), out_features}, input.options());
  q4_matmul(input, q4_, output);
  return output;
}

}  // namespace llm
