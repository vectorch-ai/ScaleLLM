#pragma once

#include <c10/core/TensorImpl.h>
#include <torch/torch.h>

#include "model_loader/state_dict.h"
#include "models/model_args.h"
#include "qlinear_impl.h"

namespace llm {

// quantized linear layers using gptq
using VecQuantMatmulFunc = void (*)(torch::Tensor vec,
                                    torch::Tensor mat,
                                    torch::Tensor mul,
                                    torch::Tensor scales,
                                    torch::Tensor zeros,
                                    torch::Tensor g_idx,
                                    int64_t bits);

// Quantized Linear layer with column parallelism.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class ColumnParallelQLinearGPTQImpl : public ColumnParallelQLinearImpl {
 public:
  ColumnParallelQLinearGPTQImpl(int64_t in_features,
                                int64_t out_features,
                                bool bias,
                                const QuantArgs& quant_args,
                                bool gather_output,
                                const ParallelArgs& parallel_args,
                                torch::ScalarType dtype,
                                const torch::Device& device);

  ~ColumnParallelQLinearGPTQImpl() override;

  torch::Tensor quant_matmul(const torch::Tensor& input,
                             const torch::Tensor& qweight,
                             const torch::Tensor& qzeros,
                             const torch::Tensor& scales) const override;

 private:
  // parameter members, must be registered
  torch::Tensor g_idx_{nullptr};

  // quantization parameters
  int64_t bits_ = 0;

  VecQuantMatmulFunc vec_quant_matmul_func_ = nullptr;
};

// Linear layer with row parallelism.
//     The linear layer is defined as Y = XA + b. A is parallelized along
//     its first dimension and X along its second dimension as:
//                -   -
//               | A_1 |
//               | .   |
//           A = | .   |       X = [X_1, ..., X_p]
//               | .   |
//               | A_p |
//                -   -
class RowParallelQLinearGPTQImpl : public RowParallelQLinearImpl {
 public:
  RowParallelQLinearGPTQImpl(int64_t in_features,
                             int64_t out_features,
                             bool bias,
                             const QuantArgs& quant_args,
                             bool input_is_parallelized,
                             const ParallelArgs& parallel_args,
                             torch::ScalarType dtype,
                             const torch::Device& device);

  ~RowParallelQLinearGPTQImpl() override;

  torch::Tensor quant_matmul(const torch::Tensor& input,
                             const torch::Tensor& qweight,
                             const torch::Tensor& qzeros,
                             const torch::Tensor& scales) const override;

 private:
  // parameter members, must be registered
  torch::Tensor g_idx_{nullptr};

  // quantization parameters
  int64_t bits_ = 0;

  VecQuantMatmulFunc vec_quant_matmul_func_ = nullptr;
};
}  // namespace llm
