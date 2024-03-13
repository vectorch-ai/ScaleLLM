#pragma once

#include <c10/core/TensorImpl.h>
#include <torch/torch.h>

#include "model_loader/state_dict.h"
#include "models/model_args.h"
#include "qlinear_impl.h"

namespace llm {

// Quantized Linear layer with column parallelism.
class ColumnParallelQLinearExllamaImpl : public ColumnParallelQLinearImpl {
 public:
  ColumnParallelQLinearExllamaImpl(int64_t in_features,
                                   int64_t out_features,
                                   bool bias,
                                   const QuantArgs& quant_args,
                                   bool gather_output,
                                   const ParallelArgs& parallel_args,
                                   const torch::TensorOptions& options);

  ~ColumnParallelQLinearExllamaImpl() override;

  torch::Tensor quant_matmul(const torch::Tensor& input,
                             const torch::Tensor& qweight,
                             const torch::Tensor& qzeros,
                             const torch::Tensor& scales) const override;

 private:
  // parameter members, must be registered
  torch::Tensor g_idx_{nullptr};

  // Q4Matrix handler for exllama
  mutable uintptr_t q4_ = 0;
};

// Linear layer with row parallelism.
class RowParallelQLinearExllamaImpl : public RowParallelQLinearImpl {
 public:
  RowParallelQLinearExllamaImpl(int64_t in_features,
                                int64_t out_features,
                                bool bias,
                                const QuantArgs& quant_args,
                                bool input_is_parallelized,
                                const ParallelArgs& parallel_args,
                                const torch::TensorOptions& options);

  ~RowParallelQLinearExllamaImpl() override;

  torch::Tensor quant_matmul(const torch::Tensor& input,
                             const torch::Tensor& qweight,
                             const torch::Tensor& qzeros,
                             const torch::Tensor& scales) const override;

 private:
  // parameter members, must be registered
  torch::Tensor g_idx_{nullptr};

  // Q4Matrix handler for exllama
  mutable uintptr_t q4_ = 0;
};
}  // namespace llm
