#pragma once

#include <c10/core/TensorImpl.h>
#include <torch/torch.h>

#include "model_loader/state_dict.h"
#include "models/args.h"
#include "qlinear_impl.h"

namespace llm {

// Quantized Linear layer with column parallelism.
class ColumnParallelQLinearExllamaImpl : public ColumnParallelQLinearImpl {
 public:
  ColumnParallelQLinearExllamaImpl(int64_t in_features,
                                   int64_t out_features,
                                   bool bias,
                                   const QuantizationArgs& quant_args,
                                   bool gather_output,
                                   const ParallelArgs& parallel_args,
                                   torch::ScalarType dtype,
                                   const torch::Device& device);

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
                                const QuantizationArgs& quant_args,
                                bool input_is_parallelized,
                                const ParallelArgs& parallel_args,
                                torch::ScalarType dtype,
                                const torch::Device& device);

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
