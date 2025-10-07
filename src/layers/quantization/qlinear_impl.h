#pragma once

#include <ATen/core/TensorBody.h>
#include <torch/torch.h>

#include "layers/linear/parallel_linear.h"
#include "layers/linear/weight_utils.h"
#include "model_loader/state_dict.h"
#include "model_parallel/model_parallel.h"

namespace llm {

namespace detail {
// slow implementation for quantized matmul, used for testing and comparison
// construct weights matrix for gptq from quantized weights
// return the weights matrix [in_features, out_features] with following formula:
// weights = scales * (qweights - qzeros)
torch::Tensor construct_weights(
    const torch::Tensor& qweights,  // [n_ints, out_features] IntTensor
    const torch::Tensor& qzeros,    // [n_groups, n_ints] IntTensor
    const torch::Tensor& scales,    // [n_groups, out_features] HalfTensor
    const torch::Tensor& g_idx,     // [in_features] IntTensor
    int64_t bits);

// construct weights matrix for gptq from quantized weights without using g_idx
// slower than construct_weights with g_idx
// return the weights matrix [in_features, out_features] with following formula:
// weights = scales * (qweights - qzeros)
torch::Tensor construct_weights(
    const torch::Tensor& qweights,  // [n_ints, out_features] IntTensor
    const torch::Tensor& qzeros,    // [n_groups, n_ints] IntTensor
    const torch::Tensor& scales,    // [n_groups, out_features] HalfTensor
    int64_t bits);

}  // namespace detail

// Base QLinear class that handles quantized weights loading.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class ColumnParallelQLinearImpl : public ParallelLinearImpl {
 public:
  ColumnParallelQLinearImpl(int64_t in_features,
                            int64_t out_features,
                            bool bias,
                            const QuantArgs& quant_args,
                            int64_t qweight_pack_dim,
                            bool gather_output,
                            const ParallelArgs& parallel_args,
                            const torch::TensorOptions& options);

  // verify if the weight is loaded correctly
  void verify_loaded_weights(const std::string& prefix = "") const override;

  // all subclasses must implement this function
  virtual torch::Tensor quant_matmul(const torch::Tensor& input,
                                     const torch::Tensor& qweight,
                                     const torch::Tensor& qzeros_,
                                     const torch::Tensor& scales_) const;

  torch::Tensor forward(torch::Tensor input) override {
    auto output = quant_matmul(input, qweight_, qzeros_, scales_);
    if (bias_.defined()) {
      output.add_(bias_);
    }
    if (parallel_args_.world_size() > 1 && gather_output_) {
      output = gather_from_model_parallel_region(output, parallel_args_);
    }
    return output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) override;

  // special load_state_dict for fused cases
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes) override;

 private:
  // parameter members, must be registered
  DEFINE_FUSED_WEIGHT(qweight);
  DEFINE_FUSED_WEIGHT(qzeros);
  DEFINE_FUSED_WEIGHT(scales);
  DEFINE_FUSED_WEIGHT(bias);

  // quantization parameters
  int64_t bits_ = 0;

  // whether to gather the output
  bool gather_output_;

  // parallel args
  ParallelArgs parallel_args_;
};

// Base QLinear class that handles quantized weights loading.
//     The linear layer is defined as Y = XA + b. A is parallelized along
//     its first dimension and X along its second dimension as:
//                -   -
//               | A_1 |
//               | .   |
//           A = | .   |       X = [X_1, ..., X_p]
//               | .   |
//               | A_p |
//                -   -
class RowParallelQLinearImpl : public ParallelLinearImpl {
 public:
  RowParallelQLinearImpl(int64_t in_features,
                         int64_t out_features,
                         bool bias,
                         const QuantArgs& quant_args,
                         int64_t qweight_pack_dim,
                         bool input_is_parallelized,
                         const ParallelArgs& parallel_args,
                         const torch::TensorOptions& options);

  // all subclasses must implement this function
  virtual torch::Tensor quant_matmul(const torch::Tensor& input,
                                     const torch::Tensor& qweight,
                                     const torch::Tensor& qzeros_,
                                     const torch::Tensor& scales_) const;

  torch::Tensor forward(torch::Tensor input) override {
    if (!input_is_parallelized_) {
      input = scatter_to_model_parallel_region(input, parallel_args_);
    }

    auto output = quant_matmul(input, qweight_, qzeros_, scales_);
    if (parallel_args_.world_size() > 1) {
      output = reduce_from_model_parallel_region(output, parallel_args_);
    }
    // N.B. need to apply bias after the reduce
    if (bias_.defined()) {
      output.add_(bias_);
    }
    return output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) override;

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix = "") const override;

 private:
  // parameter members, must be registered
  DEFINE_WEIGHT(qweight);
  DEFINE_WEIGHT(qzeros);
  DEFINE_WEIGHT(scales);
  DEFINE_WEIGHT(bias);

  // quantization parameters
  int64_t bits_ = 0;

  // whether the input is already parallelized
  bool input_is_parallelized_;

  // parallel args
  ParallelArgs parallel_args_;
};
}  // namespace llm
