#pragma once

#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "linear.h"
#include "model_loader/state_dict.h"
#include "models/args.h"

namespace llm {
namespace details {

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

}  // namespace details

// quantized linear layers using gptq

// Quantized Linear layer with column parallelism.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class ColumnParallelQuantLinearImpl : public ParallelLinearImpl {
 public:
  ColumnParallelQuantLinearImpl(int64_t in_features,
                                int64_t out_features,
                                int64_t bits,
                                int64_t group_size,
                                bool gather_output,
                                const ParallelArgs& parallel_args,
                                const torch::ScalarType& dtype,
                                const torch::Device& device);

  torch::Tensor forward(torch::Tensor input) const override;

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) override;

  // special load_state_dict for fused cases
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string_view>& prefixes) override;

  // verify if the weight is loaded correctly
  void verify_loaded_weights(const std::string& prefix = "") const override;

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " qweight=" << qweight_.sizes()
           << " qzeros=" << qzeros_.sizes() << " scales=" << scales_.sizes()
           << " g_idx=" << g_idx_.sizes() << " device=" << qweight_.device();
  }

 private:
  bool load_weights(std::vector<torch::Tensor>& weight_list,
                    torch::Tensor& weight);

  // parameter members, must be registered
  torch::Tensor qweight_{nullptr};
  torch::Tensor qzeros_{nullptr};
  torch::Tensor scales_{nullptr};
  torch::Tensor g_idx_{nullptr};

  int64_t in_features_ = 0;
  int64_t out_features_ = 0;

  // quantization parameters
  int64_t bits_ = 0;
  int64_t group_size_ = 0;

  bool qweight_is_loaded_ = false;
  bool qzeros_is_loaded_ = false;
  bool scales_is_loaded_ = false;
  std::vector<torch::Tensor> qweight_list_;
  std::vector<torch::Tensor> qzeros_list_;
  std::vector<torch::Tensor> scales_list_;

  // parallel args
  ParallelArgs parallel_args_;

  // whether to gather the output
  bool gather_output_;
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
class RowParallelQuantLinearImpl : public ParallelLinearImpl {
 public:
  RowParallelQuantLinearImpl(int64_t in_features,
                             int64_t out_features,
                             int64_t bits,
                             int64_t group_size,
                             bool input_is_parallelized,
                             const ParallelArgs& parallel_args,
                             const torch::ScalarType& dtype,
                             const torch::Device& device);

  torch::Tensor forward(torch::Tensor input) const override;

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) override;

  // special load_state_dict for fused cases
  void load_state_dict(
      const StateDict& /*state_dict*/,
      const std::vector<std::string_view>& /*prefixes*/) override {
    LOG(FATAL) << "not implemented";
  }

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix = "") const override;

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " qweight=" << qweight_.sizes()
           << " qzeros=" << qzeros_.sizes() << " scales=" << scales_.sizes()
           << " g_idx=" << g_idx_.sizes() << " device=" << qweight_.device();
  }

 private:
  // parameter members, must be registered
  torch::Tensor qweight_{nullptr};
  torch::Tensor qzeros_{nullptr};
  torch::Tensor scales_{nullptr};
  torch::Tensor g_idx_{nullptr};

  int64_t in_features_ = 0;
  int64_t out_features_ = 0;

  // quantization parameters
  int64_t bits_ = 0;
  int64_t group_size_ = 0;

  bool qweight_is_loaded_ = false;
  bool qzeros_is_loaded_ = false;
  bool scales_is_loaded_ = false;

  // parallel args
  ParallelArgs parallel_args_;

  // whether the input is already parallelized
  bool input_is_parallelized_;
};
}  // namespace llm
