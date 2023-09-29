#pragma once

#include <ATen/core/TensorBody.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "../linear_impl.h"
#include "model_loader/state_dict.h"
#include "models/args.h"

namespace llm {

// Base QLinear class that handles quantized weights loading.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class ColumnParallelQLinearImpl : public ParallelLinearImpl {
 public:
  ColumnParallelQLinearImpl(int64_t in_features,
                            int64_t out_features,
                            bool bias,
                            int64_t bits,
                            int64_t group_size,
                            int64_t qweight_pack_dim,
                            int rank,
                            int world_size,
                            torch::ScalarType dtype,
                            const torch::Device& device);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) override;

  // special load_state_dict for fused cases
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string_view>& prefixes) override;

  // verify if the weight is loaded correctly
  void verify_loaded_weights(const std::string& prefix = "") const override;

 protected:
  // parameter members, must be registered
  torch::Tensor qweight_{nullptr};
  torch::Tensor qzeros_{nullptr};
  torch::Tensor scales_{nullptr};

  torch::Tensor bias_{nullptr};

 private:
  bool qweight_is_loaded_ = false;
  bool qzeros_is_loaded_ = false;
  bool scales_is_loaded_ = false;
  bool bias_is_loaded_ = false;
  std::vector<torch::Tensor> qweight_list_;
  std::vector<torch::Tensor> qzeros_list_;
  std::vector<torch::Tensor> scales_list_;
  std::vector<torch::Tensor> bias_list_;

  int rank_ = 0;
  int world_size_ = 0;
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
                         int64_t bits,
                         int64_t group_size,
                         int64_t qweight_pack_dim,
                         int rank,
                         int world_size,
                         torch::ScalarType dtype,
                         const torch::Device& device);

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

 protected:
  // parameter members, must be registered
  torch::Tensor qweight_{nullptr};
  torch::Tensor qzeros_{nullptr};
  torch::Tensor scales_{nullptr};

  torch::Tensor bias_{nullptr};

 private:
  bool qweight_is_loaded_ = false;
  bool qzeros_is_loaded_ = false;
  bool scales_is_loaded_ = false;
  bool bias_is_loaded_ = false;

  int rank_ = 0;
  int world_size_ = 0;
};
}  // namespace llm
