#pragma once

#include <torch/torch.h>

#include <vector>

#include "state_dict.h"

namespace llm {

class TensorUtils {
 public:
  static bool load_weights(const StateDict& state_dict,
                           const std::string& name,
                           torch::Tensor& weight);

  static bool load_sharded_weights(const StateDict& state_dict,
                                   const std::string& name,
                                   int64_t dim,
                                   int32_t rank,
                                   int32_t world_size,
                                   torch::Tensor& weight);

  using TensorTransform = std::function<torch::Tensor(const torch::Tensor&)>;
  static bool load_sharded_weights(const StateDict& state_dict,
                                   const std::string& name,
                                   TensorTransform transform_func,
                                   int64_t dim,
                                   int32_t rank,
                                   int32_t world_size,
                                   torch::Tensor& weight);

  static void load_fused_weights(
      const StateDict& state_dict,
      const std::vector<std::string>& prefixes,
      const std::string& name,
      int64_t dim,
      int32_t rank,
      int32_t world_size,
      std::vector<torch::Tensor>& accumulated_weights,
      torch::Tensor& weight,
      bool& weight_is_loaded);
};

}  // namespace llm