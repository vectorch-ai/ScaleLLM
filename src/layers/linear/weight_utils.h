#pragma once

#include <torch/torch.h>

#include <vector>

#include "model_loader/state_dict.h"

namespace llm {

class WeightUtils {
 public:
  static void load_weight(const StateDict& state_dict,
                          const std::string& name,
                          torch::Tensor& weight,
                          bool& weight_is_loaded);

  static void load_sharded_weight(const StateDict& state_dict,
                                  const std::string& name,
                                  int64_t dim,
                                  int32_t rank,
                                  int32_t world_size,
                                  torch::Tensor& weight,
                                  bool& weight_is_loaded);

  using TensorTransform = std::function<torch::Tensor(const torch::Tensor&)>;
  static void load_sharded_weight(const StateDict& state_dict,
                                  const std::string& name,
                                  TensorTransform transform_func,
                                  int64_t dim,
                                  int32_t rank,
                                  int32_t world_size,
                                  torch::Tensor& weight,
                                  bool& weight_is_loaded);

  static void load_fused_weight(const StateDict& state_dict,
                                const std::vector<std::string>& prefixes,
                                const std::string& name,
                                int64_t dim,
                                int32_t rank,
                                int32_t world_size,
                                std::vector<torch::Tensor>& accumulated_tensors,
                                torch::Tensor& weight,
                                bool& weight_is_loaded);
};

// helper macros for defining and loading weights
#define DEFINE_WEIGHT(name) \
  torch::Tensor name##_;    \
  bool name##_is_loaded_ = false;

#define DEFINE_FUSED_WEIGHT(name) \
  torch::Tensor name##_;          \
  bool name##_is_loaded_ = false; \
  std::vector<torch::Tensor> name##_list_;

#define LOAD_FUSED_WEIGHT(name, dim)           \
  WeightUtils::load_fused_weight(state_dict,   \
                                 prefixes,     \
                                 #name,        \
                                 dim,          \
                                 rank,         \
                                 world_size,   \
                                 name##_list_, \
                                 name##_,      \
                                 name##_is_loaded_);

#define LOAD_SHARDED_WEIGHT(name, dim) \
  WeightUtils::load_sharded_weight(    \
      state_dict, #name, dim, rank, world_size, name##_, name##_is_loaded_);

#define LOAD_SHARDED_WEIGHT_WITH_TRANSFORM(name, dim) \
  WeightUtils::load_sharded_weight(state_dict,        \
                                   #name,             \
                                   transform_func,    \
                                   dim,               \
                                   rank,              \
                                   world_size,        \
                                   name##_,           \
                                   name##_is_loaded_);

#define LOAD_WEIGHT(name) \
  WeightUtils::load_weight(state_dict, #name, name##_, name##_is_loaded_);

}  // namespace llm
