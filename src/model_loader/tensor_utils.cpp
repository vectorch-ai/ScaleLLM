#include "tensor_utils.h"

#include <torch/torch.h>

#include <vector>

#include "state_dict.h"

namespace llm {
namespace {

// merge weights from multiple shards used for weights fusion
bool merge_weights(const std::string& tensor_name,
                   std::vector<torch::Tensor> tensors,
                   int64_t dim,
                   std::vector<torch::Tensor>& accumulated_tensors,
                   torch::Tensor& weight) {
  // resize the accumulated weight list if needed
  if (accumulated_tensors.size() < tensors.size()) {
    accumulated_tensors.resize(tensors.size());
  }

  // copy over accumulated weights
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (accumulated_tensors[i].defined()) {
      CHECK(!tensors[i].defined()) << tensor_name << " weight already set";
      tensors[i] = accumulated_tensors[i];
    }
  }

  const bool all_loaded =
      std::all_of(tensors.begin(), tensors.end(), [](const torch::Tensor& t) {
        return t.defined();
      });
  if (!all_loaded) {
    // accumulate the weights for future merge
    for (size_t i = 0; i < tensors.size(); ++i) {
      if (!accumulated_tensors[i].defined() && tensors[i].defined()) {
        // make a clone for safety
        accumulated_tensors[i] = tensors[i].clone();
      }
    }
    return false;
  }

  const auto merged_weight = torch::cat(tensors, /*dim=*/dim);
  CHECK_EQ(weight.sizes(), merged_weight.sizes())
      << "weight size mismatch for " << tensor_name;
  weight.copy_(merged_weight);
  // release the memory for weight_list
  accumulated_tensors.clear();
  return true;
}
}  // namespace

bool TensorUtils::load_weights(const StateDict& state_dict,
                               const std::string& name,
                               torch::Tensor& weight) {
  const auto tensor = state_dict.get_tensor(name);
  if (!tensor.defined()) {
    return false;
  }
  CHECK_EQ(weight.sizes(), tensor.sizes())
      << "weight size mismatch for " << name;
  weight.copy_(tensor);
  return true;
}

bool TensorUtils::load_sharded_weights(const StateDict& state_dict,
                                       const std::string& name,
                                       int64_t dim,
                                       int32_t rank,
                                       int32_t world_size,
                                       torch::Tensor& weight) {
  const auto tensor =
      state_dict.get_sharded_tensor(name, dim, rank, world_size);
  if (!tensor.defined()) {
    return false;
  }

  CHECK_EQ(weight.sizes(), tensor.sizes())
      << "weight size mismatch for " << name;
  weight.copy_(tensor);
  return true;
}

bool TensorUtils::load_sharded_weights(const StateDict& state_dict,
                                       const std::string& name,
                                       TensorTransform transform_func,
                                       int64_t dim,
                                       int32_t rank,
                                       int32_t world_size,
                                       torch::Tensor& weight) {
  auto tensor = state_dict.get_sharded_tensor(name, dim, rank, world_size);
  if (!tensor.defined()) {
    return false;
  }

  tensor = transform_func(tensor);
  CHECK_EQ(weight.sizes(), tensor.sizes())
      << "weight size mismatch for " << name;
  weight.copy_(tensor);
  return true;
}

void TensorUtils::load_fused_weights(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes,
    const std::string& name,
    int64_t dim,
    int32_t rank,
    int32_t world_size,
    std::vector<torch::Tensor>& accumulated_weights,
    torch::Tensor& weight,
    bool& weight_is_loaded) {
  // return if the weight is already loaded
  if (weight_is_loaded) {
    return;
  }

  // load the weights from the state_dict
  std::vector<torch::Tensor> tensors(prefixes.size());
  for (size_t i = 0; i < prefixes.size(); ++i) {
    const std::string tensor_name = prefixes[i] + name;
    const auto tensor =
        state_dict.get_sharded_tensor(tensor_name, dim, rank, world_size);
    if (tensor.defined()) {
      CHECK(!tensors[i].defined())
          << "weight already loaded, name: " << tensor_name;
      tensors[i] = tensor;
    }
  }

  // merge with accumulated weights
  weight_is_loaded =
      merge_weights(name, std::move(tensors), dim, accumulated_weights, weight);
}

}  // namespace llm