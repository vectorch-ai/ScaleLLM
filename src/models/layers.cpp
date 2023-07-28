#include "layers.h"

#include <glog/logging.h>
#include <torch/nn/functional/embedding.h>
#include <torch/nn/module.h>
#include <torch/torch.h>

namespace llm {

// we don't need to initialize the weight here, since we will initialize it
// from the checkpoint with load_state_dict function
ParallelEmbeddingImpl::ParallelEmbeddingImpl(int64_t num_embeddings,
                                             int64_t embedding_dim,
                                             int64_t world_size)
    : world_size_(world_size) {
  CHECK(embedding_dim % world_size == 0)
      << "out_features " << embedding_dim << " not divisible by world_size "
      << world_size;
  const int64_t embedding_dim_per_partition = embedding_dim / world_size;

  // register the weight parameter
  weight_ = register_parameter(
      "weight",
      torch::empty({num_embeddings, embedding_dim_per_partition}),
      /*requires_grad=*/false);
}

// The input to the module is a list of indices, and the output is the
// corresponding word embeddings.
torch::Tensor ParallelEmbeddingImpl::forward(torch::Tensor input) {
  namespace F = torch::nn::functional;
  auto output = F::embedding(input, weight_);
  if (world_size_ > 1) {
    // call all gather
    // torch::distributed::all_gather(input_);
  }
  return output;
}

// load the weight from the checkpoint
void ParallelEmbeddingImpl::load_state_dict(const StateDict& state_dict) {
  const auto weight = state_dict.get_tensor("weight");
  if (weight.defined()) {
    CHECK_EQ(weight_.sizes(), weight.sizes()) << "weight size mismatch";
    weight_.copy_(weight);
  } else {
    LOG(WARNING) << "weight is not defined";
  }
}

VocabParallelEmbeddingImpl::VocabParallelEmbeddingImpl(int64_t num_embeddings,
                                                       int64_t embedding_dim,
                                                       int64_t world_size)
    : world_size_(world_size) {
  const int64_t num_embeddings_per_partition = num_embeddings / world_size;

  // register the weight parameter
  weight_ = register_parameter(
      "weight",
      torch::empty({num_embeddings_per_partition, embedding_dim}),
      /*requires_grad=*/false);
}

// The input to the module is a list of indices, and the output is the
// corresponding word embeddings.
torch::Tensor VocabParallelEmbeddingImpl::forward(torch::Tensor input) {
  namespace F = torch::nn::functional;
  auto output = F::embedding(input, weight_);
  if (world_size_ > 1) {
    // call all gather
    // torch::distributed::all_gather(input_);
  }
  return output;
}

// load the weight from the checkpoint
void VocabParallelEmbeddingImpl::load_state_dict(const StateDict& state_dict) {
  const auto weight = state_dict.get_tensor("weight");
  if (weight.defined()) {
    CHECK_EQ(weight_.sizes(), weight.sizes()) << "weight size mismatch";
    weight_.copy_(weight);
  } else {
    LOG(WARNING) << "weight is not defined";
  }
}

ColumnParallelLinearImpl::ColumnParallelLinearImpl(int64_t in_features,
                                                   int64_t out_features,
                                                   int64_t world_size)
    : world_size_(world_size) {
  CHECK(out_features % world_size == 0)
      << "out_features " << out_features << " not divisible by world_size "
      << world_size;
  const int64_t out_features_per_partition = out_features / world_size;

  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  weight_ = register_parameter(
      "weight",
      torch::empty({out_features_per_partition, in_features}),
      /*requires_grad=*/false);
}

torch::Tensor ColumnParallelLinearImpl::forward(torch::Tensor input) {
  namespace F = torch::nn::functional;
  auto output = F::linear(input, weight_);
  if (world_size_ > 1) {
    // call all reduce or all gather with concat
    // torch::distributed::all_reduce(input_);
  }
  return output;
}

// load the weight from the checkpoint
void ColumnParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  const auto weight = state_dict.get_tensor("weight");
  if (weight.defined()) {
    CHECK_EQ(weight_.sizes(), weight.sizes()) << "weight size mismatch";
    weight_.copy_(weight);
  } else {
    LOG(WARNING) << "weight is not defined";
  }
}

RowParallelLinearImpl::RowParallelLinearImpl(int64_t in_features,
                                             int64_t out_features,
                                             int64_t world_size)
    : world_size_(world_size) {
  CHECK(in_features % world_size == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size;
  const int64_t in_features_per_partition = in_features / world_size;
  // Allocate the transpose since linear performs XA^T.
  weight_ = register_parameter(
      "weight",
      torch::empty({out_features, in_features_per_partition}),
      /*requires_grad=*/false);
}

torch::Tensor RowParallelLinearImpl::forward(torch::Tensor input) {
  namespace F = torch::nn::functional;
  auto output = F::linear(input, weight_);
  if (world_size_ > 1) {
    // call all reduce or all gather with concat
    // torch::distributed::all_reduce(input_);
  }
  return output;
}

// load the weight from the checkpoint
void RowParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  const auto weight = state_dict.get_tensor("weight");
  if (weight.defined()) {
    CHECK_EQ(weight_.sizes(), weight.sizes()) << "weight size mismatch";
    weight_.copy_(weight);
  } else {
    LOG(WARNING) << "weight is not defined";
  }
}

}  // namespace llm
