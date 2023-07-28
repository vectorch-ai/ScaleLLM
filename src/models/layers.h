#pragma once

#include <torch/nn/functional/embedding.h>
#include <torch/nn/module.h>
#include <torch/torch.h>

#include "common/state_dict.h"

namespace llm {

// A simple lookup table that stores embeddings of a fixed dictionary and size.
// This module is often used to store word embeddings and retrieve them using
// indices.
// Embedding parallelized in the embedding dimension.
class ParallelEmbeddingImpl : public torch::nn::Module {
 public:
  ParallelEmbeddingImpl(int64_t num_embeddings,
                        int64_t embedding_dim,
                        int64_t world_size);

  // The input to the module is a list of indices, and the output is the
  // corresponding word embeddings.
  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes();
  }

 private:
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // configs
  int64_t world_size_;
};
TORCH_MODULE(ParallelEmbedding);

// Embedding parallelized in the vocabulary dimension
class VocabParallelEmbeddingImpl : public torch::nn::Module {
 public:
  VocabParallelEmbeddingImpl(int64_t num_embeddings,
                             int64_t embedding_dim,
                             int64_t world_size);

  // The input to the module is a list of indices, and the output is the
  // corresponding word embeddings.
  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes();
  }

 private:
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // configs
  int64_t world_size_;
};
TORCH_MODULE(VocabParallelEmbedding);

// Linear layer with column parallelism.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class ColumnParallelLinearImpl : public torch::nn::Module {
 public:
  ColumnParallelLinearImpl(int64_t in_features,
                           int64_t out_features,
                           int64_t world_size);

  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes();
  }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features_per_partition, in_features]
  torch::Tensor weight_{nullptr};

  // configs
  int64_t world_size_;
};
TORCH_MODULE(ColumnParallelLinear);

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
class RowParallelLinearImpl : public torch::nn::Module {
 public:
  RowParallelLinearImpl(int64_t in_features,
                        int64_t out_features,
                        int64_t world_size);

  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes();
  }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features, in_features_per_partition]
  torch::Tensor weight_{nullptr};

  // configs
  int64_t world_size_;
};
TORCH_MODULE(RowParallelLinear);
}  // namespace llm
