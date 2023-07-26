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
// Question: how to partition the embedding table?
class ColumnParallelEmbeddingImpl : public torch::nn::Module {
 public:
  ColumnParallelEmbeddingImpl(int64_t num_embeddings,
                              int64_t embedding_dim,
                              int64_t world_size);

  // The input to the module is a list of indices, and the output is the
  // corresponding word embeddings.
  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

 private:
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // configs
  int64_t world_size_;
};
TORCH_MODULE(ColumnParallelEmbedding);

// TODO: add RowParallelEmbedding, parallelized in the vocabulary dimension.

}  // namespace llm
