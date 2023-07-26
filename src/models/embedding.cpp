#include "embedding.h"

#include <torch/nn/functional/embedding.h>
#include <torch/nn/module.h>
#include <torch/torch.h>

namespace llm {

// we don't need to initialize the weight here, since we will initialize it
// from the checkpoint with load_state_dict function
ColumnParallelEmbeddingImpl::ColumnParallelEmbeddingImpl(int64_t num_embeddings,
                                                         int64_t embedding_dim,
                                                         int64_t world_size)
    : world_size_(world_size) {
  // register the weight parameter
  weight_ = register_parameter("weight",
                               torch::randn({num_embeddings, embedding_dim}));
}

// The input to the module is a list of indices, and the output is the
// corresponding word embeddings.
torch::Tensor ColumnParallelEmbeddingImpl::forward(torch::Tensor input) {
  namespace F = torch::nn::functional;
  auto output = F::embedding(input, weight_);
  if (world_size_ > 1) {
    // call all reduce
    // torch::distributed::all_reduce(input_);
  }
  return output;
}

// load the weight from the checkpoint
void ColumnParallelEmbeddingImpl::load_state_dict(const StateDict& state_dict) {
  weight_.copy_(state_dict.get_tensor("weight"));
}

}  // namespace llm
