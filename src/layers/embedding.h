#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>

#include "model_loader/state_dict.h"
#include "model_parallel/model_parallel.h"
#include "module/module.h"
#include "module/module_holder.h"

namespace llm {

// A simple lookup table that stores embeddings of a fixed dictionary and size.
// This module is often used to store word embeddings and retrieve them using
// indices.

class EmbeddingImpl : public llm::nn::Module {
 public:
  EmbeddingImpl(int64_t num_embeddings,
                int64_t embedding_dim,
                const torch::TensorOptions& options) {
    // register the weight parameter
    weight_ = register_parameter(
        "weight",
        torch::empty({num_embeddings, embedding_dim}, options),
        /*requires_grad=*/false);
  }

  // The input to the module is a list of indices, and the output is the
  // corresponding word embeddings.
  torch::Tensor forward(torch::Tensor input) {
    namespace F = torch::nn::functional;
    return F::embedding(input, weight_);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto weight = state_dict.get_tensor("weight");
    if (weight.defined()) {
      CHECK_EQ(weight_.sizes(), weight.sizes())
          << "weight size mismatch for " << name();
      weight_.copy_(weight);
      is_loaded_ = true;
    }
  }

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_loaded_) << "weight is not loaded for " << prefix + "weight";
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // whether the weight is loaded
  bool is_loaded_ = false;
};
LLM_MODULE(Embedding);

// Embedding parallelized in the embedding dimension.
class ParallelEmbeddingImpl : public llm::nn::Module {
 public:
  ParallelEmbeddingImpl(int64_t num_embeddings,
                        int64_t embedding_dim,
                        const ParallelArgs& parallel_args,
                        const torch::TensorOptions& options)
      : parallel_args_(parallel_args) {
    const auto world_size = parallel_args_.world_size();
    CHECK(embedding_dim % world_size == 0)
        << "out_features " << embedding_dim << " not divisible by world_size "
        << world_size;
    const int64_t embedding_dim_per_partition = embedding_dim / world_size;

    // register the weight parameter
    weight_ = register_parameter(
        "weight",
        torch::empty({num_embeddings, embedding_dim_per_partition}, options),
        /*requires_grad=*/false);
  }

  // The input to the module is a list of indices, and the output is the
  // corresponding word embeddings.
  torch::Tensor forward(torch::Tensor input) {
    namespace F = torch::nn::functional;
    auto output = F::embedding(input, weight_);
    if (parallel_args_.world_size() > 1) {
      output = gather_from_model_parallel_region(output, parallel_args_);
    }
    return output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto weight = state_dict.get_sharded_tensor(
        "weight",
        /*dim=*/1,
        /*rank=*/parallel_args_.rank(),
        /*world_size=*/parallel_args_.world_size());
    if (weight.defined()) {
      CHECK_EQ(weight_.sizes(), weight.sizes())
          << "weight size mismatch for " << name();
      weight_.copy_(weight);
      is_loaded_ = true;
    }
  }

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_loaded_) << "weight is not loaded for " << prefix + "weight";
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // whether the weight is loaded
  bool is_loaded_ = false;

  // parallel args
  ParallelArgs parallel_args_;
};
LLM_MODULE(ParallelEmbedding);

// Embedding parallelized in the vocabulary dimension
class VocabParallelEmbeddingImpl : public llm::nn::Module {
 public:
  VocabParallelEmbeddingImpl(int64_t num_embeddings,
                             int64_t embedding_dim,
                             const ParallelArgs& parallel_args,
                             const torch::TensorOptions& options)
      : parallel_args_(parallel_args) {
    const int64_t num_embeddings_per_partition =
        num_embeddings / parallel_args_.world_size();
    start_index_ = num_embeddings_per_partition * parallel_args_.rank();
    end_index_ = start_index_ + num_embeddings_per_partition;

    // register the weight parameter
    weight_ = register_parameter(
        "weight",
        torch::empty({num_embeddings_per_partition, embedding_dim}, options),
        /*requires_grad=*/false);
  }

  // The input to the module is a list of indices, and the output is the
  // corresponding word embeddings.
  torch::Tensor forward(torch::Tensor input) {
    namespace F = torch::nn::functional;
    if (parallel_args_.world_size() == 1) {
      return F::embedding(input, weight_);
    }

    // build mask to filter out tokens that are not in the local vocab
    auto masked_input = input - start_index_;
    const auto input_mask = (input < start_index_) | (input >= end_index_);
    masked_input.masked_fill_(input_mask, 0);

    auto output = F::embedding(masked_input, weight_);
    // mask out tokens not in the local vocab before reduce
    output.masked_fill_(input_mask.unsqueeze(-1), 0);
    // reduce across all gpus
    return reduce_from_model_parallel_region(output, parallel_args_);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto weight = state_dict.get_sharded_tensor(
        "weight",
        /*dim=*/0,
        /*rank=*/parallel_args_.rank(),
        /*world_size=*/parallel_args_.world_size());
    if (weight.defined()) {
      CHECK_EQ(weight_.sizes(), weight.sizes())
          << "weight size mismatch for " << name();
      weight_.copy_(weight);
      is_loaded_ = true;
    }
  }

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix = "") const {
    CHECK(is_loaded_) << "weight is not loaded for " << prefix + "weight";
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // whether the weight is loaded
  bool is_loaded_ = false;

  // parallel args
  ParallelArgs parallel_args_;

  // index used to mask out tokens that are not in the local vocab
  int64_t start_index_ = 0;
  int64_t end_index_ = 0;
};
LLM_MODULE(VocabParallelEmbedding);
}  // namespace llm
