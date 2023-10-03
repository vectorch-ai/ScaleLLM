#pragma once

#include <optional>

#include "common/arg.h"
#include "common/process_group.h"

namespace llm {

struct ModelArgs {
  DEFINE_ARG(std::vector<std::string>, architectures);

  DEFINE_ARG(std::string, model_type);

  // dimension of the encoder layer.
  DEFINE_ARG(int64_t, hidden_size) = 4096;

  // dimension of the 'intermediate' (aka feed-forward) layer.
  DEFINE_ARG(int64_t, intermediate_size) = 11008;

  // number of hidden layers in the encoder.
  DEFINE_ARG(int64_t, n_layers) = 32;

  // number of attention heads.
  DEFINE_ARG(int64_t, n_heads) = 32;

  // number of attention heads for key/value.
  DEFINE_ARG(std::optional<int64_t>, n_kv_heads);

  // number of tokens in the vocabulary.
  DEFINE_ARG(int64_t, vocab_size) = -1;

  // the epsilon value to use for rms norm.
  DEFINE_ARG(float, rms_norm_eps) = 1e-5;

  // the epsilon value to use for layer norm.
  DEFINE_ARG(float, layer_norm_eps) = 1e-5;

  // args for rotary position embeddings
  // the base period of the rotary position embeddings.
  DEFINE_ARG(float, rope_theta) = 10000.0f;

  // rope scaling factor.
  DEFINE_ARG(float, rope_scaling) = 0.0f;

  // percentage of hidden dimension to allocate to rotary position embeddings.
  DEFINE_ARG(float, rotary_pct) = 1.0f;

  // the maximum sequence length to use for rotary position embeddings.
  DEFINE_ARG(int64_t, max_position_embeddings) = 4096;

  // token id for beginning of sentence.
  DEFINE_ARG(int32_t, bos_token_id) = 0;

  // token id for end of sentence.
  DEFINE_ARG(int32_t, eos_token_id) = 0;

  // configs for gpt_neox
  // whether to use a 'parallel' formulation in each transformer layer, which
  // can provide a slight training speedup at large scales (e.g. 20B).
  DEFINE_ARG(bool, use_parallel_residual) = true;
};

struct QuantizationArgs {
  DEFINE_ARG(std::string, quant_method) = "";

  // quantization bits
  DEFINE_ARG(int64_t, bits) = 0;

  // quantization group size
  DEFINE_ARG(int64_t, group_size) = 4096;

  DEFINE_ARG(bool, desc_act) = false;

  DEFINE_ARG(bool, true_sequential) = false;
};

struct ParallelArgs {
  ParallelArgs(int32_t rank, int32_t world_size, ProcessGroup* process_group)
      : rank_(rank), world_size_(world_size), process_group_(process_group) {}

  // rank of current process
  DEFINE_ARG(int32_t, rank) = 0;

  // world size
  DEFINE_ARG(int32_t, world_size) = 0;

  // pointer to process group, nullptr if world size is 1
  DEFINE_PTR_ARG(ProcessGroup, process_group) = nullptr;
};

}  // namespace llm
