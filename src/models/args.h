#pragma once

#include <optional>

#include "common/arg.h"
#include "common/process_group.h"

namespace llm {

struct ModelArgs {
  DEFINE_ARG(std::vector<std::string>, architectures);

  DEFINE_ARG(int64_t, dim) = 4096;

  DEFINE_ARG(int64_t, hidden_dim) = 11008;

  DEFINE_ARG(int64_t, n_layers) = 32;

  DEFINE_ARG(int64_t, n_heads) = 32;

  DEFINE_ARG(std::optional<int64_t>, n_kv_heads);

  // defined later by tokenizer
  DEFINE_ARG(int64_t, vocab_size) = -1;

  // make SwiGLU hidden layer size multiple of large power of 2
  DEFINE_ARG(int64_t, multiple_of) = 256;

  DEFINE_ARG(std::optional<float>, ffn_dim_multiplier);

  DEFINE_ARG(float, norm_eps) = 1e-5;

  DEFINE_ARG(float, rope_theta) = 10000.0f;

  DEFINE_ARG(float, rope_scaling) = 0.0f;

  // TODO: following two should not be part of model args
  DEFINE_ARG(int64_t, max_batch_size) = 32;

  DEFINE_ARG(int64_t, max_seq_len) = 2048;
};

struct QuantizationArgs {
  DEFINE_ARG(std::string, quant_method) = "";

  // quantization bits
  DEFINE_ARG(int64_t, bits) = 0;

  // quantization group size
  DEFINE_ARG(int64_t, group_size) = 4096;
};

struct ParallelArgs {
  ParallelArgs(int32_t rank, int32_t world_size, ProcessGroup* process_group)
      : rank_(rank), world_size_(world_size), process_group_(process_group) {
  }

  // rank of current process
  DEFINE_ARG(int32_t, rank) = 0;

  // world size
  DEFINE_ARG(int32_t, world_size) = 0;

    // pointer to process group, nullptr if world size is 1
  DEFINE_PTR_ARG(ProcessGroup, process_group) = nullptr;
};

}  // namespace llm
