#pragma once

#include <ATen/core/TensorBody.h>

#include <memory>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include "memory/block_manager.h"
#include "memory/cache_args.h"
#include "models/parameters.h"
#include "request/request.h"
#include "tokenizer/tokenizer.h"
#include "worker.h"

DECLARE_int32(max_seq_len);
DECLARE_int32(max_batch_size);
DECLARE_int32(block_size);
DECLARE_int64(max_cache_size);
DECLARE_double(max_memory_utilization);

namespace llm {

// The Large Language Model (LLM) engine is a model runner designed to execute
// inference procedures incrementally using batches of requests. It comprises
// three critical components: a model, a tokenizer, and a resource manager.
// The inference process is primarily divided into two stages: 'prefill' and
// 'generate'.
// * 'Prefill': This is the more costly phase, as it involves processing a
// new prompt and generating the entire initial attention matrix.
// * 'Generate': In this phase, subsequent tokens are generated using the
// previously cached attention matrix.
// A single batch may contain requests from various stages of the inference
// process. The engine must be adept at handling these diverse requests,
// ensuring optimal resource management.

class Engine {
 public:
  // create an engine with the given devices
  Engine(const torch::ScalarType& dtype,
         const std::vector<torch::Device>& devices);

  virtual ~Engine() = default;

  bool init(const std::string& model_weights_path,
            const std::string& tokenizer_path);

  // step the engine forward by one step with the batch
  OutputParameters execute_model(const std::vector<Sequence*>& batch);

  const Tokenizer* tokenizer() const { return tokenizer_.get(); }

  BlockManager* block_manager() const { return block_manager_.get(); }

  const ModelArgs& model_args() const { return args_; }

  const CacheArgs& cache_args() const { return cache_args_; }

 private:
  bool init_model(const std::string& model_weights_path);

  bool init_kv_cache();

  // dtype
  torch::ScalarType dtype_;

  // devices
  std::vector<torch::Device> devices_;

  // process groups
  std::vector<std::unique_ptr<c10d::Backend>> process_groups_;

  // model args
  ModelArgs args_;

  // a list of workers, with each worker handling a partial of model
  std::vector<std::unique_ptr<Worker>> workers_;

  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;

  // cache args
  CacheArgs cache_args_;

  // block manager
  std::unique_ptr<BlockManager> block_manager_;
};

}  // namespace llm
