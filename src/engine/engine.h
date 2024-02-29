#pragma once

#include <ATen/core/TensorBody.h>

#include <memory>
#include <torch/csrc/distributed/c10d/Backend.hpp>

#include "memory/block_manager.h"
#include "quantization/quant_args.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_args.h"
#include "worker.h"

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
  Engine(const std::vector<torch::Device>& devices);

  virtual ~Engine() = default;

  virtual bool init(const std::string& model_weights_path);

  // step the engine forward by one step with the batch
  virtual OutputParameters execute_model(const std::vector<Sequence*>& batch);

  // validate multiple speculative tokens when use speculative decoding
  virtual OutputParameters validate(const std::vector<Sequence*>& batch);

  virtual std::unique_ptr<Tokenizer> tokenizer() const {
    return tokenizer_->clone();
  }

  virtual BlockManager* block_manager() const {
    return block_manager_.get();
  }

  const ModelArgs& model_args() const { return args_; }

  const QuantArgs& quant_args() const { return quant_args_; }

  const TokenizerArgs& tokenizer_args() const { return tokenizer_args_; }

 private:
  bool init_model(const std::string& model_weights_path);

  bool init_kv_cache();

  // dtype
  torch::ScalarType dtype_;

  // devices
  std::vector<torch::Device> devices_;

  // model args
  ModelArgs args_;

  // quantization args
  QuantArgs quant_args_;

  // Tokenizer args
  TokenizerArgs tokenizer_args_;

  // a list of process groups, with each process group handling a single device
  std::vector<std::unique_ptr<ProcessGroup>> process_groups_;

  // a list of workers, with each worker handling a partial of model
  std::vector<std::unique_ptr<Worker>> workers_;

  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;

  // block manager
  std::unique_ptr<BlockManager> block_manager_;
};

}  // namespace llm
