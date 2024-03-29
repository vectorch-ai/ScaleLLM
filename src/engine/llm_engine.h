#pragma once

#include <memory>

#include "batch.h"
#include "engine.h"
#include "memory/block_manager.h"
#include "quantization/quant_args.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_args.h"
#include "worker.h"

namespace llm {

// The Large Language Model (LLM) engine is a model runner designed to execute
// inference procedures incrementally using batches of requests. It comprises
// three critical components: a model, a tokenizer, and a resource manager.
// The inference process is primarily divided into two stages: 'prefill' and
// 'decode'.
// * 'Prefill': This is the more costly phase, as it involves processing the
// prompt and generating kv caches.
// * 'decode': In this stage, subsequent tokens are generated using the
// previously generated kv caches.
//
// A single batch may contain requests from two stages of the inference
// process. The engine must be adept at handling these diverse requests,
// ensuring optimal resource management.

class LLMEngine : public Engine {
 public:
  // create an engine with the given devices
  LLMEngine(const std::vector<torch::Device>& devices);

  virtual ~LLMEngine() = default;

  virtual bool init(const std::string& model_weights_path);

  // step the engine forward by one step with the batch
  void execute_model(Batch& batch) override;

  // validate multiple speculative tokens when use speculative decoding
  void validate(Batch& batch) override;

  std::unique_ptr<Tokenizer> tokenizer() const override {
    return tokenizer_->clone();
  }

  BlockManager* block_manager() const override { return block_manager_.get(); }

  const ModelArgs& model_args() const override { return args_; }

  const QuantArgs& quant_args() const override { return quant_args_; }

  const TokenizerArgs& tokenizer_args() const override {
    return tokenizer_args_;
  }

  bool init_model(const std::string& model_weights_path);

  bool init_kv_cache(int64_t cache_size_in_bytes);

  // returns the memory size for the kv cache
  int64_t profile_memory_for_kv_cache();

 private:
  bool warmup_model();

  // devices
  const std::vector<torch::Device> devices_;

  // dtype
  torch::ScalarType dtype_;

  // model args
  ModelArgs args_;

  // quantization args
  QuantArgs quant_args_;

  // Tokenizer args
  TokenizerArgs tokenizer_args_;

  // block manager
  std::unique_ptr<BlockManager> block_manager_;

  // a list of process groups, with each process group handling a single device
  std::vector<std::unique_ptr<ProcessGroup>> process_groups_;

  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;

  // a list of workers, with each worker handling a partial of model
  std::vector<std::unique_ptr<Worker>> workers_;
};

}  // namespace llm
