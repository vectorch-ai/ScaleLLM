#pragma once

#include "common/macros.h"
#include "engine/batch.h"
#include "engine/engine.h"
#include "engine/llm_engine.h"
#include "memory/block_manager.h"
#include "models/model_args.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_args.h"

namespace llm {

class SpeculativeEngine : public Engine {
 public:
  struct Options {
    DEFINE_ARG(std::vector<torch::Device>, devices);

    DEFINE_ARG(std::vector<torch::Device>, draft_devices);

    // the number of slots per block, default 16, value must be multiple of 16
    DEFINE_ARG(int32_t, block_size) = 16;

    // the maximum cache size in bytes, default 10GB
    DEFINE_ARG(int64_t, max_cache_size) = 0;

    // maximum memory utilization allowed, default 0.9
    DEFINE_ARG(double, max_memory_utilization) = 0;

    // enable prefix cache
    DEFINE_ARG(bool, enable_prefix_cache) = true;

    // the number of speculative tokens per step
    DEFINE_ARG(int32_t, num_speculative_tokens) = 0;

    // enable cuda graph
    DEFINE_ARG(bool, enable_cuda_graph) = true;

    // max sequence length used to capture cuda graphs
    DEFINE_ARG(int64_t, cuda_graph_max_seq_len) = 1024;

    // batch sizes to capture cuda graphs
    DEFINE_ARG(std::optional<std::vector<uint32_t>>, cuda_graph_batch_sizes);

    // batch sizes to capture cuda graphs for draft model
    DEFINE_ARG(std::optional<std::vector<uint32_t>>, draft_cuda_graph_batch_sizes);
  };

  // create an engine with the given devices
  SpeculativeEngine(const Options& options);

  virtual ~SpeculativeEngine() = default;

  bool init(const std::string& model_weights_path,
            const std::string& draft_model_weights_path);

  // step the engine forward by one step with the batch
  // N.B. the model output is the output of the target model.
  ModelOutput execute_model(Batch& batch) override;

  std::unique_ptr<Tokenizer> tokenizer() const override {
    return engine_->tokenizer();
  }

  BlockManager* block_manager() const override {
    return engine_->block_manager();
  }

  const ModelArgs& model_args() const override { return model_args_; }

  const TokenizerArgs& tokenizer_args() const override {
    return engine_->tokenizer_args();
  }

 private:
  bool init_model(const std::string& model_weights_path,
                  const std::string& draft_model_weights_path);

  bool init_kv_cache();

  int64_t calculate_kv_cache_blocks(int64_t cache_size_in_bytes) const;

  static void validate(Batch& batch,
                       const std::vector<ModelOutput>& draft_outputs,
                       const ModelOutput& target_output);

  // options
  const Options options_;

  // engine
  std::unique_ptr<LLMEngine> engine_;

  // draft engine
  std::unique_ptr<LLMEngine> draft_engine_;

  // whether target and draft engine are sharing the same device
  bool share_device_ = false;

  ModelArgs model_args_;
};

}  // namespace llm
