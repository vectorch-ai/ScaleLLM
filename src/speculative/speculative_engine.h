#pragma once

#include "engine/batch.h"
#include "engine/engine.h"
#include "engine/llm_engine.h"
#include "memory/block_manager.h"
#include "quantization/quant_args.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_args.h"

namespace llm {

class SpeculativeEngine : public Engine {
 public:
  // create an engine with the given devices
  SpeculativeEngine(const std::vector<torch::Device>& devices,
                    const std::vector<torch::Device>& draft_devices);

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

  const ModelArgs& model_args() const override { return engine_->model_args(); }

  const TokenizerArgs& tokenizer_args() const override {
    return engine_->tokenizer_args();
  }

 private:
  bool init_model(const std::string& model_weights_path,
                  const std::string& draft_model_weights_path);

  bool init_kv_cache();

  int64_t calculate_kv_cache_blocks(int64_t cache_size_in_bytes) const;

  void validate(Batch& batch,
                const std::vector<ModelOutput>& draft_outputs,
                const ModelOutput& target_output);

  // engine
  std::unique_ptr<LLMEngine> engine_;

  // draft engine
  std::unique_ptr<LLMEngine> draft_engine_;

  // whether target and draft engine are sharing the same device
  bool share_device_ = false;
};

}  // namespace llm
