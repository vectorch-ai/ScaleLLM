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
  void execute_model(Batch& batch) override;

  // validate multiple speculative tokens when use speculative decoding
  void validate(Batch& batch) override;

  std::unique_ptr<Tokenizer> tokenizer() const override;

  BlockManager* block_manager() const override;

  const ModelArgs& model_args() const override;

  const QuantArgs& quant_args() const override;

  const TokenizerArgs& tokenizer_args() const override;

 private:
  bool init_model(const std::string& model_weights_path,
                  const std::string& draft_model_weights_path);

  bool init_kv_cache(int64_t cache_size_in_bytes);

  // engine
  std::unique_ptr<LLMEngine> engine_;

  // draft engine
  std::unique_ptr<LLMEngine> draft_engine_;
};

}  // namespace llm
