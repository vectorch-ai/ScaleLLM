#pragma once

#include "engine/llm_engine.h"
#include "request/stopping_criteria.h"
#include "sampling_params.h"

namespace llm {

class LLM {
 public:
  LLM(const std::string& model_path,
      const SamplingParams& sp,
      int64_t max_seq_len,
      const std::string& device_str);

  void generate(const std::vector<std::string>& batched_prompt);

 private:
  std::vector<torch::Device> parse_devices(const std::string& device_str);

 private:
  LLMEngine* engine_;
  BlockManager* block_manager_;
  SamplingParameter sampling_param_;
  StoppingCriteria stopping_criteria_;
  std::unique_ptr<Tokenizer> tokenizer_;
  int64_t max_seq_len_;
};

}  // namespace llm
