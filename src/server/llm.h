#pragma once

#include "engine/engine.h"
#include "request/sampling_parameter.h"
#include "request/stopping_criteria.h"

namespace llm {

class LLM {
 public:
  LLM(const std::string& model_path, const SamplingParameter& sp,
      const StoppingCriteria& sc, int64_t max_seq_len,
      const std::string& device_str);

  explicit LLM(const std::string& model_path);

  
  void generate(const std::vector<std::string>& batched_prompt);
 
 private:
  std::vector<torch::Device> parse_devices(const std::string& device_str);

 private:
  Engine* engine_;
  BlockManager* block_manager_;
  SamplingParameter sampling_param_;
  StoppingCriteria stopping_criteria_;
  std::unique_ptr<Tokenizer> tokenizer_;
  int64_t max_seq_len_;
};

} // namespace llm
