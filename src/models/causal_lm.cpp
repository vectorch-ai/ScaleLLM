#include "causal_lm.h"

#include <torch/torch.h>

#include <vector>

#include "input_parameters.h"
#include "llama2.h"
#include "llama2_hf.h"
#include "memory/kv_cache.h"
#include "model_args.h"
#include "model_loader/state_dict.h"
#include "parallel_args.h"

namespace llm {

std::unique_ptr<CausalLM> CausalLM::create(const ModelArgs& args,
                                           const ParallelArgs& parallel_args,
                                           const torch::ScalarType& dtype,
                                           const torch::Device& device) {
  // create models based on model architecure
  for (const auto& arch : args.architectures()) {
    if (arch == "llama2") {
      llama2::Model llama2(args, parallel_args, dtype, device);
      llama2->eval();
      return std::make_unique<llm::CausalLMImpl<llama2::Model>>(
          std::move(llama2));
    }
    // huggingface models
    if (arch == "LlamaForCausalLM" || arch == "LLaMAForCausalLM") {
      hf::llama2::Model llama2(args, parallel_args, dtype, device);
      // set the module in evaluation/inference mode
      llama2->eval();
      return std::make_unique<llm::CausalLMImpl<hf::llama2::Model>>(
          std::move(llama2));
    }
  }
  return nullptr;
}

}  // namespace llm
