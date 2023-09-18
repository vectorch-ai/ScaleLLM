#include "causal_lm.h"

#include <torch/torch.h>

#include <vector>

#include "memory/kv_cache.h"
#include "model_loader/state_dict.h"
#include "models/llama2.h"
#include "models/model_args.h"
#include "models/parallel_args.h"
#include "parameters.h"

namespace llm {

std::unique_ptr<CausalLM> CausalLM::create(const ModelArgs& args,
                                           const ParallelArgs& parallel_args,
                                           const torch::ScalarType& dtype,
                                           const torch::Device& device) {
  // TODO: create models based on model name;
  llm::Llama2 llama2(args, parallel_args, dtype, device);
  // set the module in evaluation/inference mode
  llama2->eval();
  return std::make_unique<llm::CausalLMImpl<llm::Llama2>>(
      std::move(llama2));
}

}  // namespace llm
