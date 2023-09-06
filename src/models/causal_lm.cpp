#include "causal_lm.h"

#include <torch/torch.h>

#include <vector>

#include "memory/kv_cache.h"
#include "models/llama/transformer.h"
#include "models/model_args.h"
#include "parameters.h"
#include "torch_utils/state_dict.h"

namespace llm {

std::unique_ptr<CausalLM> CausalLM::create(const ModelArgs& args,
                                           const torch::Device& device) {
  // TODO: create models based on model name;
  llm::Transformer transformer(args, /*world_size=*/1, device);
  // set the module in evaluation/inference mode
  transformer->eval();
  return std::make_unique<llm::CausalLMImpl<llm::Transformer>>(
      std::move(transformer));
}

}  // namespace llm
