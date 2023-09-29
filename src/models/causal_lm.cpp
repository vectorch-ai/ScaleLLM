#include "causal_lm.h"

#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <vector>

#include "args.h"
#include "gpt_neox.h"
#include "input_parameters.h"
#include "llama2.h"
#include "llama2_hf.h"
#include "memory/kv_cache.h"
#include "model_loader/state_dict.h"

namespace llm {

std::unique_ptr<CausalLM> CausalLM::create(const ModelArgs& args,
                                           const QuantizationArgs& quant_args,
                                           const ParallelArgs& parallel_args,
                                           torch::ScalarType dtype,
                                           const torch::Device& device) {
  // create models based on model type
  if (boost::iequals(args.model_type(), "llama2")) {
    llama2::Model llama2(args, quant_args, parallel_args, dtype, device);
    llama2->eval();
    return std::make_unique<llm::CausalLMImpl<llama2::Model>>(
        std::move(llama2));
  }
  // llama from hf models
  if (boost::iequals(args.model_type(), "llama")) {
    hf::llama2::Model llama2(args, quant_args, parallel_args, dtype, device);
    // set the module in evaluation/inference mode
    llama2->eval();
    return std::make_unique<llm::CausalLMImpl<hf::llama2::Model>>(
        std::move(llama2));
  }

  if (boost::iequals(args.model_type(), "gpt_neox")) {
    hf::GPTNeoXModel gpt_neox(args, quant_args, parallel_args, dtype, device);
    // set the module in evaluation/inference mode
    gpt_neox->eval();
    return std::make_unique<llm::CausalLMImpl<hf::GPTNeoXModel>>(
        std::move(gpt_neox));
  }

  LOG(ERROR) << "Unsupported model type: " << args.model_type();
  return nullptr;
}

}  // namespace llm
