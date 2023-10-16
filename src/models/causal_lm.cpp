#include "causal_lm.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <vector>

#include "args.h"
#include "huggingface/aquila.h"
#include "huggingface/gpt2.h"
#include "huggingface/gpt_j.h"
#include "huggingface/gpt_neox.h"
#include "huggingface/internlm.h"
#include "huggingface/llama.h"
#include "huggingface/mistral.h"
#include "input_parameters.h"
#include "llama.h"
#include "memory/kv_cache.h"
#include "model_loader/state_dict.h"

namespace llm {

std::unique_ptr<CausalLM> CausalLM::create(const ModelArgs& args,
                                           const QuantizationArgs& quant_args,
                                           const ParallelArgs& parallel_args,
                                           torch::ScalarType dtype,
                                           const torch::Device& device) {
#define REGISTER_CAUSAL_MODEL(model_name, ModelClass)                         \
  if (boost::iequals(args.model_type(), model_name)) {                        \
    ModelClass model(args, quant_args, parallel_args, dtype, device);         \
    model->eval();                                                            \
    return std::make_unique<llm::CausalLMImpl<ModelClass>>(std::move(model)); \
  }

  // register causal models here
  REGISTER_CAUSAL_MODEL("llama2", LlamaModel);
  REGISTER_CAUSAL_MODEL("llama", hf::LlamaForCausalLM);
  REGISTER_CAUSAL_MODEL("gpt2", hf::GPT2ForCausalLM);
  REGISTER_CAUSAL_MODEL("gptj", hf::GPTJForCausalLM);
  REGISTER_CAUSAL_MODEL("gpt_neox", hf::GPTNeoXForCausalLM);
  REGISTER_CAUSAL_MODEL("mistral", hf::MistralForCausalLM);
  REGISTER_CAUSAL_MODEL("aquila", hf::AquilaForCausalLM);
  REGISTER_CAUSAL_MODEL("internlm", hf::InternlmForCausalLM);

  LOG(ERROR) << "Unsupported model type: " << args.model_type();
  return nullptr;
}

}  // namespace llm
