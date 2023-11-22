#include "causal_lm.h"

#include <torch/torch.h>

#include <vector>

#include "args.h"
#include "common/logging.h"
#include "input_parameters.h"
#include "memory/kv_cache.h"
#include "model_loader/state_dict.h"
#include "models/model_registry.h"

namespace llm {

std::unique_ptr<CausalLM> CausalLM::create(const ModelArgs& args,
                                           const QuantArgs& quant_args,
                                           const ParallelArgs& parallel_args,
                                           torch::ScalarType dtype,
                                           const torch::Device& device) {
  // get the factory function for the model type from model registry
  auto factory = ModelRegistry::get_causallm_factory(args.model_type());
  if (factory) {
    return factory(args, quant_args, parallel_args, dtype, device);
  }

  GLOG(ERROR) << "Unsupported model type: " << args.model_type();
  return nullptr;
}

}  // namespace llm
