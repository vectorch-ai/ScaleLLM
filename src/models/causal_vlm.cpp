#include "causal_vlm.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "model_args.h"
#include "models/model_registry.h"

namespace llm {

std::unique_ptr<CausalVLM> CausalVLM::create(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  // get the factory function for the model type from model registry
  auto factory = ModelRegistry::get_causalvlm_factory(args.model_type());
  if (factory) {
    return factory(args, quant_args, parallel_args, options);
  }

  LOG(ERROR) << "Unsupported model type: " << args.model_type();
  return nullptr;
}

}  // namespace llm
