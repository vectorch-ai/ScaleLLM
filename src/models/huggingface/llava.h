#pragma once

#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include "chat_template/common_chat_template.h"
#include "layers/activation.h"
#include "layers/attention/attention.h"
#include "layers/attention/handler.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/normalization.h"
#include "layers/qkv_linear.h"
#include "memory/kv_cache.h"
#include "models/huggingface/llama.h"
#include "models/model_args.h"
#include "models/model_registry.h"
#include "models/parameters.h"

// llava model compatible with huggingface weights
namespace llm::hf {
class LlavaProjectorImpl : public torch::nn::Module {
 public:
  LlavaProjectorImpl(const ModelArgs& args,
                     const QuantArgs& quant_args,
                     const ParallelArgs& parallel_args,
                     const torch::TensorOptions& options) {
    act_func_ = Activation::get_act_func("gelu", options.device());
    CHECK(act_func_ != nullptr);

    auto int64_t vision_hidden_size = args.vision_hidden_size();
    auto int64_t text_hidden_size = args.text_hidden_size();
    // register the weight parameter
    linear_1_ =
        register_module("linear_1",
                        ColumnParallelLinear(vision_hidden_size,
                                             text_hidden_size,
                                             /*bias=*/true,
                                             /*gather_output=*/false,
                                             quant_args,
                                             parallel_args,
                                             options));
    linear_2_ =
        register_module("linear_2",
                        RowParallelLinear(text_hidden_size,
                                          text_hidden_size,
                                          /*bias=*/true,
                                          /*input_is_parallelized=*/true,
                                          quant_args,
                                          parallel_args,
                                          options));
  }
  
  torch::Tensor forward(torch::Tensor image_features) {
    return linear_2_(act_func_(linear_1(image_features)));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
  }

  void verify_loaded_weights(const std::string& prefix) const {
  }

 private:
  ColumnParallelLinear linear_1_{nullptr};
  RowParallelLinear linear_2_{nullptr};

  ActFunc act_func_{nullptr};
};

}  // namespace llm::hf
