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
TORCH_MODULE(LlavaProjector);

// TODO: CLIP, https://github.com/huggingface/transformers
class CLIPVisionModelImpl : public torch::nn::Module {
 public:
  CLIPVisionModelImpl(const ModelArgs& args,
                      const QuantArgs& quant_args,
                      const ParallelArgs& parallel_args,
                      const torch::TensorOptions& options) {
  }

  torch::Tensor forward(const torch::Tensor& images) {
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
  }

  void verify_loaded_weights() const {
  }
};
TORCH_MODULE(CLIPVisionModel);

class LlavaModelForCausalLMImpl : public torch::nn::Module {
 public:
  LlavaModelForCasalLMImpl(const ModelArgs& args,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options) {
    vision_model_ = register_module(
        "vision_model", CLIPVisionModel(args, quant_args, parallel_args, options));

    projector_ = register_module(
        "projector", LlavaProjector(args, quant_args, parallel_args, options));

    llm_model_ = register_module(
        "llm_model", LlamaModel(args, quant_args, parallel_args, options));

    lm_head_ = register_module("lm_head",
                               ColumnParallelLinear(args.hidden_size(),
                                                    args.vocab_size(),
                                                    /*bias=*/false,
                                                    /*gather_output*/true,
                                                    parallel_args,
                                                    options));
  }

  torch::Tensor forward(const torch::Tensor& images,
                        const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto& image_features = vision_model_(images);
    auto& vision_embedding = projector_(image_features);
    // TODO: Merge vision_embedding, text embedding
    return llm_model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    auto h = hidden_states;
    if (selected_idxes.defined()) {
      h = h.index_select(/*dim=*/0, selected_idxes);
    }
    return lm_head_(h);
  }
  
  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
  }

  void verify_loaded_weights() const {
  }

 private:
  LlamaModel llm_model_{nullptr};

  ColumnParallelLinear lm_head_{nullptr};
};
TORCH_MODULE(LlavaForCausalLM);

}  // namespace llm::hf
