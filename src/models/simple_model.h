#pragma once

#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include "layers/activation.h"
#include "layers/embedding.h"
#include "layers/fused_linear.h"
#include "layers/linear.h"
#include "memory/kv_cache.h"
#include "models/model_args.h"
#include "models/model_registry.h"
#include "models/parameters.h"

// simple model for test
namespace llm {

class SimpleMLPImpl : public torch::nn::Module {
 public:
  SimpleMLPImpl(const ModelArgs& args,
                const QuantArgs& quant_args,
                const ParallelArgs& parallel_args,
                const torch::TensorOptions& options) {
    act_func_ = Activation::get_act_func("silu", options.device());
    CHECK(act_func_ != nullptr);

    const int64_t hidden_size = args.hidden_size();
    const int64_t intermediate_size = args.intermediate_size();

    gate_up_proj_ = register_module(
        "gate_up_proj",
        FusedColumnParallelLinear(
            hidden_size,
            std::vector<int64_t>{intermediate_size, intermediate_size},
            false,
            false,
            quant_args,
            parallel_args,
            options));
    down_proj_ = register_module("down_proj",
                                 RowParallelLinear(intermediate_size,
                                                   hidden_size,
                                                   false,
                                                   true,
                                                   quant_args,
                                                   parallel_args,
                                                   options));
  }

  torch::Tensor forward(torch::Tensor x) {
    const auto gate_up = gate_up_proj_(x);
    return down_proj_(act_func_(gate_up[0]) * gate_up[1]);
  }

  void load_state_dict(const StateDict& state_dict) {
    gate_up_proj_->load_state_dict(state_dict, {"gate_proj.", "up_proj."});
    down_proj_->load_state_dict(state_dict.select("down_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    gate_up_proj_->verify_loaded_weights(prefix + "[gate_proj,up_proj].");
    down_proj_->verify_loaded_weights(prefix + "down_proj.");
  }

 private:
  FusedColumnParallelLinear gate_up_proj_{nullptr};
  RowParallelLinear down_proj_{nullptr};

  ActFunc act_func_{nullptr};
};

TORCH_MODULE(SimpleMLP);

class SimpleDecoderLayerImpl : public torch::nn::Module {
 public:
  SimpleDecoderLayerImpl(const ModelArgs& args,
                         const QuantArgs& quant_args,
                         const ParallelArgs& parallel_args,
                         const torch::TensorOptions& options) {
    mlp_ = register_module("mlp",
                           SimpleMLP(args, quant_args, parallel_args, options));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    return mlp_(x);
  }

  void load_state_dict(const StateDict& state_dict) {
    mlp_->load_state_dict(state_dict.select("mlp."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    mlp_->verify_loaded_weights(prefix + "mlp.");
  }

 private:
  SimpleMLP mlp_{nullptr};
};

TORCH_MODULE(SimpleDecoderLayer);

class SimpleModelImpl : public torch::nn::Module {
 public:
  SimpleModelImpl(const ModelArgs& args,
                  const QuantArgs& quant_args,
                  const ParallelArgs& parallel_args,
                  const torch::TensorOptions& options) {
    embed_tokens_ = register_module(
        "embed_tokens",
        ParallelEmbedding(
            args.vocab_size(), args.hidden_size(), parallel_args, options));

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = SimpleDecoderLayer(args, quant_args, parallel_args, options);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = embed_tokens_(tokens);
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, kv_caches[i], input_params);
    }
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(state_dict.select("embed_tokens."));
    for (int i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          state_dict.select("layers." + std::to_string(i) + "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");
    for (int i = 0; i < layers_.size(); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
  }

 private:
  ParallelEmbedding embed_tokens_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<SimpleDecoderLayer> layers_;
};
TORCH_MODULE(SimpleModel);

class SimpleForCausalLMImpl : public torch::nn::Module {
 public:
  SimpleForCausalLMImpl(const ModelArgs& args,
                        const QuantArgs& quant_args,
                        const ParallelArgs& parallel_args,
                        const torch::TensorOptions& options) {
    model_ = register_module(
        "model", SimpleModel(args, quant_args, parallel_args, options));
  }

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {
    model_->load_state_dict(state_dict.select("model."));
  }

  void verify_loaded_weights() const {
    model_->verify_loaded_weights("model.");
  }

 private:
  SimpleModel model_{nullptr};
};
TORCH_MODULE(SimpleForCausalLM);
REGISTER_CAUSAL_MODEL(simple, SimpleForCausalLM);

REGISTER_MODEL_ARGS(simple, [&] {
  LOAD_ARG_OR(model_type, "model_type", "simple");
  LOAD_ARG_OR(dtype, "torch_type", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 32000);
  LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 1);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 32);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 11008);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 2048);
});

}  // namespace llm
