#pragma once

#include <torch/torch.h>

#include "layers/activation.h"
#include "layers/attention_rope.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/normalization.h"
#include "memory/kv_cache.h"
#include "models/input_parameters.h"
#include "models/model_args.h"
#include "models/model_registry.h"

// Phi model compatible with huggingface weights

namespace llm::hf {

class PhiMLPImpl : public torch::nn::Module {
 public:
  PhiMLPImpl(const ModelArgs& args,
             const QuantArgs& quant_args,
             const ParallelArgs& parallel_args,
             torch::ScalarType dtype,
             const torch::Device& device) {
    const int64_t hidden_size = args.hidden_size();
    const int64_t intermediate_size = args.intermediate_size();

    act_ = Activation::get_act_func(args.hidden_act(), device);
    CHECK(act_ != nullptr);

    // register the weight parameter
    fc1_ = register_module("fc1",
                           ColumnParallelLinear(hidden_size,
                                                intermediate_size,
                                                /*bias=*/true,
                                                /*gather_output=*/false,
                                                quant_args,
                                                parallel_args,
                                                dtype,
                                                device));
    fc2_ = register_module("fc2",
                           RowParallelLinear(intermediate_size,
                                             hidden_size,
                                             /*bias=*/true,
                                             /*input_is_parallelized=*/true,
                                             quant_args,
                                             parallel_args,
                                             dtype,
                                             device));
  }

  torch::Tensor forward(torch::Tensor x) { return fc2_(act_(fc1_(x))); }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    fc1_->load_state_dict(state_dict.select("fc1."));
    fc2_->load_state_dict(state_dict.select("fc2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    fc1_->verify_loaded_weights(prefix + "fc1.");
    fc2_->verify_loaded_weights(prefix + "fc2.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear fc1_{nullptr};
  RowParallelLinear fc2_{nullptr};

  ActFunc act_{nullptr};
};
TORCH_MODULE(PhiMLP);

class PhiAttentionImpl : public torch::nn::Module {
 public:
  PhiAttentionImpl(const ModelArgs& args,
                   const QuantArgs& quant_args,
                   const ParallelArgs& parallel_args,
                   torch::ScalarType dtype,
                   const torch::Device& device) {
    const int32_t world_size = parallel_args.world_size();
    const int64_t hidden_size = args.hidden_size();
    const int64_t n_heads = args.n_heads();
    const int64_t n_kv_heads = args.n_kv_heads().value_or(n_heads);
    const int64_t head_dim = hidden_size / n_heads;
    const int64_t n_local_heads = n_heads / world_size;
    const int64_t n_local_kv_heads = n_kv_heads / world_size;

    // size for q, k, v
    qkv_sizes_ = {n_local_heads * head_dim,
                  n_local_kv_heads * head_dim,
                  n_local_kv_heads * head_dim};

    // register submodules
    Wqkv_ = register_module(
        "Wqkv",
        ColumnParallelLinear(hidden_size,
                             (n_heads + 2 * n_kv_heads) * head_dim,
                             /*bias=*/true,
                             /*gather_output=*/false,
                             quant_args,
                             parallel_args,
                             dtype,
                             device));

    out_proj_ =
        register_module("out_proj",
                        RowParallelLinear(hidden_size,
                                          hidden_size,
                                          /*bias=*/true,
                                          /*input_is_parallelized=*/true,
                                          quant_args,
                                          parallel_args,
                                          dtype,
                                          device));

    // initialize attention
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    atten_ = register_module("atten",
                             AttentionWithRoPE(n_local_heads,
                                               n_local_kv_heads,
                                               head_dim,
                                               scale,
                                               args.rotary_dim(),
                                               args.rope_scaling(),
                                               args.rope_theta(),
                                               args.max_position_embeddings(),
                                               /*interleaved=*/false,
                                               dtype,
                                               device));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    // (num_tokens, dim) x (dim, n_local_heads * head_dim)
    // => (num_tokens, n_local_heads * head_dim)
    auto qkv = Wqkv_(x).split(/*split_size=*/qkv_sizes_, /*dim=*/-1);
    DCHECK_EQ(qkv.size(), 3);

    // calculate attention, output: (num_tokens, n_local_heads * head_dim)
    auto output =
        atten_(qkv[0], qkv[1], qkv[2], positions, kv_cache, input_params);
    return out_proj_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    Wqkv_->load_state_dict(state_dict.select("Wqkv."));
    out_proj_->load_state_dict(state_dict.select("out_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    Wqkv_->verify_loaded_weights(prefix + "Wqkv.");
    out_proj_->verify_loaded_weights(prefix + "out_proj.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear Wqkv_{nullptr};

  RowParallelLinear out_proj_{nullptr};

  // module members without parameters
  AttentionWithRoPE atten_{nullptr};

  // size for q, k, v
  std::vector<int64_t> qkv_sizes_;
};
TORCH_MODULE(PhiAttention);

class PhiBlockImpl : public torch::nn::Module {
 public:
  PhiBlockImpl(const ModelArgs& args,
               const QuantArgs& quant_args,
               const ParallelArgs& parallel_args,
               torch::ScalarType dtype,
               const torch::Device& device) {
    // register submodules
    mixer_ = register_module(
        "mixer", PhiAttention(args, quant_args, parallel_args, dtype, device));
    mlp_ = register_module(
        "mlp", PhiMLP(args, quant_args, parallel_args, dtype, device));
    ln_ = register_module("ln",
                          LayerNorm(args.hidden_size(),
                                    args.layer_norm_eps(),
                                    /*bias=*/true,
                                    dtype,
                                    device));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    // x = x + attn(ln(x)) + mlp(ln(x))
    const auto h = ln_(x);
    const auto attn_output = mixer_(h, positions, kv_cache, input_params);
    const auto mlp_output = mlp_(h);
    return x + attn_output + mlp_output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    mixer_->load_state_dict(state_dict.select("mixer."));
    mlp_->load_state_dict(state_dict.select("mlp."));
    ln_->load_state_dict(state_dict.select("ln."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    mixer_->verify_loaded_weights(prefix + "mixer.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    ln_->verify_loaded_weights(prefix + "ln.");
  }

 private:
  // parameter members, must be registered
  PhiAttention mixer_{nullptr};

  PhiMLP mlp_{nullptr};

  LayerNorm ln_{nullptr};
};
TORCH_MODULE(PhiBlock);

class PhiModelImpl : public torch::nn::Module {
 public:
  PhiModelImpl(const ModelArgs& args,
               const QuantArgs& quant_args,
               const ParallelArgs& parallel_args,
               torch::ScalarType dtype,
               const torch::Device& device) {
    // register submodules
    wte_ = register_module("wte",
                           ParallelEmbedding(args.vocab_size(),
                                             args.hidden_size(),
                                             parallel_args,
                                             dtype,
                                             device));

    blocks_ = register_module("h", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = PhiBlock(args, quant_args, parallel_args, dtype, device);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = wte_(tokens);
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, kv_caches[i], input_params);
    }
    return h;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    wte_->load_state_dict(state_dict.select("embd.wte."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.select("h." + std::to_string(i) + "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    wte_->verify_loaded_weights(prefix + "embd.wte.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "h." + std::to_string(i) +
                                        ".");
    }
  }

 private:
  // parameter members, must be registered
  ParallelEmbedding wte_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<PhiBlock> layers_;
};
TORCH_MODULE(PhiModel);

class PhiLMHeadImpl : public torch::nn::Module {
 public:
  PhiLMHeadImpl(const ModelArgs& args,
                const ParallelArgs& parallel_args,
                torch::ScalarType dtype,
                const torch::Device& device) {
    // register submodules
    ln_ = register_module("ln",
                          LayerNorm(args.hidden_size(),
                                    args.layer_norm_eps(),
                                    /*bias=*/true,
                                    dtype,
                                    device));

    linear_ = register_module("linear",
                              ColumnParallelLinear(args.hidden_size(),
                                                   args.vocab_size(),
                                                   /*bias=*/true,
                                                   /*gather_output=*/true,
                                                   parallel_args,
                                                   dtype,
                                                   device));
  }

  torch::Tensor forward(torch::Tensor x) { return linear_(ln_(x)); }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    ln_->load_state_dict(state_dict.select("ln."));
    linear_->load_state_dict(state_dict.select("linear."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    ln_->verify_loaded_weights(prefix + "ln.");
    linear_->verify_loaded_weights(prefix + "linear.");
  }

 private:
  // parameter members, must be registered
  LayerNorm ln_{nullptr};

  ColumnParallelLinear linear_{nullptr};
};
TORCH_MODULE(PhiLMHead);

class PhiForCausalLMImpl : public torch::nn::Module {
 public:
  PhiForCausalLMImpl(const ModelArgs& args,
                     const QuantArgs& quant_args,
                     const ParallelArgs& parallel_args,
                     torch::ScalarType dtype,
                     const torch::Device& device) {
    // register submodules
    transformer_ = register_module(
        "transformer",
        PhiModel(args, quant_args, parallel_args, dtype, device));

    lm_head_ = register_module("lm_head",
                               PhiLMHead(args, parallel_args, dtype, device));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = transformer_(tokens, positions, kv_caches, input_params);
    // select last token for each sequence
    h = h.index_select(/*dim=*/0, input_params.last_token_idxes);
    return lm_head_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    transformer_->load_state_dict(state_dict.select("transformer."));
    lm_head_->load_state_dict(state_dict.select("lm_head."));
  }

  void verify_loaded_weights() const {
    transformer_->verify_loaded_weights("transformer.");
    lm_head_->verify_loaded_weights("lm_head.");
  }

 private:
  // parameter members, must be registered
  PhiModel transformer_{nullptr};

  PhiLMHead lm_head_{nullptr};
};
TORCH_MODULE(PhiForCausalLM);

// clang-format off
// register the model to make it available
REGISTER_CAUSAL_MODEL_WITH_VARNAME(phi, phi-msft, PhiForCausalLM);
REGISTER_MODEL_ARGS_WITH_VARNAME(phi, phi-msft, [&] {
  // example config:
  // https://huggingface.co/microsoft/phi-2/blob/main/config.json
  LOAD_ARG_OR(model_type, "model_type", "phi-msft");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 51200);
  LOAD_ARG_OR(hidden_size, "n_embd", 2560);
  LOAD_ARG_OR(n_layers, "n_layer", 32);
  LOAD_ARG_OR(n_heads, "n_head", 32);
  LOAD_ARG(n_kv_heads, "n_head_kv");
  LOAD_ARG_OR(rotary_dim, "rotary_dim", 32);
  LOAD_ARG_OR(hidden_act, "activation_function", "gelu_new");
  LOAD_ARG_OR(max_position_embeddings, "n_positions", 2048);
  LOAD_ARG_OR(layer_norm_eps, "layer_norm_epsilon", 1e-5);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 50256);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 50256);

  LOAD_ARG_OR_FUNC(intermediate_size, "n_inner", [&] {
    // set it to 4 times n_embd
    return args->hidden_size() * 4;
  });
});
// clang-format on
}  // namespace llm::hf
