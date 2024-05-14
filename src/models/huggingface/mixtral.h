#pragma once
#include <torch/torch.h>

#include "chat_template/coded_chat_template.h"
#include "layers/activation.h"
#include "layers/attention/attention.h"
#include "layers/attention/handler.h"
#include "layers/embedding.h"
#include "layers/fused_moe.h"
#include "layers/linear.h"
#include "layers/linear_impl.h"
#include "layers/normalization.h"
#include "memory/kv_cache.h"
#include "models/model_args.h"
#include "models/model_registry.h"
#include "models/parameters.h"

namespace llm::hf {

class MixtralMoEImpl : public torch::nn::Module {
 public:
  MixtralMoEImpl(const ModelArgs& args,
                 const QuantArgs& quant_args,
                 const ParallelArgs& parallel_args,
                 const torch::TensorOptions& options) {
    args_ = args;
    gate_ = register_module("gate",
                            ReplicatedLinear(args_.hidden_size(),
                                             args_.n_local_experts(),
                                             /*bias*/ false,
                                             /*skip_bias_add*/ false,
                                             quant_args,
                                             options));
    fused_moe_ = register_module("fused_moe",
                                 FusedMoeLayer(
                                     /*renormalize*/ true,
                                     /*inplace*/ true,
                                     args,
                                     quant_args,
                                     parallel_args,
                                     options));
  }

  torch::Tensor forward(torch::Tensor hidden_states) {
    auto sizes = hidden_states.sizes();
    auto num_token = sizes[0];
    auto hidden_size = sizes[1];
    hidden_states = hidden_states.view({-1, hidden_size});
    torch::Tensor out_bias;
    auto router_logits = gate_(hidden_states, out_bias);
    // router_logits: (num_tokens, n_experts)
    auto final_hidden_states = fused_moe_(hidden_states, router_logits);

    /*
    if self.tp_size > 1: # 张量并行的做法
          final_hidden_states = tensor_model_parallel_all_reduce(
              final_hidden_states)
    */
    return final_hidden_states.view({num_token, hidden_size});
  }
  void load_state_dict(const StateDict& state_dict) {
    gate_->load_state_dict(state_dict.select("gate."));
    fused_moe_->load_state_dict(state_dict.select("experts."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    gate_->verify_loaded_weights(prefix + "gate.");
    fused_moe_->verify_loaded_weights(prefix + "experts.");
  }

 private:
  ModelArgs args_;

  ReplicatedLinear gate_{nullptr};
  FusedMoeLayer fused_moe_{nullptr};
};
TORCH_MODULE(MixtralMoE);

class MixtralAttentionImpl : public torch::nn::Module {
 public:
  MixtralAttentionImpl(const ModelArgs& args,
                       const QuantArgs& quant_args,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options,
                       AttentionHandler* handler) {
    const int32_t world_size = parallel_args.world_size();
    const int64_t hidden_size = args.hidden_size();
    const int64_t n_heads = args.n_heads();
    const int64_t head_dim = args.head_dim();
    const int64_t n_kv_heads = args.n_kv_heads().value_or(n_heads);
    const int64_t n_local_heads = n_heads / world_size;
    const int64_t n_local_kv_heads = n_kv_heads / world_size;

    // size for q, k, v
    qkv_sizes_ = {n_local_heads * head_dim,
                  n_local_kv_heads * head_dim,
                  n_local_kv_heads * head_dim};

    // register submodules
    qkv_proj_ = register_module(
        "qkv_proj",
        ColumnParallelLinear(hidden_size,
                             (n_heads + 2 * n_kv_heads) * head_dim,
                             /*bias=*/false,
                             /*gather_output=*/false,
                             quant_args,
                             parallel_args,
                             options));

    o_proj_ = register_module("o_proj",
                              RowParallelLinear(n_heads * head_dim,
                                                hidden_size,
                                                /*bias=*/false,
                                                /*input_is_parallelized=*/true,
                                                quant_args,
                                                parallel_args,
                                                options));

    // initialize attention
    atten_ = register_module(
        "atten", Attention(n_local_heads, n_local_kv_heads, head_dim, handler));
  }
  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    // (num_tokens, dim) x (dim, n_local_heads * head_dim)
    // => (num_tokens, n_local_heads * head_dim)
    auto qkv = qkv_proj_(x).split(/*split_size=*/qkv_sizes_, /*dim=*/-1);
    DCHECK_EQ(qkv.size(), 3);

    // calculate attention,
    // output: (num_tokens, n_local_heads*head_dim)
    auto output =
        atten_(qkv[0], qkv[1], qkv[2], positions, kv_cache, input_params);
    return o_proj_(output);
  }

  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    qkv_proj_->load_state_dict(state_dict, {"q_proj.", "k_proj.", "v_proj."});
    o_proj_->load_state_dict(state_dict.select("o_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    qkv_proj_->verify_loaded_weights(prefix + "[q_proj,k_proj,v_proj].");
    o_proj_->verify_loaded_weights(prefix + "o_proj.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear qkv_proj_{nullptr};

  RowParallelLinear o_proj_{nullptr};

  // module members without parameters
  Attention atten_{nullptr};

  // size for q, k, v
  std::vector<int64_t> qkv_sizes_;
};
TORCH_MODULE(MixtralAttention);

class MixtralDecoderLayerImpl : public torch::nn::Module {
 public:
  MixtralDecoderLayerImpl(const ModelArgs& args,
                          const QuantArgs& quant_args,
                          const ParallelArgs& parallel_args,
                          const torch::TensorOptions& options,
                          AttentionHandler* handler) {
    // register submodules
    self_attn_ = register_module(
        "self_attn",
        MixtralAttention(args, quant_args, parallel_args, options, handler));

    moe_ = register_module(
        "moe", MixtralMoE(args, quant_args, parallel_args, options));

    input_layernorm_ = register_module(
        "input_layernorm",
        RMSNormResidual(args.hidden_size(), args.rms_norm_eps(), options));

    post_attention_layernorm_ = register_module(
        "post_attention_layernorm",
        RMSNormResidual(args.hidden_size(), args.rms_norm_eps(), options));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params,
                        torch::Tensor& residual) {
    auto hidden_states = input_layernorm_(x, residual);

    hidden_states =
        self_attn_(hidden_states, positions, kv_cache, input_params);

    // fully connected
    hidden_states = post_attention_layernorm_(hidden_states, residual);

    return moe_(hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    self_attn_->load_state_dict(state_dict.select("self_attn."));
    input_layernorm_->load_state_dict(state_dict.select("input_layernorm."));
    post_attention_layernorm_->load_state_dict(
        state_dict.select("post_attention_layernorm."));
    moe_->load_state_dict(state_dict.select("block_sparse_moe."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    self_attn_->verify_loaded_weights(prefix + "self_attn.");
    input_layernorm_->verify_loaded_weights(prefix + "input_layernorm.");
    post_attention_layernorm_->verify_loaded_weights(
        prefix + "post_attention_layernorm.");
    moe_->verify_loaded_weights(prefix + "block_sparse_moe.");
  }

 private:
  MixtralAttention self_attn_{nullptr};

  MixtralMoE moe_{nullptr};

  RMSNormResidual input_layernorm_{nullptr};

  RMSNormResidual post_attention_layernorm_{nullptr};
};
TORCH_MODULE(MixtralDecoderLayer);

class MixtralModelImpl : public torch::nn::Module {
 public:
  MixtralModelImpl(const ModelArgs& args,
                   const QuantArgs& quant_args,
                   const ParallelArgs& parallel_args,
                   const torch::TensorOptions& options) {
    modelArgs_ = args;

    // TODO: If we have implemented the lora, the vocab_size should be
    // processed.
    embed_tokens_ = register_module(
        "embed_tokens",
        ParallelEmbedding(
            args.vocab_size(), args.hidden_size(), parallel_args, options));

    handler_ = AttentionHandler::create_handler_with_rope(
        args, /*interleaved=*/false, options);

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = MixtralDecoderLayer(
          args, quant_args, parallel_args, options, handler_.get());
      layers_.push_back(block);
      blocks_->push_back(block);
    }

    norm_ = register_module(
        "norm",
        RMSNormResidual(args.hidden_size(), args.rms_norm_eps(), options));
  }

  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = embed_tokens_(tokens);

    torch::Tensor residual;
    for (int32_t i = 0; i < modelArgs_.n_layers(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, kv_caches[i], input_params, residual);
    }

    return norm_(h, residual);
  }

  void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(state_dict.select("embed_tokens.weight"));

    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.select("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.select("norm.weight"));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.weight");

    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }

    norm_->verify_loaded_weights(prefix + "norm.weight");
  }

 private:
  ModelArgs modelArgs_;
  // parameter members, must be registered
  // embedding module
  ParallelEmbedding embed_tokens_{nullptr};

  RMSNormResidual norm_{nullptr};

  // attention handler
  std::unique_ptr<AttentionHandler> handler_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<MixtralDecoderLayer> layers_{nullptr};
};
TORCH_MODULE(MixtralModel);

class MixtralForCausalLMImpl : public torch::nn::Module {
 public:
  MixtralForCausalLMImpl(const ModelArgs& args,
                         const QuantArgs& quant_args,
                         const ParallelArgs& parallel_args,
                         const torch::TensorOptions& options) {
    model_ = register_module(
        "model", MixtralModel(args, quant_args, parallel_args, options));

    // TODO: we can need the lora flag in the future
    lm_head_ = register_module("lm_head",
                               ColumnParallelLinear(args.hidden_size(),
                                                    args.vocab_size(),
                                                    /*bias=*/false,
                                                    /*gather_output=*/true,
                                                    parallel_args,
                                                    options));
  }

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    // select tokens if provided
    auto h = hidden_states;
    if (selected_idxes.defined()) {
      h = h.index_select(/*dim=*/0, selected_idxes);
    }
    return lm_head_(h);
  }

  void load_state_dict(const StateDict& state_dict) {
    model_->load_state_dict(state_dict.select("model."));

    lm_head_->load_state_dict(state_dict.select("lm_head."));
  }

  void verify_loaded_weights() const {
    model_->verify_loaded_weights("model.");
    lm_head_->verify_loaded_weights("lm_head.");
  }

 private:
  MixtralModel model_{nullptr};

  ColumnParallelLinear lm_head_{nullptr};
};
TORCH_MODULE(MixtralForCausalLM);

// register the model to make it available
REGISTER_CAUSAL_MODEL(mixtral, MixtralForCausalLM);

REGISTER_MODEL_ARGS(mixtral, [&] {
  // example config from huggingface
  // https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
  LOAD_ARG_OR(model_type, "model_type", "mixtral");
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 1);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
  LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 14336);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(n_experts_per_tok, "num_experts_per_tok", 2);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 32);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 8);
  LOAD_ARG_OR(n_local_experts, "num_local_experts", 8);
  LOAD_ARG_OR(out_router_logits, "output_router_logits", false);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(router_aux_loss_coef, "router_aux_loss_coef", 0.02);
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(vocab_size, "vocab_size", 32000);

  LOAD_ARG_OR(hidden_act, "hidden_activation", "silu");

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
});

}  // namespace llm::hf