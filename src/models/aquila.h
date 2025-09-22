#pragma once

#include <torch/torch.h>

#include "chat_template/coded_chat_template.h"
#include "layers/activation.h"
#include "layers/attention/attention.h"
#include "layers/attention/handler.h"
#include "layers/embedding.h"
#include "layers/fused_linear.h"
#include "layers/linear.h"
#include "layers/normalization.h"
#include "memory/kv_cache.h"
#include "models/model_args.h"
#include "models/model_registry.h"
#include "models/parameters.h"
#include "module/module.h"
#include "module/module_holder.h"
#include "module/module_list.h"

// Aquila model compatible with huggingface weights
namespace llm::hf {

class AquilaMLPImpl : public Module {
 public:
  AquilaMLPImpl(const ModelArgs& args,
                const QuantArgs& quant_args,
                const ParallelArgs& parallel_args,
                const torch::TensorOptions& options) {
    act_func_ = Activation::get_act_func("silu", options.device());
    CHECK(act_func_ != nullptr);

    const int64_t hidden_size = args.hidden_size();
    const int64_t intermediate_size = args.intermediate_size();

    // register the weight parameter
    gate_up_proj_ = register_module(
        "gate_up_proj",
        FusedColumnParallelLinear(
            hidden_size,
            std::vector<int64_t>{intermediate_size, intermediate_size},
            /*bias=*/false,
            /*gather_output=*/false,
            quant_args,
            parallel_args,
            options));
    down_proj_ =
        register_module("down_proj",
                        RowParallelLinear(intermediate_size,
                                          hidden_size,
                                          /*bias=*/false,
                                          /*input_is_parallelized=*/true,
                                          quant_args,
                                          parallel_args,
                                          options));
  }

  torch::Tensor forward(torch::Tensor x) {
    const auto gate_up = gate_up_proj_(x);
    return down_proj_(act_func_(gate_up[0]) * gate_up[1]);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    gate_up_proj_->load_state_dict(state_dict, {"gate_proj.", "up_proj."});
    down_proj_->load_state_dict(state_dict.select("down_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    gate_up_proj_->verify_loaded_weights(prefix + "[gate_proj,up_proj].");
    down_proj_->verify_loaded_weights(prefix + "down_proj.");
  }

 private:
  // parameter members, must be registered
  FusedColumnParallelLinear gate_up_proj_{nullptr};
  RowParallelLinear down_proj_{nullptr};

  // activation function
  ActFunc act_func_{nullptr};
};
LLM_MODULE(AquilaMLP);

class AquilaAttentionImpl : public Module {
 public:
  AquilaAttentionImpl(const ModelArgs& args,
                      const QuantArgs& quant_args,
                      const ParallelArgs& parallel_args,
                      const torch::TensorOptions& options,
                      AttentionHandler* handler) {
    const int32_t world_size = parallel_args.world_size();
    const int64_t hidden_size = args.hidden_size();
    const int64_t n_heads = args.n_heads();
    const int64_t n_kv_heads = args.n_kv_heads().value_or(n_heads);
    const int64_t head_dim = args.head_dim();
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
                              RowParallelLinear(hidden_size,
                                                hidden_size,
                                                /*bias=*/false,
                                                /*input_is_parallelized=*/true,
                                                quant_args,
                                                parallel_args,
                                                options));

    // initialize attention
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
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

    // calculate attention, output: (num_tokens, n_local_heads * head_dim)
    auto output =
        atten_(qkv[0], qkv[1], qkv[2], positions, kv_cache, input_params);
    return o_proj_(output);
  }

  // load the weight from the checkpoint
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
LLM_MODULE(AquilaAttention);

class AquilaDecoderLayerImpl : public Module {
 public:
  AquilaDecoderLayerImpl(const ModelArgs& args,
                         const QuantArgs& quant_args,
                         const ParallelArgs& parallel_args,
                         const torch::TensorOptions& options,
                         AttentionHandler* handler) {
    // register submodules
    self_attn_ = register_module(
        "self_attn",
        AquilaAttention(args, quant_args, parallel_args, options, handler));
    mlp_ = register_module("mlp",
                           AquilaMLP(args, quant_args, parallel_args, options));
    input_layernorm_ = register_module(
        "input_layernorm",
        RMSNorm(args.hidden_size(), args.rms_norm_eps(), options));
    post_attention_layernorm_ = register_module(
        "post_attention_layernorm",
        RMSNorm(args.hidden_size(), args.rms_norm_eps(), options));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    auto h =
        x + self_attn_(input_layernorm_(x), positions, kv_cache, input_params);
    return h + mlp_(post_attention_layernorm_(h));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    self_attn_->load_state_dict(state_dict.select("self_attn."));
    mlp_->load_state_dict(state_dict.select("mlp."));
    input_layernorm_->load_state_dict(state_dict.select("input_layernorm."));
    post_attention_layernorm_->load_state_dict(
        state_dict.select("post_attention_layernorm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    self_attn_->verify_loaded_weights(prefix + "self_attn.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    input_layernorm_->verify_loaded_weights(prefix + "input_layernorm.");
    post_attention_layernorm_->verify_loaded_weights(
        prefix + "post_attention_layernorm.");
  }

 private:
  // parameter members, must be registered
  AquilaAttention self_attn_{nullptr};

  AquilaMLP mlp_{nullptr};

  RMSNorm input_layernorm_{nullptr};

  RMSNorm post_attention_layernorm_{nullptr};
};
LLM_MODULE(AquilaDecoderLayer);

class AquilaModelImpl : public Module {
 public:
  AquilaModelImpl(const ModelArgs& args,
                  const QuantArgs& quant_args,
                  const ParallelArgs& parallel_args,
                  const torch::TensorOptions& options) {
    // register submodules
    embed_tokens_ = register_module(
        "embed_tokens",
        ParallelEmbedding(
            args.vocab_size(), args.hidden_size(), parallel_args, options));

    handler_ = AttentionHandler::create_handler_with_rope(
        args, /*interleaved=*/false, options);

    blocks_ = register_module("layers", ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = AquilaDecoderLayer(
          args, quant_args, parallel_args, options, handler_.get());
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    norm_ = register_module(
        "norm", RMSNorm(args.hidden_size(), args.rms_norm_eps(), options));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = embed_tokens_(tokens);
    // TODO: set working space for attention handler
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, kv_caches[i], input_params);
    }
    return norm_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(state_dict.select("embed_tokens."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.select("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.select("norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

 private:
  // parameter members, must be registered
  ParallelEmbedding embed_tokens_{nullptr};

  // attention handler
  std::unique_ptr<AttentionHandler> handler_{nullptr};

  ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<AquilaDecoderLayer> layers_;

  RMSNorm norm_{nullptr};
};
LLM_MODULE(AquilaModel);

class AquilaForCausalLMImpl : public Module {
 public:
  AquilaForCausalLMImpl(const ModelArgs& args,
                        const QuantArgs& quant_args,
                        const ParallelArgs& parallel_args,
                        const torch::TensorOptions& options) {
    // register submodules
    model_ = register_module(
        "model", AquilaModel(args, quant_args, parallel_args, options));

    lm_head_ = register_module("lm_head",
                               ColumnParallelLinear(args.hidden_size(),
                                                    args.vocab_size(),
                                                    /*bias=*/false,
                                                    /*gather_output=*/true,
                                                    parallel_args,
                                                    options));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    // select tokens if provided
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return lm_head_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    model_->load_state_dict(state_dict.select("model."));
    lm_head_->load_state_dict(state_dict.select("lm_head."));
  }

  void verify_loaded_weights() const {
    model_->verify_loaded_weights("model.");
    lm_head_->verify_loaded_weights("lm_head.");
  }

 private:
  // parameter members, must be registered
  AquilaModel model_{nullptr};

  ColumnParallelLinear lm_head_{nullptr};
};
LLM_MODULE(AquilaForCausalLM);

class AquilaChatTemplate final : public CodedChatTemplate {
 public:
  // generate prompt from dialogs
  // Prompt template for Aquila:
  // System: A chat between a curious ...
  // Human: {prompt}
  // Assistant:
  std::optional<std::string> get_prompt(
      const std::string_view& system_message,
      const std::vector<std::string_view>& messages) const override {
    // at least one user message
    if (messages.size() % 2 == 0) {
      return std::nullopt;
    }

    std::stringstream ss;
    // start with system message
    if (!system_message.empty()) {
      ss << "System: " << system_message << "\n";
    }

    // then user and assistant message pairs (u/a/u/a/u...)
    for (size_t i = 0; i < messages.size(); ++i) {
      const char* role = (i % 2) == 0 ? "Human: " : "Assistant: ";
      ss << role << messages[i] << "\n";
    }
    // end with assistant message
    ss << "Assistant:";
    return ss.str();
  }
};

// register the model to make it available
REGISTER_CAUSAL_MODEL(aquila, AquilaForCausalLM);
REGISTER_DEFAULT_CHAT_TEMPLATE(aquila, AquilaChatTemplate);
REGISTER_MODEL_ARGS(aquila, [&] {
  // example config:
  // https://huggingface.co/BAAI/Aquila-7B/blob/main/config.json.
  // set default values for args explicitly with values from:
  // https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/configuration_mistral.py#L104
  LOAD_ARG_OR(model_type, "model_type", "aquila");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 100008);
  LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 32);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 11008);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 2048);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 1);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
});
}  // namespace llm::hf
