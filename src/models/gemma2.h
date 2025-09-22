#pragma once
#include <absl/strings/match.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <string>

#include "chat_template/coded_chat_template.h"
#include "layers/activation.h"
#include "layers/attention/attention.h"
#include "layers/attention/handler.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/normalization.h"
#include "layers/qkv_linear.h"
#include "memory/kv_cache.h"
#include "models/model_args.h"
#include "models/model_registry.h"
#include "models/parameters.h"
#include "module/module.h"
#include "module/module_holder.h"
#include "module/modulelist.h"

// Gemma2 model compatible with huggingface weight
namespace llm::hf {

class Gemma2MLPImpl : public llm::nn::Module {
 public:
  Gemma2MLPImpl(const ModelArgs& args,
                const QuantArgs& quant_args,
                const ParallelArgs& parallel_args,
                const torch::TensorOptions& options) {
    act_func_ = Activation::get_act_func(args.hidden_act(), options.device());
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
LLM_MODULE(Gemma2MLP);

class Gemma2AttentionImpl : public llm::nn::Module {
 public:
  Gemma2AttentionImpl(const ModelArgs& args,
                      const QuantArgs& quant_args,
                      const ParallelArgs& parallel_args,
                      const torch::TensorOptions& options,
                      AttentionHandler* handler,
                      int32_t sliding_window) {
    const int32_t world_size = parallel_args.world_size();
    const int64_t hidden_size = args.hidden_size();
    const int64_t n_heads = args.n_heads();
    const int64_t n_kv_heads = args.n_kv_heads().value_or(n_heads);
    const int64_t head_dim = args.head_dim();
    const int64_t n_local_heads = n_heads / world_size;
    const int64_t n_local_kv_heads =
        std::max<int64_t>(1, n_kv_heads / world_size);

    // register submodules
    qkv_proj_ = register_module("qkv_proj",
                                QKVColumnParallelLinear(hidden_size,
                                                        n_heads,
                                                        n_kv_heads,
                                                        head_dim,
                                                        args.attn_bias(),
                                                        /*gather_output=*/false,
                                                        quant_args,
                                                        parallel_args,
                                                        options));

    o_proj_ = register_module("o_proj",
                              RowParallelLinear(n_heads * head_dim,
                                                hidden_size,
                                                args.attn_bias(),
                                                /*input_is_parallelized=*/true,
                                                quant_args,
                                                parallel_args,
                                                options));

    // initialize attention
    atten_ = register_module("atten",
                             Attention(n_local_heads,
                                       n_local_kv_heads,
                                       head_dim,
                                       handler,
                                       sliding_window));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    // (num_tokens, dim) x (dim, n_local_heads * head_dim)
    // => (num_tokens, n_local_heads * head_dim)
    const auto qkv = qkv_proj_(x);
    // calculate attention,
    // output: (num_tokens, n_local_heads*head_dim)
    const auto output =
        atten_(qkv[0], qkv[1], qkv[2], positions, kv_cache, input_params);
    return o_proj_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    qkv_proj_->load_state_dict(
        state_dict, {"q_proj.", "k_proj.", "v_proj."}, {"k_proj.", "v_proj."});
    o_proj_->load_state_dict(state_dict.select("o_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    qkv_proj_->verify_loaded_weights(prefix + "[q_proj,k_proj,v_proj].");
    o_proj_->verify_loaded_weights(prefix + "o_proj.");
  }

 private:
  // parameter members, must be registered
  QKVColumnParallelLinear qkv_proj_{nullptr};

  RowParallelLinear o_proj_{nullptr};

  // module members without parameters
  Attention atten_{nullptr};

  // size for q, k, v
  std::vector<int64_t> qkv_sizes_;
};
LLM_MODULE(Gemma2Attention);

class Gemma2DecoderLayerImpl : public llm::nn::Module {
 public:
  Gemma2DecoderLayerImpl(const ModelArgs& args,
                         const QuantArgs& quant_args,
                         const ParallelArgs& parallel_args,
                         const torch::TensorOptions& options,
                         AttentionHandler* handler,
                         int32_t sliding_window) {
    // register submodules
    self_attn_ = register_module(
        "self_attn",
        Gemma2Attention(
            args, quant_args, parallel_args, options, handler, sliding_window));

    mlp_ = register_module("mlp",
                           Gemma2MLP(args, quant_args, parallel_args, options));

    input_layernorm_ = register_module(
        "input_layernorm",
        GemmaRMSNorm(args.hidden_size(), args.rms_norm_eps(), options));

    post_attention_layernorm_ = register_module(
        "post_attention_layernorm",
        GemmaRMSNorm(args.hidden_size(), args.rms_norm_eps(), options));

    pre_feedforward_layernorm_ = register_module(
        "pre_feedforward_layernorm",
        GemmaRMSNorm(args.hidden_size(), args.rms_norm_eps(), options));

    post_feedforward_layernorm_ = register_module(
        "post_feedforward_layernorm",
        GemmaRMSNorm(args.hidden_size(), args.rms_norm_eps(), options));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    auto residual = x;
    auto hidden_states = input_layernorm_(x);

    hidden_states =
        self_attn_(hidden_states, positions, kv_cache, input_params);
    hidden_states = post_attention_layernorm_(hidden_states);
    hidden_states += residual;

    residual = hidden_states;
    hidden_states = pre_feedforward_layernorm_(hidden_states);
    hidden_states = mlp_(hidden_states);
    hidden_states = post_feedforward_layernorm_(hidden_states);
    hidden_states += residual;
    return hidden_states;
  }
  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    input_layernorm_->load_state_dict((state_dict.select("input_layernorm.")));
    self_attn_->load_state_dict(state_dict.select("self_attn."));
    mlp_->load_state_dict(state_dict.select("mlp."));
    post_attention_layernorm_->load_state_dict(
        (state_dict.select("post_attention_layernorm.")));
    pre_feedforward_layernorm_->load_state_dict(
        state_dict.select("pre_feedforward_layernorm."));
    post_feedforward_layernorm_->load_state_dict(
        state_dict.select("post_feedforward_layernorm."));
  }
  void verify_loaded_weights(const std::string& prefix) const {
    input_layernorm_->verify_loaded_weights(prefix + "input_layernorm.");
    self_attn_->verify_loaded_weights(prefix + "self_attn.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    post_attention_layernorm_->verify_loaded_weights(
        prefix + "post_attention_layernorm.");
    pre_feedforward_layernorm_->verify_loaded_weights(
        prefix + "pre_feedforward_layernorm.");
    post_feedforward_layernorm_->verify_loaded_weights(
        prefix + "post_feedforward_layernorm.");
  }

 private:
  Gemma2Attention self_attn_{nullptr};

  Gemma2MLP mlp_{nullptr};

  GemmaRMSNorm input_layernorm_{nullptr};

  GemmaRMSNorm post_attention_layernorm_{nullptr};

  GemmaRMSNorm pre_feedforward_layernorm_{nullptr};

  GemmaRMSNorm post_feedforward_layernorm_{nullptr};
};
LLM_MODULE(Gemma2DecoderLayer);

class Gemma2ModelImpl : public llm::nn::Module {
 public:
  Gemma2ModelImpl(const ModelArgs& args,
                  const QuantArgs& quant_args,
                  const ParallelArgs& parallel_args,
                  const torch::TensorOptions& options) {
    modelArgs_ = args;
    // register submodules
    embed_tokens_ = register_module(
        "embed_tokens",
        ParallelEmbedding(
            args.vocab_size(), args.hidden_size(), parallel_args, options));

    // normalize the embedding by sqrt(hidden_size)
    // N.B. the data type of the normalizer should be the same as the embedding
    const float normalizer = std::sqrt(args.hidden_size());
    normalizer_ =
        register_buffer("normalizer", torch::tensor({normalizer}, options));

    norm_ = register_module(
        "norm", GemmaRMSNorm(args.hidden_size(), args.rms_norm_eps(), options));

    handler_ = AttentionHandler::create_handler_with_rope(
        args, /*interleaved=*/false, options);

    blocks_ = register_module("layers", llm::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      // Attention Type: [LOCAL_SLIDING, Global, LOCAL_SLIDING, Global, ...]
      // https://github.com/google/gemma_pytorch/blob/main/gemma/config.py#L118
      const int32_t sliding_window = (i % 2) == 0 ? args.sliding_window() : -1;
      auto block = Gemma2DecoderLayer(args,
                                      quant_args,
                                      parallel_args,
                                      options,
                                      handler_.get(),
                                      sliding_window);
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
    // embedding tokens
    auto h = embed_tokens_(tokens) * normalizer_;
    for (int32_t i = 0; i < modelArgs_.n_layers(); i++) {
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
    norm_->load_state_dict((state_dict.select("norm.")));
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
  ModelArgs modelArgs_;

  // parameter members, must be registered
  // embedding module
  ParallelEmbedding embed_tokens_{nullptr};

  // embedding normalizer
  torch::Tensor normalizer_{nullptr};

  GemmaRMSNorm norm_{nullptr};
  // attention handler
  std::unique_ptr<AttentionHandler> handler_{nullptr};

  llm::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<Gemma2DecoderLayer> layers_;
};
LLM_MODULE(Gemma2Model);

class Gemma2ForCausalLMImpl : public llm::nn::Module {
 public:
  Gemma2ForCausalLMImpl(const ModelArgs& args,
                        const QuantArgs& quant_args,
                        const ParallelArgs& parallel_args,
                        const torch::TensorOptions& options)
      : final_logit_soft_cap_(args.final_logit_soft_cap()) {
    CHECK_NE(final_logit_soft_cap_, 0.0f);

    // register submodules
    model_ = register_module(
        "model", Gemma2Model(args, quant_args, parallel_args, options));

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
  // return: [num_tokens,hidden_size]
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
    // apply final_logit_softcapping
    return torch::tanh(lm_head_(h) / final_logit_soft_cap_) *
           final_logit_soft_cap_;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    model_->load_state_dict(state_dict.select("model."));

    // Share the embedding weights with the final llm_head layer.
    lm_head_->load_state_dict(state_dict.select("model.embed_tokens."));
  }

  void verify_loaded_weights() const {
    model_->verify_loaded_weights("model.");
    lm_head_->verify_loaded_weights("model.embed_tokens.");
  }

 private:
  // parameter members, must be registered
  Gemma2Model model_{nullptr};

  ColumnParallelLinear lm_head_{nullptr};

  float final_logit_soft_cap_{0.0f};
};
LLM_MODULE(Gemma2ForCausalLM);

class Gemma2ChatTemplate final : public CodedChatTemplate {
 public:
  std::optional<std::string> get_prompt(
      const std::string_view& /*system_message*/,
      const std::vector<std::string_view>& messages) const override {
    // ignore system message since it's not supported by the model
    // at least one user message
    if (messages.size() % 2 == 0) {
      return std::nullopt;
    }

    // <start_of_turn>{role}\n{message}<end_of_turn>\n
    // <start_of_turn>model\n
    std::stringstream ss;
    for (size_t i = 0; i < messages.size(); i++) {
      const char* role = (i % 2) == 0 ? "user" : "model";
      ss << "<start_of_turn>" << role << "\n"
         << messages[i] << "<end_of_turn>\n";
    }

    ss << "<start_of_turn>model\n";
    return ss.str();
  }
};

// register the model to make it available
REGISTER_CAUSAL_MODEL(gemma2, Gemma2ForCausalLM);

REGISTER_DEFAULT_CHAT_TEMPLATE(gemma2, Gemma2ChatTemplate);

REGISTER_MODEL_ARGS(gemma2, [&] {
  // example config from
  // https://huggingface.co/google/gemma-2-2b/blob/main/config.json
  LOAD_ARG_OR(model_type, "model_type", "gemma2");
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 2);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 16384);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 8192);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 8);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 18);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 1);  // MQA
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(vocab_size, "vocab_size", 256000);
  LOAD_ARG_OR(hidden_act, "hidden_activation", "gelu_pytorch_tanh");
  LOAD_ARG_OR(attn_scalar, "query_pre_attn_scalar", 224);
  LOAD_ARG_OR(attn_bias, "attention_bias", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(attn_logit_soft_cap, "attn_logit_softcapping", 50.0f);
  LOAD_ARG_OR(final_logit_soft_cap, "final_logit_softcapping", 30.0f);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
});

}  // namespace llm::hf
