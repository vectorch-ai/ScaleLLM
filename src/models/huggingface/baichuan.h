#pragma once

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "chat_template/coded_chat_template.h"
#include "layers/activation.h"
#include "layers/attention/attention.h"
#include "layers/attention/handler.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/normalization.h"
#include "memory/kv_cache.h"
#include "models/model_args.h"
#include "models/model_registry.h"
#include "models/parameters.h"

// Baichuan model compatible with huggingface weights

namespace llm::hf {

enum class BaichuanType : uint8_t {
  Baichuan_7B,
  Baichuan2_7B,
  Baichuan_13B,
  Baichuan2_13B,
};

class BaichuanMLPImpl : public torch::nn::Module {
 public:
  BaichuanMLPImpl(const ModelArgs& args,
                  const QuantArgs& quant_args,
                  const ParallelArgs& parallel_args,
                  const torch::TensorOptions& options) {
    const int64_t hidden_size = args.hidden_size();
    const int64_t intermediate_size = args.intermediate_size();

    act_with_mul_ =
        Activation::get_act_with_mul_func(args.hidden_act(), options.device());
    CHECK(act_with_mul_ != nullptr);

    // register the weight parameter
    gate_up_proj_ =
        register_module("gate_up_proj",
                        ColumnParallelLinear(hidden_size,
                                             intermediate_size * 2,
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
    return down_proj_(act_with_mul_(gate_up_proj_(x)));
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
  ColumnParallelLinear gate_up_proj_{nullptr};
  RowParallelLinear down_proj_{nullptr};

  // calculate act(x) * y
  ActFunc act_with_mul_{nullptr};
};
TORCH_MODULE(BaichuanMLP);

class BaichuanAttentionImpl : public torch::nn::Module {
 public:
  BaichuanAttentionImpl(const ModelArgs& args,
                        const QuantArgs& quant_args,
                        const ParallelArgs& parallel_args,
                        const torch::TensorOptions& options,
                        BaichuanType baichuan_type,
                        AttentionHandler* handler)
      : baichuan_type_(baichuan_type) {
    const int32_t world_size = parallel_args.world_size();
    const int64_t hidden_size = args.hidden_size();
    const int64_t n_heads = args.n_heads();
    const int64_t head_dim = args.head_dim();
    const int64_t n_local_heads = n_heads / world_size;

    // size for local q, k, v
    qkv_sizes_ = {n_local_heads * head_dim,
                  n_local_heads * head_dim,
                  n_local_heads * head_dim};

    // register submodules
    W_pack_ = register_module("W_pack",
                              ColumnParallelLinear(hidden_size,
                                                   (3 * n_heads) * head_dim,
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

    // initialize attention module
    atten_ = register_module(
        "atten", Attention(n_local_heads, n_local_heads, head_dim, handler));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    // (num_tokens, dim) x (dim, n_local_heads * head_dim)
    // => (num_tokens, n_local_heads * head_dim)
    auto qkv = W_pack_(x).split(/*split_size=*/qkv_sizes_, /*dim=*/-1);
    DCHECK_EQ(qkv.size(), 3);

    torch::Tensor output;
    // calculate attention, output: (num_tokens, n_local_heads * head_dim)
    output = atten_(qkv[0], qkv[1], qkv[2], positions, kv_cache, input_params);
    return o_proj_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    W_pack_->load_state_dict(state_dict.select("W_pack."));
    o_proj_->load_state_dict(state_dict.select("o_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    W_pack_->verify_loaded_weights(prefix + "W_pack.");
    o_proj_->verify_loaded_weights(prefix + "o_proj.");
  }

 private:
  BaichuanType baichuan_type_;

  // parameter members, must be registered
  ColumnParallelLinear W_pack_{nullptr};

  RowParallelLinear o_proj_{nullptr};

  // module members without parameters
  Attention atten_{nullptr};

  // size for local q, k, v
  std::vector<int64_t> qkv_sizes_;
};
TORCH_MODULE(BaichuanAttention);

class BaichuanDecoderLayerImpl : public torch::nn::Module {
 public:
  BaichuanDecoderLayerImpl(const ModelArgs& args,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options,
                           BaichuanType baichuan_type,
                           AttentionHandler* handler) {
    // register submodules
    self_attn_ = register_module(
        "self_attn",
        BaichuanAttention(
            args, quant_args, parallel_args, options, baichuan_type, handler));
    mlp_ = register_module(
        "mlp", BaichuanMLP(args, quant_args, parallel_args, options));

    input_layernorm_ = register_module(
        "input_layernorm",
        RMSNormResidual(args.hidden_size(), args.rms_norm_eps(), options));
    post_attention_layernorm_ = register_module(
        "post_attention_layernorm",
        RMSNormResidual(args.hidden_size(), args.rms_norm_eps(), options));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        torch::Tensor& residual,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    auto hidden_states = input_layernorm_(x, residual);

    hidden_states =
        self_attn_(hidden_states, positions, kv_cache, input_params);
    hidden_states = post_attention_layernorm_(hidden_states, residual);
    hidden_states = mlp_(hidden_states);
    return hidden_states;
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
  BaichuanAttention self_attn_{nullptr};

  BaichuanMLP mlp_{nullptr};

  // RSM Norm
  RMSNormResidual input_layernorm_{nullptr};
  RMSNormResidual post_attention_layernorm_{nullptr};
};
TORCH_MODULE(BaichuanDecoderLayer);

class BaichuanModelImpl : public torch::nn::Module {
 public:
  BaichuanModelImpl(const ModelArgs& args,
                    const QuantArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options,
                    BaichuanType baichuan_type) {
    // register submodules
    embed_tokens_ = register_module(
        "embed_tokens",
        ParallelEmbedding(
            args.vocab_size(), args.hidden_size(), parallel_args, options));
    if (baichuan_type == BaichuanType::Baichuan_7B ||
        baichuan_type == BaichuanType::Baichuan2_7B) {
      handler_ = AttentionHandler::create_handler_with_rope(
          args, /*interleaved=*/false, options);
    } else {
      const torch::Tensor alibi_slopes =
          prepare_alibi_slopes(args.n_heads(), parallel_args);

      handler_ = AttentionHandler::create_handler_with_alibi(
          args, alibi_slopes, options);
    }

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = BaichuanDecoderLayer(args,
                                        quant_args,
                                        parallel_args,
                                        options,
                                        baichuan_type,
                                        handler_.get());
      layers_.push_back(block);
      blocks_->push_back(block);
    }

    norm_ = register_module(
        "norm",
        RMSNormResidual(args.hidden_size(), args.rms_norm_eps(), options));
  }

  // tokens: [num_tokens]
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = embed_tokens_(tokens);
    torch::Tensor residual;

    // TODO: set working space for attention handler
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, residual, kv_caches[i], input_params);
    }
    return norm_(h, residual);
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
  // returns alibi_slopes for attention handler [n_heads]
  static torch::Tensor prepare_alibi_slopes(int64_t n_heads,
                                            const ParallelArgs& parallel_args) {
    // calculate alibi_slopes
    const int64_t closest_power_of_2 =
        std::pow(2, std::floor(std::log2(n_heads)));
    const float base =
        std::pow(2, -(std::pow(2, -(std::log2(closest_power_of_2) - 3))));
    const torch::Tensor powers = torch::arange(
        /*start=*/1, /*end=*/1 + closest_power_of_2, torch::kFloat32);
    torch::Tensor slopes = torch::pow(base, powers);

    if (closest_power_of_2 != n_heads) {
      const float extra_base =
          std::pow(2, -(std::pow(2, -(std::log2(2 * closest_power_of_2) - 3))));
      const int64_t n_remaining_heads =
          std::min(closest_power_of_2, n_heads - closest_power_of_2);
      const torch::Tensor extra_powers =
          torch::arange(/*start=*/1,
                        /*end=*/1 + 2 * n_remaining_heads,
                        /*step=*/2,
                        torch::kFloat32);
      const torch::Tensor extra_slopes = torch::pow(extra_base, extra_powers);
      slopes = torch::cat({slopes, extra_slopes}, /*dim=*/0);
    }
    if (parallel_args.world_size() > 1) {
      slopes = slopes.chunk(/*chunks=*/parallel_args.world_size(),
                            /*dim=*/0)[parallel_args.rank()];
    }
    return slopes;
  }

 private:
  // parameter members, must be registered
  ParallelEmbedding embed_tokens_{nullptr};

  // attention handler
  std::unique_ptr<AttentionHandler> handler_{nullptr};

  // parameter members, must be registered
  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<BaichuanDecoderLayer> layers_;

  // final layer norm
  RMSNormResidual norm_{nullptr};
};
TORCH_MODULE(BaichuanModel);

class BaichuanForCausalLMImpl : public torch::nn::Module {
 public:
  BaichuanForCausalLMImpl(const ModelArgs& args,
                          const QuantArgs& quant_args,
                          const ParallelArgs& parallel_args,
                          const torch::TensorOptions& options) {
    if (args.vocab_size() == 125696) {
      // Baichuan2
      if (args.hidden_size() == 4096) {
        baichuan_type_ = BaichuanType::Baichuan2_7B;
      } else {
        baichuan_type_ = BaichuanType::Baichuan2_13B;
      }
    } else {
      // Baichuan
      if (args.hidden_size() == 4096) {
        baichuan_type_ = BaichuanType::Baichuan_7B;
      } else {
        baichuan_type_ = BaichuanType::Baichuan_13B;
      }
    }
    // register submodules
    model_ = register_module(
        "model",
        BaichuanModel(
            args, quant_args, parallel_args, options, baichuan_type_));

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
    if (baichuan_type_ == BaichuanType::Baichuan2_13B ||
        baichuan_type_ == BaichuanType::Baichuan2_7B) {
      // Baichuan2 normalizes the head weights:
      // https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/main/modeling_baichuan.py#L508
      lm_head_->load_state_dict(state_dict.select_with_transform(
          "lm_head.", [](torch::Tensor tensor) {
            return torch::nn::functional::normalize(tensor);
          }));
    } else {
      lm_head_->load_state_dict(state_dict.select("lm_head."));
    }
  }

  void verify_loaded_weights() const {
    model_->verify_loaded_weights("model.");
    lm_head_->verify_loaded_weights("lm_head.");
  }

 private:
  BaichuanType baichuan_type_;
  // parameter members, must be registered
  BaichuanModel model_{nullptr};

  ColumnParallelLinear lm_head_{nullptr};
};
TORCH_MODULE(BaichuanForCausalLM);

class BaichuanChatTemplate final : public CodedChatTemplate {
 public:
  // generate prompt from dialogs
  // Prompt template:
  // <|im_start|>user\n {message} <|im_end|>\n
  // <|im_start|>assistant\n
  std::optional<std::string> get_prompt(
      const std::string_view& system_message,
      const std::vector<std::string_view>& messages) const override {
    // at least one user message
    if (messages.size() % 2 == 0) {
      return std::nullopt;
    }

    std::stringstream ss;
    if (!system_message.empty()) {
      ss << "<|im_start|>system\n" << system_message << "<|im_end|>\n";
    }

    // then user and assistant message pairs (u/a/u/a/u...)
    for (size_t i = 0; i < messages.size(); ++i) {
      const char* role = (i % 2) == 0 ? "user" : "assistant";
      ss << "<|im_start|>" << role << "\n" << messages[i] << "<|im_end|>\n";
    }
    // end with assistant message
    ss << "<|im_start|>assistant\n";
    return ss.str();
  }
};

// register the model to make it available
REGISTER_CAUSAL_MODEL(baichuan, BaichuanForCausalLM);
REGISTER_DEFAULT_CHAT_TEMPLATE(baichuan, BaichuanChatTemplate);
REGISTER_MODEL_ARGS(baichuan, [&] {
  // example config:
  // TODO:
  // https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/main/config.json
  LOAD_ARG_OR(model_type, "model_type", "baichuan");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
  LOAD_ARG_OR(eos_token_id, "bos_token_id", 1);
  // LOAD_ARG_OR(pad_token_id, "pad_token_id", 0);
  LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
  // initializer_range in Qwen & Falcon have but never use
  // LOAD_ARG_OR(initializer_range, "initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 11008);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 4096);
  // LOAD_ARG_OR(model_max_length, "model_max_length", 4096);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 32);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(vocab_size, "vocab_size", 125696);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(rope_scaling, "rope_scaling", 1.0f);
  // LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
});

}  // namespace llm::hf
