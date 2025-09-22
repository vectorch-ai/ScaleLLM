#pragma once

#include <torch/torch.h>

#include <optional>

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
#include "module/module.h"
#include "module/module_holder.h"
#include "module/modulelist.h"
// mpt model compatible with huggingface weights
namespace llm::hf {

class MPTMLPImpl : public Module {
 public:
  MPTMLPImpl(const ModelArgs& args,
             const QuantArgs& quant_args,
             const ParallelArgs& parallel_args,
             const torch::TensorOptions& options) {
    act_ = Activation::get_act_func("gelu", options.device());
    CHECK(act_ != nullptr);

    const int64_t hidden_size = args.hidden_size();
    const int64_t intermediate_size = args.intermediate_size();

    // register the weight parameter
    up_proj_ = register_module("up_proj",
                               ColumnParallelLinear(hidden_size,
                                                    intermediate_size,
                                                    /*bias=*/!args.no_bias(),
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args,
                                                    options));
    down_proj_ =
        register_module("down_proj",
                        RowParallelLinear(intermediate_size,
                                          hidden_size,
                                          /*bias=*/!args.no_bias(),
                                          /*input_is_parallelized=*/true,
                                          quant_args,
                                          parallel_args,
                                          options));
  }

  torch::Tensor forward(torch::Tensor x) {
    return down_proj_(act_(up_proj_(x)));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    up_proj_->load_state_dict(state_dict.select("up_proj."));
    down_proj_->load_state_dict(state_dict.select("down_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    up_proj_->verify_loaded_weights(prefix + "up_proj.");
    down_proj_->verify_loaded_weights(prefix + "down_proj.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear up_proj_{nullptr};
  RowParallelLinear down_proj_{nullptr};

  ActFunc act_{nullptr};
};
LLM_MODULE(MPTMLP);

class MPTAttentionImpl : public Module {
 public:
  MPTAttentionImpl(const ModelArgs& args,
                   const QuantArgs& quant_args,
                   const ParallelArgs& parallel_args,
                   const torch::TensorOptions& options,
                   AttentionHandler* handler)
      : qk_layer_norm_(args.attn_qk_ln()),
        attn_qkv_clip_(args.attn_qkv_clip()) {
    const auto world_size = parallel_args.world_size();
    const int64_t hidden_size = args.hidden_size();
    const int64_t n_heads = args.n_heads();
    const int64_t head_dim = args.head_dim();
    const int64_t n_local_heads = n_heads / world_size;
    hidden_size_ = hidden_size;
    head_dim_ = head_dim;

    // register submodules
    wqkv_ = register_module("Wqkv",
                            ColumnParallelLinear(hidden_size,
                                                 3 * hidden_size,
                                                 /*bias=*/!args.no_bias(),
                                                 /*gather_output=*/false,
                                                 quant_args,
                                                 parallel_args,
                                                 options));
    if (args.attn_qk_ln()) {
      q_ln_ = register_module("q_ln",
                              LayerNorm(args.hidden_size(),
                                        args.layer_norm_eps(),
                                        /*bias=*/!args.no_bias(),
                                        options));
      k_ln_ = register_module("k_ln",
                              LayerNorm(args.hidden_size(),
                                        args.layer_norm_eps(),
                                        /*bias=*/!args.no_bias(),
                                        options));
    }

    out_proj_ =
        register_module("out_proj",
                        RowParallelLinear(hidden_size,
                                          hidden_size,
                                          /*bias=*/!args.no_bias(),
                                          /*input_is_parallelized=*/true,
                                          quant_args,
                                          parallel_args,
                                          options));

    CHECK(args.attn_alibi()) << "only support alibi attention";

    atten_ = register_module(
        "atten", Attention(n_local_heads, n_local_heads, head_dim, handler));
  }

  torch::Tensor forward(torch::Tensor x,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    // (num_tokens, dim) x (dim, n_heads * head_dim)
    // => (num_tokens, n_heads * head_dim)
    auto qkv = wqkv_(x);
    if (attn_qkv_clip_) {
      const auto value = attn_qkv_clip_.value();
      qkv.clamp_(/*min=*/-value, /*max=*/value);
    }
    auto chunks = qkv.chunk(/*chunks=*/3, /*dim=*/-1);
    auto q = chunks[0];
    auto k = chunks[1];
    auto v = chunks[2];
    if (qk_layer_norm_) {
      q = q_ln_(q);
      k = k_ln_(k);
    }
    // calculate attention, output: (num_tokens, n_local_heads * head_dim)
    auto output =
        atten_(q, k, v, /*positions=*/torch::Tensor{}, kv_cache, input_params);
    return out_proj_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    auto qkv_state_dict = state_dict.select_with_transform(
        "Wqkv.",
        [this](const std::string_view& /*name*/, const torch::Tensor& tensor) {
          return reshape_qkv_before_sharding(tensor);
        });
    // reshape local qkv back to [3, n_heads, ...] after sharding
    wqkv_->load_state_dict(qkv_state_dict, [this](torch::Tensor tensor) {
      return reshape_qkv_after_sharding(tensor);
    });
    out_proj_->load_state_dict(state_dict.select("out_proj."));
    if (qk_layer_norm_) {
      q_ln_->load_state_dict(state_dict.select("q_ln."));
      k_ln_->load_state_dict(state_dict.select("k_ln."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    wqkv_->verify_loaded_weights(prefix + "Wqkv.");
    out_proj_->verify_loaded_weights(prefix + "out_proj.");
    if (qk_layer_norm_) {
      q_ln_->verify_loaded_weights(prefix + "q_ln.");
      k_ln_->verify_loaded_weights(prefix + "k_ln.");
    }
  }

 private:
  // reshape qkv tensor from [3, n_heads, ...] => [n_heads, 3, ...]
  torch::Tensor reshape_qkv_before_sharding(const torch::Tensor& tensor) {
    CHECK(tensor.dim() == 2 || tensor.dim() == 1)
        << "unexpected tensor dim: " << tensor.dim();
    if (tensor.dim() == 2) {
      return tensor.view({3, -1, head_dim_, hidden_size_})
          .permute({1, 0, 2, 3})
          .reshape({-1, hidden_size_});
    }
    return tensor.view({3, -1, head_dim_}).permute({1, 0, 2}).reshape({-1});
  }

  // reshape local qkv tensor from [n_heads, 3, ...] => [3, n_heads, ...]
  torch::Tensor reshape_qkv_after_sharding(const torch::Tensor& tensor) {
    // N.B. Fused qkv weights in GPT-NeoX has the shape of [n_heads * 3 *
    // head_dim, hidden_size], while the desired shape is [3 * n_heads *
    // head_dim, hidden_size].
    CHECK(tensor.dim() == 2 || tensor.dim() == 1)
        << "unexpected tensor dim: " << tensor.dim();
    if (tensor.dim() == 2) {
      return tensor.view({-1, 3, head_dim_, hidden_size_})
          .permute({1, 0, 2, 3})
          .reshape({-1, hidden_size_});
    }
    return tensor.view({-1, 3, head_dim_}).permute({1, 0, 2}).reshape({-1});
  }
  // parameter members, must be registered
  ColumnParallelLinear wqkv_{nullptr};

  RowParallelLinear out_proj_{nullptr};

  LayerNorm q_ln_{nullptr};

  LayerNorm k_ln_{nullptr};

  // module members without parameters
  Attention atten_{nullptr};

  // whether to apply layer norm to qk
  bool qk_layer_norm_{false};

  std::optional<float> attn_qkv_clip_;

  int64_t hidden_size_ = 0;
  int64_t head_dim_ = 0;
};
LLM_MODULE(MPTAttention);

class MPTBlockImpl : public Module {
 public:
  MPTBlockImpl(const ModelArgs& args,
               const QuantArgs& quant_args,
               const ParallelArgs& parallel_args,
               const torch::TensorOptions& options,
               AttentionHandler* handler) {
    // register submodules
    attn_ = register_module(
        "attn",
        MPTAttention(args, quant_args, parallel_args, options, handler));
    norm_1_ = register_module("norm_1",
                              LayerNorm(args.hidden_size(),
                                        args.layer_norm_eps(),
                                        /*bias=*/!args.no_bias(),
                                        options));
    norm_2_ = register_module("norm_2",
                              LayerNorm(args.hidden_size(),
                                        args.layer_norm_eps(),
                                        /*bias=*/!args.no_bias(),
                                        options));
    ffn_ = register_module("ffn",
                           MPTMLP(args, quant_args, parallel_args, options));
  }

  torch::Tensor forward(torch::Tensor x,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    auto h = x + attn_(norm_1_(x), kv_cache, input_params);
    return h + ffn_(norm_2_(h));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    attn_->load_state_dict(state_dict.select("attn."));
    norm_1_->load_state_dict(state_dict.select("norm_1."));
    norm_2_->load_state_dict(state_dict.select("norm_2."));
    ffn_->load_state_dict(state_dict.select("ffn."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    attn_->verify_loaded_weights(prefix + "attn.");
    norm_1_->verify_loaded_weights(prefix + "norm_1.");
    norm_2_->verify_loaded_weights(prefix + "norm_2.");
    ffn_->verify_loaded_weights(prefix + "ffn.");
  }

 private:
  // parameter members, must be registered
  MPTAttention attn_{nullptr};

  MPTMLP ffn_{nullptr};

  LayerNorm norm_1_{nullptr};

  LayerNorm norm_2_{nullptr};
};
LLM_MODULE(MPTBlock);

class MPTModelImpl : public Module {
 public:
  MPTModelImpl(const ModelArgs& args,
               const QuantArgs& quant_args,
               const ParallelArgs& parallel_args,
               const torch::TensorOptions& options) {
    // register submodules
    wte_ = register_module(
        "wte",
        ParallelEmbedding(
            args.vocab_size(), args.hidden_size(), parallel_args, options));

    // calculate alibi_slopes
    torch::Tensor alibi_slopes = prepare_alibi_slopes(
        args.n_heads(), args.alibi_bias_max(), parallel_args);
    handler_ = AttentionHandler::create_handler_with_alibi(
        args, alibi_slopes, options);

    blocks_ = register_module("blocks", ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block =
          MPTBlock(args, quant_args, parallel_args, options, handler_.get());
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    norm_f_ = register_module("norm_f",
                              LayerNorm(args.hidden_size(),
                                        args.layer_norm_eps(),
                                        /*bias=*/!args.no_bias(),
                                        options));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = wte_(tokens);

    // TODO: set working space for attention handler
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, kv_caches[i], input_params);
    }
    return norm_f_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    wte_->load_state_dict(state_dict.select("wte."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.select("blocks." + std::to_string(i) + "."));
    }
    norm_f_->load_state_dict(state_dict.select("norm_f."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    wte_->verify_loaded_weights(prefix + "wte.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "blocks." + std::to_string(i) +
                                        ".");
    }
    norm_f_->verify_loaded_weights(prefix + "norm_f.");
  }

 private:
  static torch::Tensor prepare_alibi_slopes(int64_t n_heads,
                                            float bias_max,
                                            const ParallelArgs& parallel_args) {
    const int64_t next_power_of_2 = std::pow(2, std::ceil(std::log2(n_heads)));
    auto m = torch::arange(
        /*start=*/1, /*end=*/next_power_of_2 + 1, torch::kFloat32);
    m.mul_(bias_max / next_power_of_2);
    auto slopes = 1.0f / torch::pow(2, m);
    if (next_power_of_2 != n_heads) {
      using ISlice = torch::indexing::Slice;
      using torch::indexing::None;
      slopes = torch::cat({slopes.index({ISlice(1, None, 2)}),
                           slopes.index({ISlice(None, None, 2)})});
    }
    if (parallel_args.world_size() > 1) {
      slopes = slopes.chunk(/*chunks=*/parallel_args.world_size(),
                            /*dim=*/0)[parallel_args.rank()];
    }
    return slopes;
  }

  // parameter members, must be registered
  ParallelEmbedding wte_{nullptr};

  // attention handler
  std::unique_ptr<AttentionHandler> handler_{nullptr};

  ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<MPTBlock> layers_;

  LayerNorm norm_f_{nullptr};
};
LLM_MODULE(MPTModel);

class MPTForCausalLMImpl : public Module {
 public:
  MPTForCausalLMImpl(const ModelArgs& args,
                     const QuantArgs& quant_args,
                     const ParallelArgs& parallel_args,
                     const torch::TensorOptions& options) {
    // register submodules
    transformer_ = register_module(
        "transformer", MPTModel(args, quant_args, parallel_args, options));

    // TODO: share weights between wte and lm_head to save memory
    lm_head_ = register_module("wte",
                               ColumnParallelLinear(args.hidden_size(),
                                                    args.vocab_size(),
                                                    /*bias=*/!args.no_bias(),
                                                    /*gather_output=*/true,
                                                    parallel_args,
                                                    options));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& /*positions*/,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    return transformer_(tokens, kv_caches, input_params);
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
    transformer_->load_state_dict(state_dict.select("transformer."));
    lm_head_->load_state_dict(state_dict.select("transformer.wte."));
  }

  void verify_loaded_weights() const {
    transformer_->verify_loaded_weights("transformer.");
    lm_head_->verify_loaded_weights("transformer.wte.");
  }

 private:
  // parameter members, must be registered
  MPTModel transformer_{nullptr};

  ColumnParallelLinear lm_head_{nullptr};
};
LLM_MODULE(MPTForCausalLM);

class MPTChatTemplate final : public CodedChatTemplate {
 public:
  // Prompt template:
  // <|im_start|>system\n {system_message} <|im_end|>\n
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

// register the causal model
REGISTER_CAUSAL_MODEL(mpt, MPTForCausalLM);
REGISTER_DEFAULT_CHAT_TEMPLATE(mpt, MPTChatTemplate);

REGISTER_MODEL_ARGS(mpt, [&] {
  LOAD_ARG_OR(model_type, "model_type", "mpt");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 50368);
  LOAD_ARG_OR(hidden_size, "d_model", 2048);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 32);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(max_position_embeddings, "max_seq_len", 2048);
  LOAD_ARG_OR(layer_norm_eps, "layer_norm_eps", 1e-5);
  LOAD_ARG_OR(no_bias, "no_bias", true);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 0);

  // load config for attention
  LOAD_ARG(attn_qkv_clip, "attn_config.clip_qkv");
  LOAD_ARG_OR(attn_qk_ln, "attn_config.qk_ln", false);
  LOAD_ARG_OR(attn_alibi, "attn_config.alibi", false);
  LOAD_ARG_OR(alibi_bias_max, "attn_config.alibi_bias_max", 0.0f);

  LOAD_ARG_OR_FUNC(intermediate_size, "intermediate_size", [&] {
    const int64_t expansion_ratio =
        json.value_or<int64_t>("expansion_ratio", 4);
    return expansion_ratio * args->hidden_size();
  });

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  // stop token ids: [50278]
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({50278}));
});

}  // namespace llm::hf
