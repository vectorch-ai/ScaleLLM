#pragma once

#include <torch/torch.h>
#include <torch/types.h>

#include "layers/activation.h"
#include "layers/attention_rope.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/normalization.h"
#include "memory/kv_cache.h"
#include "models/args.h"
#include "models/input_parameters.h"
#include "models/model_registry.h"

// ChatGLM model compatible with huggingface weights

namespace llm::hf {

class ChatGLMMLPImpl : public torch::nn::Module {
 public:
  ChatGLMMLPImpl(const ModelArgs& args,
                 const QuantArgs& quant_args,
                 const ParallelArgs& parallel_args,
                 torch::ScalarType dtype,
                 const torch::Device& device) {
    const int64_t hidden_size = args.hidden_size();
    const int64_t intermediate_size = args.intermediate_size();

    act_with_mul_ = Activation::get_act_with_mul_func("silu", device);
    GCHECK(act_with_mul_ != nullptr);

    // register the weight parameter
    dense_h_to_4h_ =
        register_module("dense_h_to_4h",
                        ColumnParallelLinear(hidden_size,
                                             intermediate_size * 2,
                                             /*bias=*/args.linear_bias(),
                                             /*gather_output=*/false,
                                             quant_args,
                                             parallel_args,
                                             dtype,
                                             device));
    dense_4h_to_h_ =
        register_module("dense_4h_to_h",
                        RowParallelLinear(intermediate_size,
                                          hidden_size,
                                          /*bias=*/args.linear_bias(),
                                          /*input_is_parallelized=*/true,
                                          quant_args,
                                          parallel_args,
                                          dtype,
                                          device));
  }

  torch::Tensor forward(torch::Tensor x) {
    return dense_4h_to_h_(act_with_mul_(dense_h_to_4h_(x)));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    dense_h_to_4h_->load_state_dict(state_dict.select("dense_h_to_4h."));
    dense_4h_to_h_->load_state_dict(state_dict.select("dense_4h_to_h."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    dense_h_to_4h_->verify_loaded_weights(prefix + "dense_h_to_4h.");
    dense_4h_to_h_->verify_loaded_weights(prefix + "dense_4h_to_h.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear dense_h_to_4h_{nullptr};
  RowParallelLinear dense_4h_to_h_{nullptr};

  // calculate act(x) * y
  ActFunc act_with_mul_{nullptr};
};
TORCH_MODULE(ChatGLMMLP);

class ChatGLMAttentionImpl : public torch::nn::Module {
 public:
  ChatGLMAttentionImpl(const ModelArgs& args,
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

    // size for local q, k, v
    qkv_sizes_ = {n_local_heads * head_dim,
                  n_local_kv_heads * head_dim,
                  n_local_kv_heads * head_dim};

    // register submodules
    query_key_value_ = register_module(
        "query_key_value",
        ColumnParallelLinear(hidden_size,
                             (n_heads + 2 * n_kv_heads) * head_dim,
                             /*bias=*/args.linear_bias() || args.qkv_bias(),
                             /*gather_output=*/false,
                             quant_args,
                             parallel_args,
                             dtype,
                             device));

    dense_ = register_module("dense",
                             RowParallelLinear(hidden_size,
                                               hidden_size,
                                               /*bias=*/args.linear_bias(),
                                               /*input_is_parallelized=*/true,
                                               quant_args,
                                               parallel_args,
                                               dtype,
                                               device));

    // initialize positional embedding and attention
    const int64_t rotary_dim =
        static_cast<int64_t>(head_dim * args.rotary_pct());
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    atten_ = register_module(
        "atten",
        AttentionWithRoPE(n_local_heads,
                          n_local_kv_heads,
                          head_dim,
                          scale,
                          rotary_dim,
                          args.rope_scaling(),
                          args.rope_theta(),
                          /*max_position=*/args.max_position_embeddings(),
                          /*interleaved=*/true,
                          dtype,
                          device));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    // (num_tokens, dim) x (dim, n_local_heads * head_dim)
    // => (num_tokens, n_local_heads * head_dim)
    auto qkv = query_key_value_(x).split(/*split_size=*/qkv_sizes_, /*dim=*/-1);
    DCHECK_EQ(qkv.size(), 3);
    // calculate attention, output: (num_tokens, n_local_heads * head_dim)
    auto output =
        atten_(qkv[0], qkv[1], qkv[2], positions, kv_cache, input_params);
    return dense_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    query_key_value_->load_state_dict(state_dict.select("query_key_value."));
    dense_->load_state_dict(state_dict.select("dense."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    query_key_value_->verify_loaded_weights(prefix + "query_key_value.");
    dense_->verify_loaded_weights(prefix + "dense.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear query_key_value_{nullptr};

  RowParallelLinear dense_{nullptr};

  // module members without parameters
  AttentionWithRoPE atten_{nullptr};

  // size for local q, k, v
  std::vector<int64_t> qkv_sizes_;
};
TORCH_MODULE(ChatGLMAttention);

class ChatGLMBlockImpl : public torch::nn::Module {
 public:
  ChatGLMBlockImpl(const ModelArgs& args,
                   const QuantArgs& quant_args,
                   const ParallelArgs& parallel_args,
                   torch::ScalarType dtype,
                   const torch::Device& device)
      : residual_post_layernorm_(args.residual_post_layernorm()),
        use_rms_norm_(args.use_rms_norm()) {
    // register submodules
    self_attention_ = register_module(
        "self_attention",
        ChatGLMAttention(args, quant_args, parallel_args, dtype, device));
    mlp_ = register_module(
        "mlp", ChatGLMMLP(args, quant_args, parallel_args, dtype, device));

    if (use_rms_norm_) {
      input_rmsnorm_ = register_module(
          "input_layernorm",
          RMSNorm(args.hidden_size(), args.layer_norm_eps(), dtype, device));
      post_attention_rmsnorm_ = register_module(
          "post_attention_layernorm",
          RMSNorm(args.hidden_size(), args.layer_norm_eps(), dtype, device));
    } else {
      input_layernorm_ = register_module("input_layernorm",
                                         LayerNorm(args.hidden_size(),
                                                   args.layer_norm_eps(),
                                                   /*bias=*/false,
                                                   dtype,
                                                   device));
      post_attention_layernorm_ =
          register_module("post_attention_layernorm",
                          LayerNorm(args.hidden_size(),
                                    args.layer_norm_eps(),
                                    /*bias=*/false,
                                    dtype,
                                    device));
    }
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    torch::Tensor ln_output;
    if (use_rms_norm_) {
      ln_output = input_rmsnorm_(x);
    } else {
      ln_output = input_layernorm_(x);
    }
    auto residual = residual_post_layernorm_ ? ln_output : x;

    auto attn_output =
        self_attention_(ln_output, positions, kv_cache, input_params);
    attn_output += residual;

    if (use_rms_norm_) {
      ln_output = post_attention_rmsnorm_(attn_output);
    } else {
      ln_output = post_attention_layernorm_(attn_output);
    }
    residual = residual_post_layernorm_ ? ln_output : attn_output;
    return mlp_(ln_output) + residual;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    self_attention_->load_state_dict(state_dict.select("self_attention."));
    mlp_->load_state_dict(state_dict.select("mlp."));
    if (use_rms_norm_) {
      input_rmsnorm_->load_state_dict(state_dict.select("input_layernorm."));
      post_attention_rmsnorm_->load_state_dict(
          state_dict.select("post_attention_layernorm."));
    } else {
      input_layernorm_->load_state_dict(state_dict.select("input_layernorm."));
      post_attention_layernorm_->load_state_dict(
          state_dict.select("post_attention_layernorm."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    self_attention_->verify_loaded_weights(prefix + "self_attention.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    if (use_rms_norm_) {
      input_rmsnorm_->verify_loaded_weights(prefix + "input_layernorm.");
      post_attention_rmsnorm_->verify_loaded_weights(
          prefix + "post_attention_layernorm.");
    } else {
      input_layernorm_->verify_loaded_weights(prefix + "input_layernorm.");
      post_attention_layernorm_->verify_loaded_weights(
          prefix + "post_attention_layernorm.");
    }
  }

 private:
  // parameter members, must be registered
  ChatGLMAttention self_attention_{nullptr};

  ChatGLMMLP mlp_{nullptr};

  LayerNorm input_layernorm_{nullptr};
  LayerNorm post_attention_layernorm_{nullptr};

  // RSM Norm
  RMSNorm input_rmsnorm_{nullptr};
  RMSNorm post_attention_rmsnorm_{nullptr};

  bool residual_post_layernorm_ = false;
  bool use_rms_norm_ = false;
};
TORCH_MODULE(ChatGLMBlock);

class ChatGLMModelImpl : public torch::nn::Module {
 public:
  ChatGLMModelImpl(const ModelArgs& args,
                   const QuantArgs& quant_args,
                   const ParallelArgs& parallel_args,
                   torch::ScalarType dtype,
                   const torch::Device& device) {
    // register submodules
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = ChatGLMBlock(args, quant_args, parallel_args, dtype, device);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    final_layernorm_ = register_module("final_layernorm",
                                       LayerNorm(args.hidden_size(),
                                                 args.layer_norm_eps(),
                                                 /*bias=*/false,
                                                 dtype,
                                                 device));
  }

  // tokens: [num_tokens]
  torch::Tensor forward(torch::Tensor h,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, kv_caches[i], input_params);
    }
    return final_layernorm_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.select("layers." + std::to_string(i) + "."));
    }
    final_layernorm_->load_state_dict(state_dict.select("final_layernorm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    final_layernorm_->verify_loaded_weights(prefix + "final_layernorm.");
  }

 private:
  // parameter members, must be registered
  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<ChatGLMBlock> layers_;

  // final layer norm
  LayerNorm final_layernorm_{nullptr};
};
TORCH_MODULE(ChatGLMModel);

class ChatGLMForCausalLMImpl : public torch::nn::Module {
 public:
  ChatGLMForCausalLMImpl(const ModelArgs& args,
                         const QuantArgs& quant_args,
                         const ParallelArgs& parallel_args,
                         torch::ScalarType dtype,
                         const torch::Device& device) {
    // register submodules
    word_embeddings_ = register_module("word_embeddings",
                                       ParallelEmbedding(args.vocab_size(),
                                                         args.hidden_size(),
                                                         parallel_args,
                                                         dtype,
                                                         device));
    model_ = register_module(
        "encoder",
        ChatGLMModel(args, quant_args, parallel_args, dtype, device));

    output_layer_ = register_module("output_layer",
                                    ColumnParallelLinear(args.hidden_size(),
                                                         args.vocab_size(),
                                                         /*bias=*/false,
                                                         /*gather_output=*/true,
                                                         parallel_args,
                                                         dtype,
                                                         device));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h =
        model_(word_embeddings_(tokens), positions, kv_caches, input_params);
    // select last token for each sequence
    h = h.index_select(/*dim=*/0, input_params.last_token_idxes);
    return output_layer_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    word_embeddings_->load_state_dict(
        state_dict.select("transformer.embedding.word_embeddings."));
    model_->load_state_dict(state_dict.select("transformer.encoder."));
    output_layer_->load_state_dict(
        state_dict.select("transformer.output_layer."));
  }

  void verify_loaded_weights() const {
    word_embeddings_->verify_loaded_weights(
        "transformer.embedding.word_embeddings.");
    model_->verify_loaded_weights("transformer.encoder.");
    output_layer_->verify_loaded_weights("transformer.output_layer.");
  }

 private:
  // parameter members, must be registered
  ParallelEmbedding word_embeddings_{nullptr};

  ChatGLMModel model_{nullptr};

  ColumnParallelLinear output_layer_{nullptr};
};
TORCH_MODULE(ChatGLMForCausalLM);

class ChatGLMConversation final : public Conversation {
 public:
  // generate prompt from dialogs
  // https://github.com/THUDM/ChatGLM3/blob/main/PROMPT.md
  std::optional<std::string> get_prompt() const override {
    // at least one user message
    if (messages_.size() % 2 == 0) {
      return std::nullopt;
    }

    std::stringstream ss;
    if (!system_message_.empty()) {
      ss << "<|system|>\n" << system_message_ << "\n";
    }

    // then user and assistant message pairs (u/a/u/a/u...)
    for (size_t i = 0; i < messages_.size(); ++i) {
      const char* role = (i % 2) == 0 ? "user" : "assistant";
      ss << "<|" << role << "|>\n" << messages_[i] << "\n";
    }
    // end with assistant message
    ss << "<|assistant|>\n";
    return ss.str();
  }
};

// register the model to make it available
REGISTER_CAUSAL_MODEL(chatglm, ChatGLMForCausalLM);
REGISTER_CONVERSATION_TEMPLATE(chatglm, ChatGLMConversation);
REGISTER_MODEL_ARGS(chatglm, [&] {
  // example config:
  // https://huggingface.co/THUDM/chatglm3-6b/blob/main/config.json
  LOAD_ARG_OR(model_type, "model_type", "chatglm");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "padded_vocab_size", 65024);
  LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
  LOAD_ARG_OR(intermediate_size, "ffn_hidden_size", 13696);
  LOAD_ARG_OR(n_layers, "num_layers", 28);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(use_rms_norm, "rmsnorm", false);
  LOAD_ARG_OR(layer_norm_eps, "layernorm_epsilon", 1e-5);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
  LOAD_ARG_OR(residual_post_layernorm,
              "apply_residual_connection_post_layernorm",
              false);
  LOAD_ARG_OR(max_position_embeddings, "seq_length", 8192);

  // assign kv heads from multi_query_group_num if multi_query_attention is used
  LOAD_ARG_OR_FUNC(n_kv_heads, "num_kv_attention_heads", [&] {
    std::optional<int64_t> n_kv_heads;
    // read kv heads from multi_query_group_num
    const bool use_mqa = json.value_or<bool>("multi_query_attention", false);
    if (use_mqa) {
      n_kv_heads = json.value<int64_t>("multi_query_group_num");
    }
    return n_kv_heads;
  });

  // rotary position embedding related args
  LOAD_ARG_OR(rotary_pct, "rotary_pct", 0.5f);
  LOAD_ARG_OR_FUNC(rope_theta, "rope_theta", [&] {
    const float rope_ratio = json.value_or<float>("rope_ratio", 1.0f);
    return rope_ratio * 10000.0f;
  });

  // stop token ids: "</s>", "<|user|>", "<|assistant|>", "<|observation|>"
  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>({2, 64795, 64796, 64797}));
});

// Register tokenizer args since chatglm is using sentencepiece tokenizer.
REGISTER_TOKENIZER_ARGS(chatglm, [&] {
  SET_ARG(tokenizer_type, "sentencepiece");
  // adapted from
  // https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenization_chatglm.py
  SET_ARG(vocab_file, "tokenizer.model");

  // set special tokens
  // clang-format off
  const std::vector<std::string> special_tokens({
    "[MASK]", "[gMASK]", "[sMASK]", "sop", "eop",
    "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"
  });
  // clang-format on
  SET_ARG(special_tokens, special_tokens);
  SET_ARG(prefix_tokens, std::vector<std::string>({"[gMASK]", "sop"}));
});

}  // namespace llm::hf
