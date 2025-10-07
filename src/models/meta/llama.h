#pragma once

#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include "chat_template/common_chat_template.h"
#include "layers/activation.h"
#include "layers/attention/attention.h"
#include "layers/attention/handler.h"
#include "layers/embedding.h"
#include "layers/linear/multi_parallel_linear.h"
#include "layers/linear/qkv_parallel_linear.h"
#include "layers/normalization.h"
#include "memory/kv_cache.h"
#include "models/model_args.h"
#include "models/model_registry.h"
#include "models/parameters.h"
#include "module/module.h"
#include "module/module_holder.h"
#include "module/module_list.h"
// llama2 model compatible with huggingface weights
namespace llm::hf {

class LlamaMLPImpl : public Module {
 public:
  LlamaMLPImpl(const ModelArgs& args,
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
        MultiColumnParallelLinear(
            hidden_size,
            std::vector<int64_t>{intermediate_size, intermediate_size},
            std::vector<std::string>{"gate_proj.", "up_proj."},
            /*bias=*/false,
            /*gather_output=*/false,
            quant_args,
            parallel_args,
            options),
        /*selector=*/nullptr);

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

 private:
  // parameter members, must be registered
  MultiColumnParallelLinear gate_up_proj_{nullptr};
  RowParallelLinear down_proj_{nullptr};

  // activation function
  ActFunc act_func_{nullptr};
};
LLM_MODULE(LlamaMLP);

class LlamaAttentionImpl : public Module {
 public:
  LlamaAttentionImpl(const ModelArgs& args,
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
    const int64_t n_local_kv_heads =
        std::max<int64_t>(1, n_kv_heads / world_size);

    // register submodules
    qkv_proj_ = register_module(
        "qkv_proj",
        QKVColumnParallelLinear(
            hidden_size,
            n_heads,
            n_kv_heads,
            head_dim,
            std::vector<std::string>{"q_proj.", "k_proj.", "v_proj."},
            /*bias=*/false,
            /*gather_output=*/false,
            quant_args,
            parallel_args,
            options),
        /*selector=*/nullptr);

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
    const auto [q, k, v] = qkv_proj_(x);
    // calculate attention, output: (num_tokens, n_local_heads * head_dim)
    const auto output = atten_(q, k, v, positions, kv_cache, input_params);
    return o_proj_(output);
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
LLM_MODULE(LlamaAttention);

class LlamaDecoderLayerImpl : public Module {
 public:
  LlamaDecoderLayerImpl(const ModelArgs& args,
                        const QuantArgs& quant_args,
                        const ParallelArgs& parallel_args,
                        const torch::TensorOptions& options,
                        AttentionHandler* handler) {
    // register submodules
    self_attn_ = register_module(
        "self_attn",
        LlamaAttention(args, quant_args, parallel_args, options, handler));
    mlp_ = register_module("mlp",
                           LlamaMLP(args, quant_args, parallel_args, options));
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

 private:
  // parameter members, must be registered
  LlamaAttention self_attn_{nullptr};

  LlamaMLP mlp_{nullptr};

  RMSNorm input_layernorm_{nullptr};

  RMSNorm post_attention_layernorm_{nullptr};
};
LLM_MODULE(LlamaDecoderLayer);

class LlamaModelImpl : public Module {
 public:
  LlamaModelImpl(const ModelArgs& args,
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
      auto block = LlamaDecoderLayer(
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

 private:
  // parameter members, must be registered
  ParallelEmbedding embed_tokens_{nullptr};

  // attention handler
  std::unique_ptr<AttentionHandler> handler_{nullptr};

  ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<LlamaDecoderLayer> layers_;

  RMSNorm norm_{nullptr};
};
LLM_MODULE(LlamaModel);

class LlamaForCausalLMImpl : public Module {
 public:
  LlamaForCausalLMImpl(const ModelArgs& args,
                       const QuantArgs& quant_args,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options) {
    // register submodules
    model_ = register_module(
        "model", LlamaModel(args, quant_args, parallel_args, options));

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

 private:
  // parameter members, must be registered
  LlamaModel model_{nullptr};

  ColumnParallelLinear lm_head_{nullptr};
};
LLM_MODULE(LlamaForCausalLM);

class YiChatTemplate final : public CodedChatTemplate {
 public:
  // generate prompt from dialogs
  // https://huggingface.co/01-ai/Yi-34B-Chat/blob/main/tokenizer_config.json#L60
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

// register the causal model
REGISTER_CAUSAL_MODEL(llama, LlamaForCausalLM);
REGISTER_CAUSAL_MODEL(llama3, LlamaForCausalLM);
REGISTER_CAUSAL_MODEL(Yi, LlamaForCausalLM);

REGISTER_DEFAULT_CHAT_TEMPLATE(llama, Llama2ChatTemplate);
REGISTER_DEFAULT_CHAT_TEMPLATE(llama3, Llama3ChatTemplate);
REGISTER_DEFAULT_CHAT_TEMPLATE(Yi, YiChatTemplate);
// register the model args
// example config:
// https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/blob/main/config.json
REGISTER_MODEL_ARGS(llama, [&] {
  LOAD_ARG_OR(model_type, "model_type", "llama");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");

  // decide model type based on vocab size
  LOAD_ARG_OR(vocab_size, "vocab_size", 128256);
  if (args->vocab_size() == 128256) {
    // choose the right chat template
    SET_ARG(model_type, "llama3");

    LOAD_ARG_OR(hidden_size, "hidden_size", 8192);
    LOAD_ARG_OR(n_layers, "num_hidden_layers", 80);
    LOAD_ARG_OR(n_heads, "num_attention_heads", 64);
    LOAD_ARG_OR(intermediate_size, "intermediate_size", 28672);
    LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 8192);
    LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
    LOAD_ARG_OR(bos_token_id, "bos_token_id", 128000);
    // TODO: support a list of eos token ids
    LOAD_ARG_OR(eos_token_id, "eos_token_id", 128001);
    LOAD_ARG_OR(rope_theta, "rope_theta", 500000.0f);
    // load rope scaling parameters
    LOAD_ARG(rope_scaling_rope_type, "rope_scaling.rope_type");
    LOAD_ARG(rope_scaling_factor, "rope_scaling.factor");
    LOAD_ARG(rope_scaling_low_freq_factor, "rope_scaling.low_freq_factor");
    LOAD_ARG(rope_scaling_high_freq_factor, "rope_scaling.high_freq_factor");
    LOAD_ARG(rope_scaling_original_max_position_embeddings,
             "rope_scaling.original_max_position_embeddings");

    // stop token ids: "<|eom_id|>", "<|eot_id|>"
    SET_ARG(stop_token_ids, std::unordered_set<int32_t>({128008, 128009}));
  } else if (args->vocab_size() == 64000) {
    // choose the right chat template
    SET_ARG(model_type, "Yi");
    LOAD_ARG_OR(hidden_size, "hidden_size", 7168);
    LOAD_ARG_OR(n_layers, "num_hidden_layers", 60);
    LOAD_ARG_OR(n_heads, "num_attention_heads", 56);
    LOAD_ARG_OR(intermediate_size, "intermediate_size", 20480);
    LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 4096);
    LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
    LOAD_ARG_OR(bos_token_id, "bos_token_id", 1);
    LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
    LOAD_ARG_OR(rope_theta, "rope_theta", 5000000.0f);
    // LOAD_ARG_OR(rope_scaling, "rope_scaling", 1.0f);

    // stop token ids: "<|im_start|>", "<|im_end|>", "<|im_sep|>"
    SET_ARG(stop_token_ids, std::unordered_set<int32_t>({6, 7, 8}));
  } else {
    // llama 2
    LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
    LOAD_ARG_OR(n_layers, "num_hidden_layers", 32);
    LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
    LOAD_ARG_OR(intermediate_size, "intermediate_size", 11008);
    LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 2048);
    LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
    LOAD_ARG_OR(bos_token_id, "bos_token_id", 1);
    LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
    LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
    // LOAD_ARG_OR(rope_scaling, "rope_scaling", 1.0f);
  }

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
});

// Register tokenizer args since Yi is using sentencepiece tokenizer.
REGISTER_TOKENIZER_ARGS(Yi, [&] {
  SET_ARG(tokenizer_type, "sentencepiece");
  SET_ARG(vocab_file, "tokenizer.model");

  // set special tokens
  // ref to:
  // https://huggingface.co/01-ai/Yi-34B-Chat-4bits/blob/main/tokenizer_config.json
  const std::vector<SpecialToken> special_tokens({{"<unk>", 0},
                                                  {"<|startoftext|>", 1},
                                                  {"<|endoftext|>", 2},
                                                  {"<|im_start|>", 6},
                                                  {"<|im_end|>", 7},
                                                  {"<|im_sep|>", 8}});
  SET_ARG(special_tokens, special_tokens);
});

}  // namespace llm::hf
