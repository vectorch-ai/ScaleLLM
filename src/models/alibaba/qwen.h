#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>

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
// QWen model compatible with huggingface weights
// Adapted from https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
namespace llm::hf {

class QWenMLPImpl : public Module {
 public:
  QWenMLPImpl(const ModelArgs& args,
              const QuantArgs& quant_args,
              const ParallelArgs& parallel_args,
              const torch::TensorOptions& options) {
    act_ = Activation::get_act_func("silu", options.device());
    CHECK(act_ != nullptr);

    const int64_t hidden_size = args.hidden_size();
    // the intermediate size is half of the size from the config
    // ref: https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py#L562
    const int64_t intermediate_size = args.intermediate_size() / 2;

    // register the weight parameter
    gate_up_proj_ = register_module(
        "gate_up_proj",
        FusedColumnParallelLinear(
            hidden_size,
            std::vector<int64_t>{intermediate_size, intermediate_size},
            std::vector<std::string>{"w1.", "w2."},
            /*bias=*/false,
            /*gather_output=*/false,
            quant_args,
            parallel_args,
            options),
        /*selector=*/nullptr);
    c_proj_ = register_module("c_proj",
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
    return c_proj_(gate_up[0] * act_(gate_up[1]));
  }

 private:
  // parameter members, must be registered
  FusedColumnParallelLinear gate_up_proj_{nullptr};
  RowParallelLinear c_proj_{nullptr};

  ActFunc act_{nullptr};
};
LLM_MODULE(QWenMLP);

class QWenAttentionImpl : public Module {
 public:
  QWenAttentionImpl(const ModelArgs& args,
                    const QuantArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options,
                    AttentionHandler* handler) {
    const auto world_size = parallel_args.world_size();
    const int64_t hidden_size = args.hidden_size();
    const int64_t n_heads = args.n_heads();
    const int64_t head_dim = args.head_dim();
    const int64_t n_local_heads = n_heads / world_size;

    // register submodules
    c_attn_ = register_module("c_attn",
                              ColumnParallelLinear(hidden_size,
                                                   3 * hidden_size,
                                                   /*bias=*/true,
                                                   /*gather_output=*/false,
                                                   quant_args,
                                                   parallel_args,
                                                   options));

    c_proj_ = register_module("c_proj",
                              RowParallelLinear(hidden_size,
                                                hidden_size,
                                                /*bias=*/false,
                                                /*input_is_parallelized=*/true,
                                                quant_args,
                                                parallel_args,
                                                options));

    // initialize attention
    atten_ = register_module(
        "atten", Attention(n_local_heads, n_local_heads, head_dim, handler));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    // (num_tokens, dim) x (dim, n_local_heads * head_dim)
    // => (num_tokens, n_local_heads * head_dim)
    auto qkv = c_attn_(x).chunk(/*chunks=*/3, /*dim=*/-1);
    // calculate attention, output: (num_tokens, n_local_heads * head_dim)
    auto output =
        atten_(qkv[0], qkv[1], qkv[2], positions, kv_cache, input_params);
    return c_proj_(output);
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear c_attn_{nullptr};

  RowParallelLinear c_proj_{nullptr};

  // module members without parameters
  Attention atten_{nullptr};
};
LLM_MODULE(QWenAttention);

class QWenBlockImpl : public Module {
 public:
  QWenBlockImpl(const ModelArgs& args,
                const QuantArgs& quant_args,
                const ParallelArgs& parallel_args,
                const torch::TensorOptions& options,
                AttentionHandler* handler) {
    // register submodules
    attn_ = register_module(
        "attn",
        QWenAttention(args, quant_args, parallel_args, options, handler));
    mlp_ = register_module("mlp",
                           QWenMLP(args, quant_args, parallel_args, options));
    ln_1_ = register_module(
        "ln_1", RMSNorm(args.hidden_size(), args.rms_norm_eps(), options));
    ln_2_ = register_module(
        "ln_2", RMSNorm(args.hidden_size(), args.rms_norm_eps(), options));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    auto h = x + attn_(ln_1_(x), positions, kv_cache, input_params);
    return h + mlp_(ln_2_(h));
  }

 private:
  // parameter members, must be registered
  QWenAttention attn_{nullptr};

  QWenMLP mlp_{nullptr};

  RMSNorm ln_1_{nullptr};

  RMSNorm ln_2_{nullptr};
};
LLM_MODULE(QWenBlock);

class QWenModelImpl : public Module {
 public:
  QWenModelImpl(const ModelArgs& args,
                const QuantArgs& quant_args,
                const ParallelArgs& parallel_args,
                const torch::TensorOptions& options) {
    // register submodules
    wte_ = register_module(
        "wte",
        ParallelEmbedding(
            args.vocab_size(), args.hidden_size(), parallel_args, options));

    handler_ = AttentionHandler::create_handler_with_rope(
        args, /*interleaved=*/false, options);

    blocks_ = register_module("h", ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block =
          QWenBlock(args, quant_args, parallel_args, options, handler_.get());
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    ln_f_ = register_module(
        "ln_f", RMSNorm(args.hidden_size(), args.rms_norm_eps(), options));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = wte_(tokens);

    // TODO: set working space for attention handler
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, kv_caches[i], input_params);
    }
    return ln_f_(h);
  }

 private:
  // parameter members, must be registered
  ParallelEmbedding wte_{nullptr};

  // attention handler
  std::unique_ptr<AttentionHandler> handler_{nullptr};

  ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<QWenBlock> layers_;

  RMSNorm ln_f_{nullptr};
};
LLM_MODULE(QWenModel);

class QWenForCausalLMImpl : public Module {
 public:
  QWenForCausalLMImpl(const ModelArgs& args,
                      const QuantArgs& quant_args,
                      const ParallelArgs& parallel_args,
                      const torch::TensorOptions& options) {
    // register submodules
    transformer_ = register_module(
        "transformer", QWenModel(args, quant_args, parallel_args, options));

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
    return transformer_(tokens, positions, kv_caches, input_params);
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
  QWenModel transformer_{nullptr};

  ColumnParallelLinear lm_head_{nullptr};
};
LLM_MODULE(QWenForCausalLM);

class QwenChatTemplate final : public CodedChatTemplate {
 public:
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
REGISTER_CAUSAL_MODEL(qwen, QWenForCausalLM);
REGISTER_DEFAULT_CHAT_TEMPLATE(qwen, QwenChatTemplate);
// register the model args
// example config:
// https://huggingface.co/Qwen/Qwen-7B/blob/main/config.json
REGISTER_MODEL_ARGS(qwen, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 151936);
  LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 32);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(no_bias, "no_bias", true);
  // LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 22016);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(layer_norm_eps, "layer_norm_epsilon", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  // stop token ids: "<|im_start|>", "<|im_end|>"
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({151644, 151645}));
});

// Register tokenizer args since Qwen is using tiktoken tokenizer.
REGISTER_TOKENIZER_ARGS(qwen, [&] {
  SET_ARG(tokenizer_type, "tiktoken");
  // adapted from
  // https://huggingface.co/Qwen/Qwen-14B-Chat-Int4/blob/main/tokenization_qwen.py
  SET_ARG(vocab_file, "qwen.tiktoken");

  // set special tokens
  std::vector<SpecialToken> special_tokens;
  int32_t next_id = 151643;
  special_tokens.emplace_back("<|endoftext|>", next_id++);
  special_tokens.emplace_back("<|im_start|>", next_id++);
  special_tokens.emplace_back("<|im_end|>", next_id++);
  for (int32_t i = 0; i < 205; ++i) {
    special_tokens.emplace_back("<|extra_" + std::to_string(i) + "|>",
                                next_id++);
  }
  SET_ARG(special_tokens, special_tokens);

  // set regex pattern for tiktoken tokenizer.
  const std::string pattern =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+[^\S]|\s+)";
  SET_ARG(pattern, pattern);
});

}  // namespace llm::hf
