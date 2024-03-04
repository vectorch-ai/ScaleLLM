#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>

#include "chat_template/coded_chat_template.h"
#include "layers/activation.h"
#include "layers/attention/attention_rope.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/normalization.h"
#include "memory/kv_cache.h"
#include "models/input_parameters.h"
#include "models/model_args.h"

// QWen model compatible with huggingface weights
// adopted from https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
namespace llm::hf {

class QWenMLPImpl : public torch::nn::Module {
 public:
  QWenMLPImpl(const ModelArgs& args,
              const QuantArgs& quant_args,
              const ParallelArgs& parallel_args,
              torch::ScalarType dtype,
              const torch::Device& device) {
    act_ = Activation::get_act_func("silu", device);
    CHECK(act_ != nullptr);

    const int64_t hidden_size = args.hidden_size();
    // the intermediate size is half of the size from the config
    // ref: https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py#L562
    const int64_t intermediate_size = args.intermediate_size() / 2;

    // register the weight parameter
    w1_w2_proj_ = register_module("gate_up_proj",
                                  ColumnParallelLinear(hidden_size,
                                                       intermediate_size * 2,
                                                       /*bias=*/false,
                                                       /*gather_output=*/false,
                                                       quant_args,
                                                       parallel_args,
                                                       dtype,
                                                       device));
    c_proj_ = register_module("c_proj",
                              RowParallelLinear(intermediate_size,
                                                hidden_size,
                                                /*bias=*/false,
                                                /*input_is_parallelized=*/true,
                                                quant_args,
                                                parallel_args,
                                                dtype,
                                                device));
  }

  torch::Tensor forward(torch::Tensor x) {
    auto gate_up_proj = w1_w2_proj_(x);
    auto chunks = gate_up_proj.chunk(/*chunks=*/2, /*dim=*/-1);
    return c_proj_(chunks[0] * act_(chunks[1]));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    w1_w2_proj_->load_state_dict(state_dict, {"w1.", "w2."});
    c_proj_->load_state_dict(state_dict.select("c_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    w1_w2_proj_->verify_loaded_weights(prefix + "[w1,w2].");
    c_proj_->verify_loaded_weights(prefix + "c_proj.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear w1_w2_proj_{nullptr};
  RowParallelLinear c_proj_{nullptr};

  ActFunc act_{nullptr};
};
TORCH_MODULE(QWenMLP);

class QWenAttentionImpl : public torch::nn::Module {
 public:
  QWenAttentionImpl(const ModelArgs& args,
                    const QuantArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    torch::ScalarType dtype,
                    const torch::Device& device) {
    const auto world_size = parallel_args.world_size();
    const int64_t hidden_size = args.hidden_size();
    const int64_t n_heads = args.n_heads();
    const int64_t head_dim = hidden_size / args.n_heads();
    const int64_t n_local_heads = n_heads / world_size;

    // register submodules
    c_attn_ = register_module("c_attn",
                              ColumnParallelLinear(hidden_size,
                                                   3 * hidden_size,
                                                   /*bias=*/true,
                                                   /*gather_output=*/false,
                                                   quant_args,
                                                   parallel_args,
                                                   dtype,
                                                   device));

    c_proj_ = register_module("c_proj",
                              RowParallelLinear(hidden_size,
                                                hidden_size,
                                                /*bias=*/false,
                                                /*input_is_parallelized=*/true,
                                                quant_args,
                                                parallel_args,
                                                dtype,
                                                device));

    // initialize attention
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    atten_ = register_module("atten",
                             AttentionWithRoPE(n_local_heads,
                                               n_local_heads,
                                               head_dim,
                                               scale,
                                               /*rotary_dim=*/head_dim,
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
    auto qkv = c_attn_(x).chunk(/*chunks=*/3, /*dim=*/-1);
    // calculate attention, output: (num_tokens, n_local_heads * head_dim)
    auto output =
        atten_(qkv[0], qkv[1], qkv[2], positions, kv_cache, input_params);
    return c_proj_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    c_attn_->load_state_dict(state_dict.select("c_attn."));
    c_proj_->load_state_dict(state_dict.select("c_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    c_attn_->verify_loaded_weights(prefix + "c_attn.");
    c_proj_->verify_loaded_weights(prefix + "c_proj.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear c_attn_{nullptr};

  RowParallelLinear c_proj_{nullptr};

  // module members without parameters
  AttentionWithRoPE atten_{nullptr};
};
TORCH_MODULE(QWenAttention);

class QWenBlockImpl : public torch::nn::Module {
 public:
  QWenBlockImpl(const ModelArgs& args,
                const QuantArgs& quant_args,
                const ParallelArgs& parallel_args,
                torch::ScalarType dtype,
                const torch::Device& device) {
    // register submodules
    attn_ = register_module(
        "attn", QWenAttention(args, quant_args, parallel_args, dtype, device));
    mlp_ = register_module(
        "mlp", QWenMLP(args, quant_args, parallel_args, dtype, device));
    ln_1_ = register_module(
        "ln_1",
        RMSNorm(args.hidden_size(), args.rms_norm_eps(), dtype, device));
    ln_2_ = register_module(
        "ln_2",
        RMSNorm(args.hidden_size(), args.rms_norm_eps(), dtype, device));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    auto h = x + attn_(ln_1_(x), positions, kv_cache, input_params);
    return h + mlp_(ln_2_(h));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    attn_->load_state_dict(state_dict.select("attn."));
    mlp_->load_state_dict(state_dict.select("mlp."));
    ln_1_->load_state_dict(state_dict.select("ln_1."));
    ln_2_->load_state_dict(state_dict.select("ln_2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    attn_->verify_loaded_weights(prefix + "attn.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    ln_1_->verify_loaded_weights(prefix + "ln_1.");
    ln_2_->verify_loaded_weights(prefix + "ln_2.");
  }

 private:
  // parameter members, must be registered
  QWenAttention attn_{nullptr};

  QWenMLP mlp_{nullptr};

  RMSNorm ln_1_{nullptr};

  RMSNorm ln_2_{nullptr};
};
TORCH_MODULE(QWenBlock);

class QWenModelImpl : public torch::nn::Module {
 public:
  QWenModelImpl(const ModelArgs& args,
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
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = QWenBlock(args, quant_args, parallel_args, dtype, device);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    ln_f_ = register_module(
        "ln_f",
        RMSNorm(args.hidden_size(), args.rms_norm_eps(), dtype, device));
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
    return ln_f_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    wte_->load_state_dict(state_dict.select("wte."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.select("h." + std::to_string(i) + "."));
    }
    ln_f_->load_state_dict(state_dict.select("ln_f."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    wte_->verify_loaded_weights(prefix + "wte.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "h." + std::to_string(i) +
                                        ".");
    }
    ln_f_->verify_loaded_weights(prefix + "ln_f.");
  }

 private:
  // parameter members, must be registered
  ParallelEmbedding wte_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<QWenBlock> layers_;

  RMSNorm ln_f_{nullptr};
};
TORCH_MODULE(QWenModel);

class QWenForCausalLMImpl : public torch::nn::Module {
 public:
  QWenForCausalLMImpl(const ModelArgs& args,
                      const QuantArgs& quant_args,
                      const ParallelArgs& parallel_args,
                      torch::ScalarType dtype,
                      const torch::Device& device) {
    // register submodules
    transformer_ = register_module(
        "transformer",
        QWenModel(args, quant_args, parallel_args, dtype, device));

    lm_head_ = register_module("lm_head",
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
  QWenModel transformer_{nullptr};

  ColumnParallelLinear lm_head_{nullptr};
};
TORCH_MODULE(QWenForCausalLM);

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
  // LOAD_ARG_OR(bos_token_id, "bos_token_id", 1);
  // LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  // LOAD_ARG_OR(rope_scaling, "rope_scaling", 1.0f);

  // stop token ids: "<|endoftext|>", "<|im_start|>", "<|im_end|>"
  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>({151643, 151644, 151645}));
});

// Register tokenizer args since Qwen is using tiktoken tokenizer.
REGISTER_TOKENIZER_ARGS(qwen, [&] {
  SET_ARG(tokenizer_type, "tiktoken");
  // adapted from
  // https://huggingface.co/Qwen/Qwen-14B-Chat-Int4/blob/main/tokenization_qwen.py
  SET_ARG(vocab_file, "qwen.tiktoken");

  // set special tokens
  std::vector<std::string> special_tokens(
      {"<|endoftext|>", "<|im_start|>", "<|im_end|>"});
  for (int32_t i = 0; i < 205; ++i) {
    special_tokens.push_back("<|extra_" + std::to_string(i) + "|>");
  }
  SET_ARG(special_tokens, special_tokens);

  // set regex pattern for tiktoken tokenizer.
  const std::string pattern =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+[^\S]|\s+)";
  SET_ARG(pattern, pattern);
  SET_ARG(special_start_id, 151643);
});

}  // namespace llm::hf