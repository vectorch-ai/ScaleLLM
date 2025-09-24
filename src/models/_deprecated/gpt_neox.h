#pragma once

#include <torch/torch.h>

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
#include "module/module_list.h"
// gpt-neox model compatible with huggingface weights

namespace llm::hf {

class GPTNeoXMLPImpl : public Module {
 public:
  GPTNeoXMLPImpl(const ModelArgs& args,
                 const QuantArgs& quant_args,
                 const ParallelArgs& parallel_args,
                 const torch::TensorOptions& options) {
    const int64_t hidden_size = args.hidden_size();
    const int64_t intermediate_size = args.intermediate_size();

    act_ = Activation::get_act_func(args.hidden_act(), options.device());
    CHECK(act_ != nullptr);

    // register the weight parameter
    dense_h_to_4h_ =
        register_module("dense_h_to_4h",
                        ColumnParallelLinear(hidden_size,
                                             intermediate_size,
                                             /*bias=*/true,
                                             /*gather_output=*/false,
                                             quant_args,
                                             parallel_args,
                                             options));
    dense_4h_to_h_ =
        register_module("dense_4h_to_h",
                        RowParallelLinear(intermediate_size,
                                          hidden_size,
                                          /*bias=*/true,
                                          /*input_is_parallelized=*/true,
                                          quant_args,
                                          parallel_args,
                                          options));
  }

  torch::Tensor forward(torch::Tensor x) {
    return dense_4h_to_h_(act_(dense_h_to_4h_(x)));
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

  ActFunc act_{nullptr};
};
LLM_MODULE(GPTNeoXMLP);

class GPTNeoXAttentionImpl : public Module {
 public:
  GPTNeoXAttentionImpl(const ModelArgs& args,
                       const QuantArgs& quant_args,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options,
                       AttentionHandler* handler) {
    const auto world_size = parallel_args.world_size();
    const int64_t n_local_heads = args.n_heads() / world_size;
    hidden_size_ = args.hidden_size();
    head_dim_ = args.head_dim();

    // register submodules
    query_key_value_ =
        register_module("query_key_value",
                        ColumnParallelLinear(hidden_size_,
                                             3 * hidden_size_,
                                             /*bias=*/true,
                                             /*gather_output=*/false,
                                             quant_args,
                                             parallel_args,
                                             options));

    dense_ = register_module("dense",
                             RowParallelLinear(hidden_size_,
                                               hidden_size_,
                                               /*bias=*/true,
                                               /*input_is_parallelized=*/true,
                                               quant_args,
                                               parallel_args,
                                               options));

    // initialize attention
    atten_ = register_module(
        "atten", Attention(n_local_heads, n_local_heads, head_dim_, handler));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    // (num_tokens, dim) x (dim, n_heads * head_dim)
    // => (num_tokens, n_heads * head_dim)
    auto qkv = query_key_value_(x).chunk(/*chunks=*/3, /*dim=*/-1);
    DCHECK_EQ(qkv.size(), 3);
    // calculate attention, output: (num_tokens, n_local_heads * head_dim)
    auto output =
        atten_(qkv[0], qkv[1], qkv[2], positions, kv_cache, input_params);
    return dense_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    query_key_value_->load_state_dict(state_dict.select("query_key_value."),
                                      [this](const torch::Tensor& tensor) {
                                        return reshape_qkv_tensor(tensor);
                                      });
    dense_->load_state_dict(state_dict.select("dense."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    query_key_value_->verify_loaded_weights(prefix + "query_key_value.");
    dense_->verify_loaded_weights(prefix + "dense.");
  }

 private:
  torch::Tensor reshape_qkv_tensor(const torch::Tensor& tensor) {
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
  ColumnParallelLinear query_key_value_{nullptr};

  RowParallelLinear dense_{nullptr};

  // module members without parameters
  Attention atten_{nullptr};

  int64_t hidden_size_ = 0;
  int64_t head_dim_ = 0;
};
LLM_MODULE(GPTNeoXAttention);

class GPTNeoXLayerImpl : public Module {
 public:
  GPTNeoXLayerImpl(uint32_t layer_id,
                   const ModelArgs& args,
                   const QuantArgs& quant_args,
                   const ParallelArgs& parallel_args,
                   const torch::TensorOptions& options,
                   AttentionHandler* handler)
      : use_parallel_residual_(args.use_parallel_residual()) {
    // register submodules
    attention_ = register_module(
        "attention",
        GPTNeoXAttention(args, quant_args, parallel_args, options, handler));
    mlp_ = register_module(
        "mlp", GPTNeoXMLP(args, quant_args, parallel_args, options));
    input_layernorm_ = register_module("input_layernorm",
                                       LayerNorm(args.hidden_size(),
                                                 args.layer_norm_eps(),
                                                 /*bias=*/true,
                                                 options));
    post_attention_layernorm_ = register_module("post_attention_layernorm",
                                                LayerNorm(args.hidden_size(),
                                                          args.layer_norm_eps(),
                                                          /*bias=*/true,
                                                          options));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    auto attn_output =
        attention_(input_layernorm_(x), positions, kv_cache, input_params);

    if (use_parallel_residual_) {
      // parallel residual: x = x + attn(ln1(x)) + mlp(ln2(x))
      return x + attn_output + mlp_(post_attention_layernorm_(x));
    }

    // x = x + attn(ln1(x))
    // x = x + mlp(ln2(x))
    auto h = x + attn_output;
    return h + mlp_(post_attention_layernorm_(h));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    attention_->load_state_dict(state_dict.select("attention."));
    mlp_->load_state_dict(state_dict.select("mlp."));
    input_layernorm_->load_state_dict(state_dict.select("input_layernorm."));
    post_attention_layernorm_->load_state_dict(
        state_dict.select("post_attention_layernorm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    attention_->verify_loaded_weights(prefix + "attention.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    input_layernorm_->verify_loaded_weights(prefix + "input_layernorm.");
    post_attention_layernorm_->verify_loaded_weights(
        prefix + "post_attention_layernorm.");
  }

 private:
  // parameter members, must be registered
  GPTNeoXAttention attention_{nullptr};

  GPTNeoXMLP mlp_{nullptr};

  LayerNorm input_layernorm_{nullptr};

  LayerNorm post_attention_layernorm_{nullptr};

  bool use_parallel_residual_;
};
LLM_MODULE(GPTNeoXLayer);

class GPTNeoXModelImpl : public Module {
 public:
  GPTNeoXModelImpl(const ModelArgs& args,
                   const QuantArgs& quant_args,
                   const ParallelArgs& parallel_args,
                   const torch::TensorOptions& options) {
    // register submodules
    embed_in_ = register_module(
        "embed_in",
        ParallelEmbedding(
            args.vocab_size(), args.hidden_size(), parallel_args, options));

    handler_ = AttentionHandler::create_handler_with_rope(
        args, /*interleaved=*/false, options);

    blocks_ = register_module("layers", ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = GPTNeoXLayer(
          i, args, quant_args, parallel_args, options, handler_.get());
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    final_layer_norm_ = register_module("final_layer_norm",
                                        LayerNorm(args.hidden_size(),
                                                  args.layer_norm_eps(),
                                                  /*bias=*/true,
                                                  options));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = embed_in_(tokens);

    // TODO: set working space for attention handler
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, kv_caches[i], input_params);
    }
    return final_layer_norm_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embed_in_->load_state_dict(state_dict.select("embed_in."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.select("layers." + std::to_string(i) + "."));
    }
    final_layer_norm_->load_state_dict(state_dict.select("final_layer_norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embed_in_->verify_loaded_weights(prefix + "embed_in.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    final_layer_norm_->verify_loaded_weights(prefix + "final_layer_norm.");
  }

 private:
  // parameter members, must be registered
  ParallelEmbedding embed_in_{nullptr};

  // attention handler
  std::unique_ptr<AttentionHandler> handler_{nullptr};

  ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<GPTNeoXLayer> layers_;

  LayerNorm final_layer_norm_{nullptr};
};
LLM_MODULE(GPTNeoXModel);

class GPTNeoXForCausalLMImpl : public Module {
 public:
  GPTNeoXForCausalLMImpl(const ModelArgs& args,
                         const QuantArgs& quant_args,
                         const ParallelArgs& parallel_args,
                         const torch::TensorOptions& options) {
    // register submodules
    gpt_neox_ = register_module(
        "gpt_neox", GPTNeoXModel(args, quant_args, parallel_args, options));

    embed_out_ = register_module("embed_out",
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
    return gpt_neox_(tokens, positions, kv_caches, input_params);
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
    return embed_out_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    gpt_neox_->load_state_dict(state_dict.select("gpt_neox."));
    embed_out_->load_state_dict(state_dict.select("embed_out."));
  }

  void verify_loaded_weights() const {
    gpt_neox_->verify_loaded_weights("gpt_neox.");
    embed_out_->verify_loaded_weights("embed_out.");
  }

 private:
  // parameter members, must be registered
  GPTNeoXModel gpt_neox_{nullptr};

  ColumnParallelLinear embed_out_{nullptr};
};
LLM_MODULE(GPTNeoXForCausalLM);

// register the model to make it available
REGISTER_CAUSAL_MODEL(gpt_neox, GPTNeoXForCausalLM);
REGISTER_MODEL_ARGS(gpt_neox, [&] {
  // example config:
  // https://huggingface.co/EleutherAI/gpt-neox-20b/blob/main/config.json set
  // set default values for args explicitly with values from:
  // https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/configuration_gpt_neox.py#L106
  LOAD_ARG_OR(model_type, "model_type", "gpt_neox");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 50432);
  LOAD_ARG_OR(hidden_size, "hidden_size", 6144);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 44);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 64);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 24576);
  LOAD_ARG_OR(hidden_act, "hidden_act", "gelu");
  LOAD_ARG_OR(rotary_pct, "rotary_pct", 0.25);
  LOAD_ARG_OR(rope_theta, "rotary_emb_base", 10000.0f);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 2048);
  LOAD_ARG_OR(layer_norm_eps, "layer_norm_eps", 1e-5);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
  LOAD_ARG_OR(use_parallel_residual, "use_parallel_residual", true);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
});

}  // namespace llm::hf
