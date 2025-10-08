#pragma once

#include <torch/torch.h>

#include "layers/activation.h"
#include "layers/attention/attention.h"
#include "layers/attention/handler.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/module/module.h"
#include "layers/module/module_holder.h"
#include "layers/module/module_list.h"
#include "layers/normalization.h"
#include "memory/kv_cache.h"
#include "models/model_args.h"
#include "models/model_registry.h"
#include "models/parameters.h"
// GPTJ model compatible with huggingface weights

namespace llm::hf {

class GPTJMLPImpl : public Module {
 public:
  GPTJMLPImpl(const ModelArgs& args,
              const QuantArgs& quant_args,
              const ParallelArgs& parallel_args,
              const torch::TensorOptions& options) {
    const int64_t hidden_size = args.hidden_size();
    const int64_t intermediate_size = args.intermediate_size();

    act_ = Activation::get_act_func(args.hidden_act(), options.device());
    CHECK(act_ != nullptr);

    // register the weight parameter
    fc_in_ = register_module("fc_in",
                             ColumnParallelLinear(hidden_size,
                                                  intermediate_size,
                                                  /*bias=*/true,
                                                  /*gather_output=*/false,
                                                  quant_args,
                                                  parallel_args,
                                                  options));
    fc_out_ = register_module("fc_out",
                              RowParallelLinear(intermediate_size,
                                                hidden_size,
                                                /*bias=*/true,
                                                /*input_is_parallelized=*/true,
                                                quant_args,
                                                parallel_args,
                                                options));
  }

  torch::Tensor forward(torch::Tensor x) { return fc_out_(act_(fc_in_(x))); }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    fc_in_->load_state_dict(state_dict.select("fc_in."));
    fc_out_->load_state_dict(state_dict.select("fc_out."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    fc_in_->verify_loaded_weights(prefix + "fc_in.");
    fc_out_->verify_loaded_weights(prefix + "fc_out.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear fc_in_{nullptr};
  RowParallelLinear fc_out_{nullptr};

  ActFunc act_{nullptr};
};
LLM_MODULE(GPTJMLP);

class GPTJAttentionImpl : public Module {
 public:
  GPTJAttentionImpl(const ModelArgs& args,
                    const QuantArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options,
                    AttentionHandler* handler) {
    const int64_t n_local_heads = args.n_heads() / parallel_args.world_size();
    const int64_t hidden_size = args.hidden_size();
    const int64_t head_dim = args.head_dim();

    // register submodules
    qkv_proj_ = register_module("qkv_proj",
                                ColumnParallelLinear(hidden_size,
                                                     3 * hidden_size,
                                                     /*bias=*/false,
                                                     /*gather_output=*/false,
                                                     quant_args,
                                                     parallel_args,
                                                     options));

    out_proj_ =
        register_module("out_proj",
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
    // (num_tokens, dim) x (dim, n_heads * head_dim)
    // => (num_tokens, n_heads * head_dim)
    auto qkv = qkv_proj_(x).chunk(/*chunks=*/3, /*dim=*/-1);
    DCHECK_EQ(qkv.size(), 3);
    // calculate attention, output: (num_tokens, n_local_heads * head_dim)
    auto output =
        atten_(qkv[0], qkv[1], qkv[2], positions, kv_cache, input_params);
    return out_proj_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    qkv_proj_->load_state_dict(state_dict, {"q_proj.", "k_proj.", "v_proj."});
    out_proj_->load_state_dict(state_dict.select("out_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    qkv_proj_->verify_loaded_weights(prefix + "[q_proj,k_proj,v_proj].");
    out_proj_->verify_loaded_weights(prefix + "out_proj.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear qkv_proj_{nullptr};

  RowParallelLinear out_proj_{nullptr};

  // module members without parameters
  Attention atten_{nullptr};
};
LLM_MODULE(GPTJAttention);

class GPTJBlockImpl : public Module {
 public:
  GPTJBlockImpl(const ModelArgs& args,
                const QuantArgs& quant_args,
                const ParallelArgs& parallel_args,
                const torch::TensorOptions& options,
                AttentionHandler* handler) {
    // register submodules
    attn_ = register_module(
        "attn",
        GPTJAttention(args, quant_args, parallel_args, options, handler));
    mlp_ = register_module("mlp",
                           GPTJMLP(args, quant_args, parallel_args, options));
    ln_1_ = register_module("ln_1",
                            LayerNorm(args.hidden_size(),
                                      args.layer_norm_eps(),
                                      /*bias=*/true,
                                      options));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    // x = x + attn(ln1(x)) + mlp(ln1(x))
    const auto h = ln_1_(x);
    const auto attn_output = attn_(h, positions, kv_cache, input_params);
    const auto mlp_output = mlp_(h);
    return x + attn_output + mlp_output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    attn_->load_state_dict(state_dict.select("attn."));
    mlp_->load_state_dict(state_dict.select("mlp."));
    ln_1_->load_state_dict(state_dict.select("ln_1."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    attn_->verify_loaded_weights(prefix + "attn.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    ln_1_->verify_loaded_weights(prefix + "ln_1.");
  }

 private:
  // parameter members, must be registered
  GPTJAttention attn_{nullptr};

  GPTJMLP mlp_{nullptr};

  LayerNorm ln_1_{nullptr};
};
LLM_MODULE(GPTJBlock);

class GPTJModelImpl : public Module {
 public:
  GPTJModelImpl(const ModelArgs& args,
                const QuantArgs& quant_args,
                const ParallelArgs& parallel_args,
                const torch::TensorOptions& options) {
    // register submodules
    wte_ = register_module(
        "wte",
        ParallelEmbedding(
            args.vocab_size(), args.hidden_size(), parallel_args, options));

    handler_ = AttentionHandler::create_handler_with_rope(
        args, /*interleaved=*/true, options);

    blocks_ = register_module("h", ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block =
          GPTJBlock(args, quant_args, parallel_args, options, handler_.get());
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    ln_f_ = register_module("ln_f",
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
    auto h = wte_(tokens);

    // TODO: set working space for attention handler
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

  // attention handler
  std::unique_ptr<AttentionHandler> handler_{nullptr};

  ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<GPTJBlock> layers_;

  LayerNorm ln_f_{nullptr};
};
LLM_MODULE(GPTJModel);

class GPTJForCausalLMImpl : public Module {
 public:
  GPTJForCausalLMImpl(const ModelArgs& args,
                      const QuantArgs& quant_args,
                      const ParallelArgs& parallel_args,
                      const torch::TensorOptions& options) {
    // register submodules
    transformer_ = register_module(
        "transformer", GPTJModel(args, quant_args, parallel_args, options));

    lm_head_ = register_module("lm_head",
                               ColumnParallelLinear(args.hidden_size(),
                                                    args.vocab_size(),
                                                    /*bias=*/true,
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
  GPTJModel transformer_{nullptr};

  ColumnParallelLinear lm_head_{nullptr};
};
LLM_MODULE(GPTJForCausalLM);

// register the model to make it available
REGISTER_CAUSAL_MODEL(gptj, GPTJForCausalLM);
// REGISTER_MODEL_ARGS_LOADER(gptj, load_gptj_model_args);
REGISTER_MODEL_ARGS(gptj, [&] {
  // example config:
  // https://huggingface.co/EleutherAI/gpt-j-6b/blob/main/config.json set
  // default values for args explicitly with values from:
  // https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/configuration_gptj.py#L98
  LOAD_ARG_OR(model_type, "model_type", "gptj");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 50400);
  LOAD_ARG_OR(hidden_size, "n_embd", 4096);
  LOAD_ARG_OR(n_layers, "n_layer", 28);
  LOAD_ARG_OR(n_heads, "n_head", 16);
  LOAD_ARG_OR(rotary_dim, "rotary_dim", 64);
  LOAD_ARG_OR(hidden_act, "activation_function", "gelu_new");
  LOAD_ARG_OR(max_position_embeddings, "n_positions", 2048);
  LOAD_ARG_OR(layer_norm_eps, "layer_norm_epsilon", 1e-5);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 50256);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 50256);

  LOAD_ARG_OR_FUNC(intermediate_size, "n_inner", [&] {
    // set it to 4 times n_embd
    return args->hidden_size() * 4;
  });

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
});
}  // namespace llm::hf
