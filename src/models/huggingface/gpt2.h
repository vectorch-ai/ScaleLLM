#pragma once

#include <c10/core/TensorOptions.h>
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

// gpt2 model compatible with huggingface weights

namespace llm::hf {

class GPT2MLPImpl : public torch::nn::Module {
 public:
  GPT2MLPImpl(const ModelArgs& args,
              const QuantArgs& quant_args,
              const ParallelArgs& parallel_args,
              const torch::TensorOptions& options) {
    const int64_t hidden_size = args.hidden_size();
    const int64_t intermediate_size = args.intermediate_size();

    act_ = Activation::get_act_func(args.hidden_act(), options.device());
    CHECK(act_ != nullptr);

    // register the weight parameter
    c_fc_ = register_module("c_fc",
                            ColumnParallelLinear(hidden_size,
                                                 intermediate_size,
                                                 /*bias=*/true,
                                                 /*gather_output=*/false,
                                                 quant_args,
                                                 parallel_args,
                                                 options));
    c_proj_ = register_module("c_proj",
                              RowParallelLinear(intermediate_size,
                                                hidden_size,
                                                /*bias=*/true,
                                                /*input_is_parallelized=*/true,
                                                quant_args,
                                                parallel_args,
                                                options));
  }

  torch::Tensor forward(torch::Tensor x) { return c_proj_(act_(c_fc_(x))); }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    // GPT-2 implementation uses Conv1D instead of Linear. As a result, we
    // need to transpose the weight.
    c_fc_->load_state_dict(state_dict.select_with_transform(
        "c_fc.", [](torch::Tensor tensor) { return tensor.t(); }));
    c_proj_->load_state_dict(state_dict.select_with_transform(
        "c_proj.", [](torch::Tensor tensor) { return tensor.t(); }));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    c_fc_->verify_loaded_weights(prefix + "c_fc.");
    c_proj_->verify_loaded_weights(prefix + "c_proj.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear c_fc_{nullptr};
  RowParallelLinear c_proj_{nullptr};

  ActFunc act_{nullptr};
};
TORCH_MODULE(GPT2MLP);

class GPT2AttentionImpl : public torch::nn::Module {
 public:
  GPT2AttentionImpl(const ModelArgs& args,
                    const QuantArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options,
                    AttentionHandler* handler) {
    const auto world_size = parallel_args.world_size();
    const int64_t n_local_heads = args.n_heads() / world_size;
    hidden_size_ = args.hidden_size();
    head_dim_ = args.head_dim();

    // register submodules
    c_attn_ = register_module("c_attn",
                              ColumnParallelLinear(hidden_size_,
                                                   3 * hidden_size_,
                                                   /*bias=*/true,
                                                   /*gather_output=*/false,
                                                   quant_args,
                                                   parallel_args,
                                                   options));

    c_proj_ = register_module("c_proj",
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
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    // (num_tokens, dim) x (dim, n_heads * head_dim)
    // => (num_tokens, n_heads * head_dim)
    auto qkv = c_attn_(x).chunk(/*chunks=*/3, /*dim=*/-1);
    DCHECK_EQ(qkv.size(), 3);
    // calculate attention, output: (num_tokens, n_local_heads * head_dim)
    auto output = atten_(qkv[0],
                         qkv[1],
                         qkv[2],
                         /*positions=*/torch::Tensor{},
                         kv_cache,
                         input_params);
    return c_proj_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // GPT-2 implementation uses Conv1D instead of Linear. As a result, we
    // need to transpose the weight.
    c_attn_->load_state_dict(state_dict.select_with_transform(
        "c_attn.", [](torch::Tensor tensor) { return tensor.t(); }));
    c_proj_->load_state_dict(state_dict.select_with_transform(
        "c_proj.", [](torch::Tensor tensor) { return tensor.t(); }));
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
  Attention atten_{nullptr};

  int64_t hidden_size_ = 0;
  int64_t head_dim_ = 0;
};
TORCH_MODULE(GPT2Attention);

class GPT2BlockImpl : public torch::nn::Module {
 public:
  GPT2BlockImpl(const ModelArgs& args,
                const QuantArgs& quant_args,
                const ParallelArgs& parallel_args,
                const torch::TensorOptions& options,
                AttentionHandler* handler) {
    // register submodules
    attn_ = register_module(
        "attn",
        GPT2Attention(args, quant_args, parallel_args, options, handler));
    mlp_ = register_module("mlp",
                           GPT2MLP(args, quant_args, parallel_args, options));
    ln_1_ = register_module("ln_1",
                            LayerNorm(args.hidden_size(),
                                      args.layer_norm_eps(),
                                      /*bias=*/true,
                                      options));
    ln_2_ = register_module("ln_2",
                            LayerNorm(args.hidden_size(),
                                      args.layer_norm_eps(),
                                      /*bias=*/true,
                                      options));
  }

  torch::Tensor forward(torch::Tensor x,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    // x = x + attn(ln1(x))
    // x = x + mlp(ln2(x))
    auto h = x + attn_(ln_1_(x), kv_cache, input_params);
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
  GPT2Attention attn_{nullptr};

  GPT2MLP mlp_{nullptr};

  LayerNorm ln_1_{nullptr};

  LayerNorm ln_2_{nullptr};
};
TORCH_MODULE(GPT2Block);

class GPT2ModelImpl : public torch::nn::Module {
 public:
  GPT2ModelImpl(const ModelArgs& args,
                const QuantArgs& quant_args,
                const ParallelArgs& parallel_args,
                const torch::TensorOptions& options) {
    // register submodules
    wte_ = register_module(
        "wte",
        ParallelEmbedding(
            args.vocab_size(), args.hidden_size(), parallel_args, options));
    wpe_ = register_module(
        "wpe",
        Embedding(args.max_position_embeddings(), args.hidden_size(), options));

    handler_ = AttentionHandler::create_handler(args, options);

    blocks_ = register_module("h", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block =
          GPT2Block(args, quant_args, parallel_args, options, handler_.get());
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
    auto h = wte_(tokens) + wpe_(positions);
    // TODO: set working space for attention handler
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, kv_caches[i], input_params);
    }
    return ln_f_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    wte_->load_state_dict(state_dict.select("wte."));
    wpe_->load_state_dict(state_dict.select("wpe."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.select("h." + std::to_string(i) + "."));
    }
    ln_f_->load_state_dict(state_dict.select("ln_f."));
  }

  void verify_loaded_weights() const {
    wte_->verify_loaded_weights("wte.");
    wpe_->verify_loaded_weights("wpe.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights("h." + std::to_string(i) + ".");
    }
    ln_f_->verify_loaded_weights("ln_f.");
  }

 private:
  // parameter members, must be registered
  ParallelEmbedding wte_{nullptr};

  // position Embedding
  Embedding wpe_{nullptr};

  // attention handler
  std::unique_ptr<AttentionHandler> handler_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<GPT2Block> layers_;

  LayerNorm ln_f_{nullptr};
};
TORCH_MODULE(GPT2Model);

class GPT2ForCausalLMImpl : public torch::nn::Module {
 public:
  GPT2ForCausalLMImpl(const ModelArgs& args,
                      const QuantArgs& quant_args,
                      const ParallelArgs& parallel_args,
                      const torch::TensorOptions& options) {
    // register submodules
    model_ = register_module(
        "model", GPT2Model(args, quant_args, parallel_args, options));

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
    model_->load_state_dict(state_dict);
    // TODO: share wte_ weight with lm_head_ to save memory
    lm_head_->load_state_dict(state_dict.select("wte."));
  }

  void verify_loaded_weights() const {
    model_->verify_loaded_weights();
    lm_head_->verify_loaded_weights("wte.");
  }

 private:
  // parameter members, must be registered
  GPT2Model model_{nullptr};

  ColumnParallelLinear lm_head_{nullptr};
};
TORCH_MODULE(GPT2ForCausalLM);

// register the model to make it available
REGISTER_CAUSAL_MODEL(gpt2, GPT2ForCausalLM);
REGISTER_MODEL_ARGS(gpt2, [&] {
  LOAD_ARG_OR(model_type, "model_type", "gpt2");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 50257);
  LOAD_ARG_OR(hidden_size, "n_embd", 768);
  LOAD_ARG_OR(n_layers, "n_layer", 12);
  LOAD_ARG_OR(n_heads, "n_head", 12);
  LOAD_ARG_OR(hidden_act, "activation_function", "gelu_new");
  LOAD_ARG_OR(max_position_embeddings, "n_positions", 1024);
  LOAD_ARG_OR(layer_norm_eps, "layer_norm_epsilon", 1e-5);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 50256);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 50256);

  LOAD_ARG_OR_FUNC(
      intermediate_size, "n_inner", [&] { return args->hidden_size() * 4; });

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
});

}  // namespace llm::hf
