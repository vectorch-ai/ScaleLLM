#pragma once

#include <torch/torch.h>

#include "layers/activation.h"
#include "layers/attention.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/normalization.h"
#include "memory/kv_cache.h"
#include "models/args.h"
#include "models/input_parameters.h"
#include "models/model_registry.h"

// gpt2 model compatible with huggingface weights

namespace llm::hf {

class GPT2MLPImpl : public torch::nn::Module {
 public:
  GPT2MLPImpl(const ModelArgs& args,
              const QuantizationArgs& quant_args,
              const ParallelArgs& parallel_args,
              torch::ScalarType dtype,
              const torch::Device& device) {
    const int64_t hidden_size = args.hidden_size();
    const int64_t intermediate_size = args.intermediate_size();

    act_ = Activation::get_act_func(args.hidden_act(), device);
    CHECK(act_ != nullptr);

    // register the weight parameter
    c_fc_ = register_module("c_fc",
                            ColumnParallelLinear(hidden_size,
                                                 intermediate_size,
                                                 /*bias=*/true,
                                                 /*gather_output=*/false,
                                                 quant_args,
                                                 parallel_args,
                                                 dtype,
                                                 device));
    c_proj_ = register_module("c_proj",
                              RowParallelLinear(intermediate_size,
                                                hidden_size,
                                                /*bias=*/true,
                                                /*input_is_parallelized=*/true,
                                                quant_args,
                                                parallel_args,
                                                dtype,
                                                device));
  }

  torch::Tensor forward(torch::Tensor x) { return c_proj_(act_(c_fc_(x))); }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    // GPT-2 implementation uses Conv1D instead of Linear. As a result, we
    // need to transpose the weight.
    c_fc_->load_state_dict(state_dict.select("c_fc.").set_tensor_transform(
        [this](const torch::Tensor& tensor) { return tensor.t(); }));
    c_proj_->load_state_dict(state_dict.select("c_proj.").set_tensor_transform(
        [this](const torch::Tensor& tensor) { return tensor.t(); }));
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
                    const QuantizationArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    torch::ScalarType dtype,
                    const torch::Device& device) {
    const auto world_size = parallel_args.world_size();
    const int64_t n_local_heads = args.n_heads() / world_size;
    hidden_size_ = args.hidden_size();
    head_dim_ = args.hidden_size() / args.n_heads();

    // register submodules
    c_attn_ = register_module("c_attn",
                              ColumnParallelLinear(hidden_size_,
                                                   3 * hidden_size_,
                                                   /*bias=*/true,
                                                   /*gather_output=*/false,
                                                   quant_args,
                                                   parallel_args,
                                                   dtype,
                                                   device));

    c_proj_ = register_module("c_proj",
                              RowParallelLinear(hidden_size_,
                                                hidden_size_,
                                                /*bias=*/true,
                                                /*input_is_parallelized=*/true,
                                                quant_args,
                                                parallel_args,
                                                dtype,
                                                device));

    // initialize positional embedding
    const int64_t rotary_dim =
        static_cast<int64_t>(head_dim_ * args.rotary_pct());
    // initialize attention
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    atten_ = register_module(
        "atten",
        Attention(
            n_local_heads, n_local_heads, head_dim_, scale, dtype, device));
  }

  torch::Tensor forward(torch::Tensor x,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    // (num_tokens, dim) x (dim, n_heads * head_dim)
    // => (num_tokens, n_heads * head_dim)
    auto qkv = c_attn_(x).chunk(/*chunks=*/3, /*dim=*/-1);
    DCHECK_EQ(qkv.size(), 3);
    // calculate attention, output: (num_tokens, n_local_heads * head_dim)
    auto output = atten_(qkv[0], qkv[1], qkv[2], kv_cache, input_params);
    return c_proj_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // GPT-2 implementation uses Conv1D instead of Linear. As a result, we
    // need to transpose the weight.
    c_attn_->load_state_dict(state_dict.select("c_attn.").set_tensor_transform(
        [this](const torch::Tensor& tensor) { return tensor.t(); }));
    c_proj_->load_state_dict(state_dict.select("c_proj.").set_tensor_transform(
        [this](const torch::Tensor& tensor) { return tensor.t(); }));
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
                const QuantizationArgs& quant_args,
                const ParallelArgs& parallel_args,
                torch::ScalarType dtype,
                const torch::Device& device) {
    // register submodules
    attn_ = register_module(
        "attn", GPT2Attention(args, quant_args, parallel_args, dtype, device));
    mlp_ = register_module(
        "mlp", GPT2MLP(args, quant_args, parallel_args, dtype, device));
    ln_1_ = register_module("ln_1",
                            LayerNorm(args.hidden_size(),
                                      args.layer_norm_eps(),
                                      /*bias=*/true,
                                      dtype,
                                      device));
    ln_2_ = register_module("ln_2",
                            LayerNorm(args.hidden_size(),
                                      args.layer_norm_eps(),
                                      /*bias=*/true,
                                      dtype,
                                      device));
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
                const QuantizationArgs& quant_args,
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
    wpe_ = register_module(
        "wpe",
        Embedding(
            args.max_position_embeddings(), args.hidden_size(), dtype, device));

    blocks_ = register_module("h", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = GPT2Block(args, quant_args, parallel_args, dtype, device);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    ln_f_ = register_module("ln_f",
                            LayerNorm(args.hidden_size(),
                                      args.layer_norm_eps(),
                                      /*bias=*/true,
                                      dtype,
                                      device));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = wte_(tokens) + wpe_(positions);
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

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<GPT2Block> layers_;

  LayerNorm ln_f_{nullptr};
};
TORCH_MODULE(GPT2Model);

class GPT2ForCausalLMImpl : public torch::nn::Module {
 public:
  GPT2ForCausalLMImpl(const ModelArgs& args,
                      const QuantizationArgs& quant_args,
                      const ParallelArgs& parallel_args,
                      torch::ScalarType dtype,
                      const torch::Device& device) {
    // register submodules
    model_ = register_module(
        "model", GPT2Model(args, quant_args, parallel_args, dtype, device));

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
    auto h = model_(tokens, positions, kv_caches, input_params);
    // select last token for each sequence
    h = h.index_select(/*dim=*/0, input_params.last_token_indicies);
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

inline bool load_gpt2_model_args(const nlohmann::json& data, ModelArgs* args) {
  // example config: https://huggingface.co/gpt2/blob/main/config.json
  // set default values for args explicitly with values from:
  // https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py#L142
  args->vocab_size() = 50257;
  args->hidden_size() = 768;
  args->n_layers() = 12;
  args->n_heads() = 12;
  args->intermediate_size() = 3072;
  args->hidden_act() = "gelu_new";
  args->max_position_embeddings() = 1024;
  args->layer_norm_eps() = 1e-5;
  args->bos_token_id() = 50256;
  args->eos_token_id() = 50256;

  if (data.contains("n_embd")) {
    args->hidden_size() = data["n_embd"].get<int64_t>();
  }
  if (data.contains("vocab_size")) {
    args->vocab_size() = data["vocab_size"].get<int64_t>();
  }
  if (data.contains("n_layer")) {
    args->n_layers() = data["n_layer"].get<int64_t>();
  }
  if (data.contains("n_head")) {
    args->n_heads() = data["n_head"].get<int64_t>();
  }
  if (data.contains("n_inner")) {
    args->intermediate_size() = data["n_inner"].get<int64_t>();
  } else {
    // set it to 4 times n_embd
    args->intermediate_size() = args->hidden_size() * 4;
  }
  if (data.contains("activation_function")) {
    args->hidden_act() = data["activation_function"].get<std::string>();
  }
  if (data.contains("n_positions")) {
    args->max_position_embeddings() = data["n_positions"].get<int64_t>();
  }
  if (data.contains("layer_norm_epsilon")) {
    args->layer_norm_eps() = data["layer_norm_epsilon"].get<float>();
  }
  if (data.contains("bos_token_id")) {
    args->bos_token_id() = data["bos_token_id"].get<int32_t>();
  }
  if (data.contains("eos_token_id")) {
    args->eos_token_id() = data["eos_token_id"].get<int32_t>();
  }
  return true;
}

// register the model to make it available
REGISTER_CAUSAL_MODEL(gpt2, GPT2ForCausalLM);
REGISTER_MODEL_ARGS_LOADER(gpt2, load_gpt2_model_args);

}  // namespace llm::hf
