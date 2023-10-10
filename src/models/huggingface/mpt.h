#pragma once

#include <torch/torch.h>

#include "layers/activation.h"
#include "layers/attention.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/normalization.h"
#include "layers/pos_embedding.h"
#include "memory/kv_cache.h"
#include "models/args.h"
#include "models/input_parameters.h"

// mpt model compatible with huggingface weights
namespace llm::hf {

class MPTMLPImpl : public torch::nn::Module {
 public:
  MPTMLPImpl(const ModelArgs& args,
             const QuantizationArgs& quant_args,
             const ParallelArgs& parallel_args,
             torch::ScalarType dtype,
             const torch::Device& device) {
    act_ = Activation::get_act_func("silu", device);
    CHECK(act_ != nullptr);

    const int64_t hidden_size = args.hidden_size();
    const int64_t intermediate_size = args.intermediate_size();

    // register the weight parameter
    up_proj_ = register_module("up_proj",
                               ColumnParallelLinear(hidden_size,
                                                    intermediate_size,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args,
                                                    dtype,
                                                    device));
    down_proj_ =
        register_module("down_proj",
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
TORCH_MODULE(MPTMLP);

class MPTAttentionImpl : public torch::nn::Module {
 public:
  MPTAttentionImpl(const ModelArgs& args,
                   const QuantizationArgs& quant_args,
                   const ParallelArgs& parallel_args,
                   torch::ScalarType dtype,
                   const torch::Device& device) {
    const auto world_size = parallel_args.world_size();
    const int64_t hidden_size = args.hidden_size();
    const int64_t n_heads = args.n_heads();
    const int64_t head_dim = hidden_size / args.n_heads();
    const int64_t n_local_heads = n_heads / world_size;

    // register submodules
    wqkv_ = register_module("Wqkv",
                            ColumnParallelLinear(hidden_size,
                                                 3 * hidden_size,
                                                 /*bias=*/false,
                                                 /*gather_output=*/false,
                                                 quant_args,
                                                 parallel_args,
                                                 dtype,
                                                 device));

    // self.clip_qkv = config.attn_config["clip_qkv"]
    //     self.qk_ln = config.attn_config["qk_ln"]
    //     self.alibi_bias_max = config.attn_config["alibi_bias_max"]
    //     assert not config.attn_config["prefix_lm"]
    //     assert config.attn_config["alibi"]

    out_proj_ =
        register_module("out_proj",
                        RowParallelLinear(hidden_size,
                                          hidden_size,
                                          /*bias=*/false,
                                          /*input_is_parallelized=*/true,
                                          quant_args,
                                          parallel_args,
                                          dtype,
                                          device));

    // initialize positional embedding
    // TODO: use ALiBi positional embedding
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
    const auto num_tokens = x.size(0);
    // (num_tokens, dim) x (dim, n_heads * head_dim)
    // => (num_tokens, n_heads * head_dim)
    auto qkv = wqkv_(x);
    // TODO: apply clip_qkv if not None

    auto chunks = qkv.chunk(/*chunks=*/3, /*dim=*/-1);
    DCHECK_EQ(chunks.size(), 3);
    // calculate attention, output: (num_tokens, n_local_heads * head_dim)
    auto output =
        atten_(qkv[0], qkv[1], qkv[2], positions, kv_cache, input_params);
    return out_proj_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    wqkv_->load_state_dict(state_dict.select("Wqkv."));
    out_proj_->load_state_dict(state_dict.select("out_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    wqkv_->verify_loaded_weights(prefix + "Wqkv.");
    out_proj_->verify_loaded_weights(prefix + "out_proj.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear wqkv_{nullptr};

  RowParallelLinear out_proj_{nullptr};

  // module members without parameters
  AttentionWithRoPE atten_{nullptr};
};
TORCH_MODULE(MPTAttention);

class MPTBlockImpl : public torch::nn::Module {
 public:
  MPTBlockImpl(const ModelArgs& args,
               const QuantizationArgs& quant_args,
               const ParallelArgs& parallel_args,
               torch::ScalarType dtype,
               const torch::Device& device) {
    // register submodules
    attn_ = register_module(
        "attn", MPTAttention(args, quant_args, parallel_args, dtype, device));
    norm_1_ = register_module("norm_1",
                              LayerNorm(args.hidden_size(),
                                        args.layer_norm_eps(),
                                        /*bias=*/false,
                                        dtype,
                                        device));
    norm_2_ = register_module("norm_2",
                              LayerNorm(args.hidden_size(),
                                        args.layer_norm_eps(),
                                        /*bias=*/false,
                                        dtype,
                                        device));
    ffn_ = register_module(
        "ffn", MPTMLP(args, quant_args, parallel_args, dtype, device));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    auto h = x + attn_(norm_1_(x), positions, kv_cache, input_params);
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
TORCH_MODULE(MPTBlock);

class MPTModelImpl : public torch::nn::Module {
 public:
  MPTModelImpl(const ModelArgs& args,
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
    blocks_ = register_module("blocks", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = MPTBlock(args, quant_args, parallel_args, dtype, device);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    norm_f_ = register_module("norm_f",
                              LayerNorm(args.hidden_size(),
                                        args.layer_norm_eps(),
                                        /*bias=*/false,
                                        dtype,
                                        device));
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
  // parameter members, must be registered
  ParallelEmbedding wte_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<MPTBlock> layers_;

  LayerNorm norm_f_{nullptr};
};
TORCH_MODULE(MPTModel);

class MPTForCausalLMImpl : public torch::nn::Module {
 public:
  MPTForCausalLMImpl(const ModelArgs& args,
                     const QuantizationArgs& quant_args,
                     const ParallelArgs& parallel_args,
                     torch::ScalarType dtype,
                     const torch::Device& device) {
    // register submodules
    transformer_ = register_module(
        "transformer",
        MPTModel(args, quant_args, parallel_args, dtype, device));

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
    h = h.index_select(/*dim=*/0, input_params.last_token_indicies);
    return lm_head_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    transformer_->load_state_dict(state_dict.select("transformer."));
    // TODO: share weights between wte and lm_head to save memory
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
TORCH_MODULE(MPTForCausalLM);

}  // namespace llm::hf
