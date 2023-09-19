#pragma once

#include <torch/torch.h>

#include "layers/attention.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/norm.h"
#include "layers/pos_embedding.h"
#include "memory/kv_cache.h"
#include "models/input_parameters.h"
#include "models/model_args.h"
#include "models/parallel_args.h"

// llama2 model based on huggingface's llama2 model weights
namespace llm::hf::llama2 {

class MLPImpl : public torch::nn::Module {
 public:
  MLPImpl(const ModelArgs& args,
          const ParallelArgs& parallel_args,
          const torch::ScalarType& dtype,
          const torch::Device& device) {
    const int64_t dim = args.dim();
    const int64_t multiple_of = args.multiple_of();
    const float ffn_dim_multiplier = args.ffn_dim_multiplier().value_or(1.0f);
    int64_t hidden_dim = 4 * dim;
    hidden_dim = 2 * hidden_dim / 3;
    // custom dim factor multiplier
    hidden_dim *= ffn_dim_multiplier;
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);

    int64_t local_hidden_dim = hidden_dim / parallel_args.world_size();
    gate_up_sizes_ = {local_hidden_dim, local_hidden_dim};

    // register the weight parameter
    gate_up_proj_ =
        register_module("gate_up_proj",
                        ColumnParallelLinear(dim,
                                             hidden_dim * 2,
                                             /*gather_output=*/false,
                                             parallel_args,
                                             dtype,
                                             device));
    down_proj_ = register_module("down_proj",
                                 RowParallelLinear(hidden_dim,
                                                   dim,
                                                   /*input_is_parallel=*/true,
                                                   parallel_args,
                                                   dtype,
                                                   device));
  }

  torch::Tensor forward(torch::Tensor x) {
    namespace F = torch::nn::functional;
    auto gate_up_proj = gate_up_proj_(x);
    auto chunks = gate_up_proj.split(/*split_size=*/gate_up_sizes_, /*dim=*/-1);
    DCHECK_EQ(chunks.size(), 2);
    return down_proj_(F::silu(chunks[0]) * chunks[1]);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    gate_up_proj_->load_state_dict(
        state_dict, {"gate_proj.", "up_proj."}, gate_up_sizes_);
    down_proj_->load_state_dict(state_dict.select("down_proj."));
  }

  bool is_loaded() const {
    return gate_up_proj_->is_loaded() && down_proj_->is_loaded();
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear gate_up_proj_{nullptr};
  RowParallelLinear down_proj_{nullptr};
  std::vector<int64_t> gate_up_sizes_;
};
TORCH_MODULE(MLP);

class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl(uint32_t layer_id,
                const ModelArgs& args,
                const ParallelArgs& parallel_args,
                const torch::ScalarType& dtype,
                const torch::Device& device)
      : layer_id_(layer_id), parallel_args_(parallel_args) {
    const auto world_size = parallel_args.world_size();
    const int64_t dim = args.dim();
    const int64_t n_heads = args.n_heads();
    const int64_t n_kv_heads = args.n_kv_heads().value_or(n_heads);

    n_local_heads_ = n_heads / world_size;
    n_local_kv_heads_ = n_kv_heads / world_size;
    head_dim_ = dim / n_heads;
    // size for q, k, v
    qkv_sizes_ = {n_local_heads_ * head_dim_,
                  n_local_kv_heads_ * head_dim_,
                  n_local_kv_heads_ * head_dim_};

    // register submodules
    qkv_proj_ = register_module(
        "qkv_proj",
        ColumnParallelLinear(dim,
                             (n_heads + 2 * n_kv_heads) * head_dim_,
                             /*gather_output=*/false,
                             parallel_args,
                             dtype,
                             device));

    o_proj_ = register_module("o_proj",
                              RowParallelLinear(n_heads * head_dim_,
                                                dim,
                                                /*input_is_parallel=*/true,
                                                parallel_args,
                                                dtype,
                                                device));

    // initialize positional embedding
    // TODO: need to adjust the max_seq_len
    const auto rotary_dim = args.dim() / args.n_heads();
    pos_emb_ = register_module(
        "pos_emb",
        RotaryEmbedding(rotary_dim,
                        args.max_seq_len(),
                        /*scaling_factor=*/args.rope_scaling(),
                        /*rope_theta=*/args.rope_theta(),
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
    auto qkv = qkv_proj_(x).split(/*split_size=*/qkv_sizes_, /*dim=*/-1);
    DCHECK_EQ(qkv.size(), 3);
    auto query = qkv[0];
    auto key = qkv[1];
    auto value = qkv[2];

    // (num_tokens, n_local_heads, head_dim)
    query = query.view({num_tokens, n_local_heads_, head_dim_});
    key = key.view({num_tokens, n_local_kv_heads_, head_dim_});
    value = value.view({num_tokens, n_local_kv_heads_, head_dim_});

    // (num_tokens, n_local_heads, head_dim)
    // apply positional embedding
    std::tie(query, key) = pos_emb_(query, key, positions);

    // store k/v into cache based on slots
    kv_cache.set_kv_cache(input_params.slot_ids, key, value);

    auto output = torch::zeros_like(query);
    const auto num_prompt_tokens = input_params.num_prompt_tokens;
    if (num_prompt_tokens > 0) {
      // process sequences with prompt tokens (prefill)
      auto sliced_output =
          output.slice(/*dim=*/0, /*start=*/0, /*end=*/num_prompt_tokens);
      auto sliced_query =
          query.slice(/*dim=*/0, /*start=*/0, /*end=*/num_prompt_tokens);
      auto sliced_key =
          key.slice(/*dim=*/0, /*start=*/0, /*end=*/num_prompt_tokens);
      auto sliced_value =
          value.slice(/*dim=*/0, /*start=*/0, /*end=*/num_prompt_tokens);
      attention::varlen_masked_self_attention(sliced_query,
                                              sliced_key,
                                              sliced_value,
                                              input_params.cu_seq_lens,
                                              input_params.max_seq_len,
                                              sliced_output);
    }

    if (num_prompt_tokens < num_tokens) {
      // process sequences without prompt tokens (decode)
      auto sliced_output = output.slice(/*dim=*/0, /*start=*/num_prompt_tokens);
      auto sliced_query = query.slice(/*dim=*/0, /*start=*/num_prompt_tokens);
      attention::single_token_masked_self_attention(
          kv_cache,
          sliced_query,
          input_params.block_tables,
          input_params.context_lens,
          input_params.max_context_len,
          sliced_output);
    }
    output = output.contiguous().view({num_tokens, -1});
    return o_proj_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    qkv_proj_->load_state_dict(
        state_dict, {"q_proj.", "k_proj.", "v_proj."}, qkv_sizes_);
    o_proj_->load_state_dict(state_dict.select("o_proj."));
  }

  bool is_loaded() const {
    return qkv_proj_->is_loaded() && o_proj_->is_loaded();
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear qkv_proj_{nullptr};

  RowParallelLinear o_proj_{nullptr};

  RotaryEmbedding pos_emb_{nullptr};

  uint32_t layer_id_;

  ParallelArgs parallel_args_;

  // configs used in forward
  int64_t n_local_heads_;
  int64_t n_local_kv_heads_;
  int64_t head_dim_;

  // size for q, k, v
  std::vector<int64_t> qkv_sizes_;
};
TORCH_MODULE(Attention);

class DecoderLayerImpl : public torch::nn::Module {
 public:
  DecoderLayerImpl(uint32_t layer_id,
                   const ModelArgs& args,
                   const ParallelArgs& parallel_args,
                   const torch::ScalarType& dtype,
                   const torch::Device& device)
      : layer_id_(layer_id), parallel_args_(parallel_args) {
    // register submodules
    self_attn_ = register_module(
        "self_attn", Attention(layer_id, args, parallel_args, dtype, device));
    mlp_ = register_module("mlp", MLP(args, parallel_args, dtype, device));
    input_layernorm_ = register_module(
        "input_layernorm", RMSNorm(args.dim(), args.norm_eps(), dtype, device));
    post_attention_layernorm_ =
        register_module("post_attention_layernorm",
                        RMSNorm(args.dim(), args.norm_eps(), dtype, device));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    auto h =
        x + self_attn_(input_layernorm_(x), positions, kv_cache, input_params);
    return h + mlp_(post_attention_layernorm_(h));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    self_attn_->load_state_dict(state_dict.select("self_attn."));
    mlp_->load_state_dict(state_dict.select("mlp."));
    input_layernorm_->load_state_dict(state_dict.select("input_layernorm."));
    post_attention_layernorm_->load_state_dict(
        state_dict.select("post_attention_layernorm."));
  }

  bool is_loaded() const {
    return self_attn_->is_loaded() && mlp_->is_loaded() &&
           input_layernorm_->is_loaded() &&
           post_attention_layernorm_->is_loaded();
  }

 private:
  // parameter members, must be registered
  Attention self_attn_{nullptr};

  MLP mlp_{nullptr};

  RMSNorm input_layernorm_{nullptr};

  RMSNorm post_attention_layernorm_{nullptr};

  uint32_t layer_id_;

  ParallelArgs parallel_args_;
};
TORCH_MODULE(DecoderLayer);

class ModelImpl : public torch::nn::Module {
 public:
  ModelImpl(const ModelArgs& args,
            const ParallelArgs& parallel_args,
            const torch::ScalarType& dtype,
            const torch::Device& device)
      : parallel_args_(parallel_args) {
    // register submodules
    embed_tokens_ = register_module(
        "embed_tokens",
        ParallelEmbedding(
            args.vocab_size(), args.dim(), parallel_args, dtype, device));
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = DecoderLayer(i, args, parallel_args, dtype, device);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    norm_ = register_module(
        "norm", RMSNorm(args.dim(), args.norm_eps(), dtype, device));
    lm_head_ = register_module("lm_head",
                               ColumnParallelLinear(args.dim(),
                                                    args.vocab_size(),
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
    auto h = embed_tokens_(tokens);
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, kv_caches[i], input_params);
    }
    h = norm_(h);
    // select last token for each sequence
    h = h.index_select(/*dim=*/0, input_params.last_token_indicies);
    return lm_head_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(state_dict.select("model.embed_tokens."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.select("model.layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.select("model.norm."));
    lm_head_->load_state_dict(state_dict.select("lm_head."));
  }

  bool is_loaded() const {
    bool all_loaded = embed_tokens_->is_loaded() && norm_->is_loaded() &&
                      lm_head_->is_loaded();
    // check if all layers are loaded
    for (const auto& layer : layers_) {
      all_loaded = all_loaded && layer->is_loaded();
    }
    return all_loaded;
  }

 private:
  // parameter members, must be registered
  ParallelEmbedding embed_tokens_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<DecoderLayer> layers_;

  RMSNorm norm_{nullptr};

  ColumnParallelLinear lm_head_{nullptr};

  ParallelArgs parallel_args_;
};
TORCH_MODULE(Model);

}  // namespace llm::hf::llama2
