#pragma once

#include <torch/torch.h>

#include "layers/attention.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/normalization.h"
#include "layers/pos_embedding.h"
#include "memory/kv_cache.h"
#include "models/args.h"
#include "models/input_parameters.h"

// port LLAMA's model to C++ API:
// https://github.com/facebookresearch/llama/blob/main/llama/model.py
namespace llm::llama2 {
class FeedForwardImpl : public torch::nn::Module {
 public:
  FeedForwardImpl(const ModelArgs& args,
                  const QuantizationArgs& quant_args,
                  const ParallelArgs& parallel_args,
                  const torch::ScalarType& dtype,
                  const torch::Device& device) {
    const int64_t hidden_size = args.hidden_size();
    const int64_t intermediate_size = args.intermediate_size();

    const int64_t local_intermediate_size = intermediate_size / parallel_args.world_size();
    w1_w3_sizes_ = {local_intermediate_size, local_intermediate_size};

    // register the weight parameter
    w1_w3_ = register_module("w1_w3",
                             ColumnParallelLinear(hidden_size,
                                                  intermediate_size * 2,
                                                  /*bias=*/false,
                                                  /*gather_output=*/false,
                                                  quant_args,
                                                  parallel_args,
                                                  dtype,
                                                  device));
    w2_ = register_module("w2",
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
    namespace F = torch::nn::functional;
    auto w1_w3 = w1_w3_(x);
    auto chunks = w1_w3.chunk(/*chunks=*/2, /*dim=*/-1);
    return w2_(F::silu(chunks[0]) * chunks[1]);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    w1_w3_->load_state_dict(state_dict, {"w1.", "w3."});
    w2_->load_state_dict(state_dict.select("w2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    w1_w3_->verify_loaded_weights(prefix + "[w1,w3].");
    w2_->verify_loaded_weights(prefix + "w2.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear w1_w3_{nullptr};
  RowParallelLinear w2_{nullptr};
  std::vector<int64_t> w1_w3_sizes_;
};
TORCH_MODULE(FeedForward);

class LlamaAttentionImpl : public torch::nn::Module {
 public:
  LlamaAttentionImpl(uint32_t layer_id,
                     const ModelArgs& args,
                     const QuantizationArgs& quant_args,
                     const ParallelArgs& parallel_args,
                     const torch::ScalarType& dtype,
                     const torch::Device& device)
      : layer_id_(layer_id), parallel_args_(parallel_args) {
    const auto world_size = parallel_args.world_size();
    const int64_t hidden_size = args.hidden_size();
    const int64_t n_heads = args.n_heads();
    const int64_t n_kv_heads = args.n_kv_heads().value_or(n_heads);

    n_local_heads_ = n_heads / world_size;
    n_local_kv_heads_ = n_kv_heads / world_size;
    head_dim_ = hidden_size / n_heads;
    // size for q, k, v
    qkv_sizes_ = {n_local_heads_ * head_dim_,
                  n_local_kv_heads_ * head_dim_,
                  n_local_kv_heads_ * head_dim_};

    // register submodules
    wqkv_ = register_module(
        "wqkv",
        ColumnParallelLinear(hidden_size,
                             (n_heads + 2 * n_kv_heads) * head_dim_,
                             /*bias=*/false,
                             /*gather_output=*/false,
                             quant_args,
                             parallel_args,
                             dtype,
                             device));
    wo_ = register_module("wo",
                          RowParallelLinear(n_heads * head_dim_,
                                            hidden_size,
                                            /*bias=*/false,
                                            /*input_is_parallel=*/true,
                                            quant_args,
                                            parallel_args,
                                            dtype,
                                            device));

    // initialize positional embedding
    // TODO: need to adjust the max_seq_len
    const auto rotary_dim = args.hidden_size() / args.n_heads();
    pos_emb_ =
        register_module("pos_emb",
                        RotaryEmbedding(rotary_dim,
                                        args.max_seq_len(),
                                        /*scaling_factor=*/args.rope_scaling(),
                                        /*rope_theta=*/args.rope_theta(),
                                        /*interleaved=*/true,
                                        dtype,
                                        device));

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    atten_ = register_module(
        "atten", Attention(n_heads, n_kv_heads, scale, dtype, device));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    const auto num_tokens = x.size(0);
    // (num_tokens, dim) x (dim, n_heads * head_dim)
    // => (num_tokens, n_heads * head_dim)
    auto qkv = wqkv_(x).split(/*split_size=*/qkv_sizes_, /*dim=*/-1);
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

    // calculate attention, output: (num_tokens, n_local_heads, head_dim)
    auto output = atten_(query, key, value, kv_cache, input_params);
    // reshape to (num_tokens, n_local_heads * head_dim)
    output = output.view({num_tokens, -1});
    return wo_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    wqkv_->load_state_dict(state_dict, {"wq.", "wk.", "wv."});
    wo_->load_state_dict(state_dict.select("wo."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    wqkv_->verify_loaded_weights(prefix + "[wq,wk,wv].");
    wo_->verify_loaded_weights(prefix + "wo.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear wqkv_{nullptr};

  RowParallelLinear wo_{nullptr};

  // module members without parameters
  RotaryEmbedding pos_emb_{nullptr};

  Attention atten_{nullptr};

  uint32_t layer_id_;

  ParallelArgs parallel_args_;

  // configs used in forward
  int64_t n_local_heads_;
  int64_t n_local_kv_heads_;
  int64_t head_dim_;

  // size for q, k, v
  std::vector<int64_t> qkv_sizes_;
};
TORCH_MODULE(LlamaAttention);

class TransformerBlockImpl : public torch::nn::Module {
 public:
  TransformerBlockImpl(uint32_t layer_id,
                       const ModelArgs& args,
                       const QuantizationArgs& quant_args,
                       const ParallelArgs& parallel_args,
                       const torch::ScalarType& dtype,
                       const torch::Device& device)
      : layer_id_(layer_id), parallel_args_(parallel_args) {
    // register submodules
    attention_ = register_module(
        "attention",
        LlamaAttention(
            layer_id, args, quant_args, parallel_args, dtype, device));
    feed_forward_ = register_module(
        "feed_forward",
        FeedForward(args, quant_args, parallel_args, dtype, device));
    attention_norm_ = register_module(
        "attention_norm",
        RMSNorm(args.hidden_size(), args.norm_eps(), dtype, device));
    ffn_norm_ = register_module(
        "ffn_norm",
        RMSNorm(args.hidden_size(), args.norm_eps(), dtype, device));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    auto h =
        x + attention_(attention_norm_(x), positions, kv_cache, input_params);
    return h + feed_forward_(ffn_norm_(h));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    attention_->load_state_dict(state_dict.select("attention."));
    feed_forward_->load_state_dict(state_dict.select("feed_forward."));
    attention_norm_->load_state_dict(state_dict.select("attention_norm."));
    ffn_norm_->load_state_dict(state_dict.select("ffn_norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    attention_->verify_loaded_weights(prefix + "attention.");
    feed_forward_->verify_loaded_weights(prefix + "feed_forward.");
    attention_norm_->verify_loaded_weights(prefix + "attention_norm.");
    ffn_norm_->verify_loaded_weights(prefix + "ffn_norm.");
  }

 private:
  // parameter members, must be registered
  LlamaAttention attention_{nullptr};

  FeedForward feed_forward_{nullptr};

  RMSNorm attention_norm_{nullptr};

  RMSNorm ffn_norm_{nullptr};

  uint32_t layer_id_;

  ParallelArgs parallel_args_;
};
TORCH_MODULE(TransformerBlock);

class ModelImpl : public torch::nn::Module {
 public:
  ModelImpl(const ModelArgs& args,
            const QuantizationArgs& quant_args,
            const ParallelArgs& parallel_args,
            const torch::ScalarType& dtype,
            const torch::Device& device)
      : parallel_args_(parallel_args) {
    // register submodules
    tok_embeddings_ = register_module("tok_embeddings",
                                      ParallelEmbedding(args.vocab_size(),
                                                        args.hidden_size(),
                                                        parallel_args,
                                                        dtype,
                                                        device));
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block =
          TransformerBlock(i, args, quant_args, parallel_args, dtype, device);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    norm_ = register_module(
        "norm", RMSNorm(args.hidden_size(), args.norm_eps(), dtype, device));
    output_ = register_module("output",
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
    auto h = tok_embeddings_(tokens);
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, kv_caches[i], input_params);
    }
    h = norm_(h);
    // select last token for each sequence
    h = h.index_select(/*dim=*/0, input_params.last_token_indicies);
    return output_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    tok_embeddings_->load_state_dict(state_dict.select("tok_embeddings."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.select("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.select("norm."));
    output_->load_state_dict(state_dict.select("output."));
  }

  void verify_loaded_weights() const {
    tok_embeddings_->verify_loaded_weights("tok_embeddings.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights("layers." + std::to_string(i) + ".");
    }
    norm_->verify_loaded_weights("norm.");
    output_->verify_loaded_weights("output.");
  }

 private:
  // parameter members, must be registered
  ParallelEmbedding tok_embeddings_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<TransformerBlock> layers_;

  RMSNorm norm_{nullptr};

  ColumnParallelLinear output_{nullptr};

  ParallelArgs parallel_args_;
};
TORCH_MODULE(Model);

}  // namespace llm::llama2
