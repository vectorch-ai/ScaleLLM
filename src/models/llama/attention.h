#pragma once

#include <torch/torch.h>

#include "layers/attention.h"
#include "layers/linear.h"
#include "layers/pos_embedding.h"
#include "memory/kv_cache.h"
#include "models/model_args.h"
#include "models/parameters.h"

namespace llm {

class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl(const ModelArgs& args,
                const ParallelArgs& parallel_args,
                const torch::ScalarType& dtype,
                const torch::Device& device) {
    const auto world_size = parallel_args.world_size();
    const int64_t dim = args.dim();
    const int64_t n_heads = args.n_heads();
    const int64_t n_kv_heads = args.n_kv_heads().value_or(args.n_heads());

    n_local_heads_ = n_heads / world_size;
    n_local_kv_heads_ = n_kv_heads / world_size;
    head_dim_ = dim / n_heads;

    // register submodules
    // TODO: fuse wq, wk, wv into one linear layer
    wq_ = register_module("wq",
                          ColumnParallelLinear(dim,
                                               n_heads * head_dim_,
                                               /*gather_output=*/false,
                                               parallel_args,
                                               dtype,
                                               device));
    wk_ = register_module("wk",
                          ColumnParallelLinear(dim,
                                               n_kv_heads * head_dim_,
                                               /*gather_output=*/false,
                                               parallel_args,
                                               dtype,
                                               device));
    wv_ = register_module("wv",
                          ColumnParallelLinear(dim,
                                               n_kv_heads * head_dim_,
                                               /*gather_output=*/false,
                                               parallel_args,
                                               dtype,
                                               device));
    wo_ = register_module("wo",
                          RowParallelLinear(n_heads * head_dim_,
                                            dim,
                                            /*input_is_parallel=*/true,
                                            parallel_args,
                                            dtype,
                                            device));

    // initialize positional embedding
    // TODO: need to adjust the max_seq_len
    const auto rotary_dim = args.dim() / args.n_heads();
    pos_emb_ = register_module("pos_emb",
                               RotaryEmbedding(rotary_dim,
                                               args.max_seq_len(),
                                               /*scaling_factor=*/0.0f,
                                               /*interleaved=*/true,
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
    auto query = wq_(x);
    auto key = wk_(x);
    auto value = wv_(x);

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
    return wo_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    wq_->load_state_dict(state_dict.select("wq."));
    wk_->load_state_dict(state_dict.select("wk."));
    wv_->load_state_dict(state_dict.select("wv."));
    wo_->load_state_dict(state_dict.select("wo."));
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear wq_{nullptr};

  ColumnParallelLinear wk_{nullptr};

  ColumnParallelLinear wv_{nullptr};

  RowParallelLinear wo_{nullptr};

  RotaryEmbedding pos_emb_{nullptr};

  // configs used in forward
  int64_t n_local_heads_;
  int64_t n_local_kv_heads_;
  int64_t head_dim_;
};

TORCH_MODULE(Attention);

}  // namespace llm
