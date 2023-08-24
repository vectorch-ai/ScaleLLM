#pragma once

#include <torch/torch.h>

#include "layers/attention.h"
#include "layers/linear.h"
#include "layers/pos_embedding.h"
#include "memory/kv_cache.h"
#include "models/input_parameters.h"
#include "models/model_args.h"

namespace llm {

class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl(const ModelArgs& args,
                int64_t world_size,
                const torch::Device& device)
      : world_size_(world_size) {
    if (args.n_kv_heads().has_value()) {
      n_kv_heads_ = args.n_kv_heads().value();
    } else {
      n_kv_heads_ = args.n_heads();
    }
    n_local_heads_ = args.n_heads() / world_size_;
    n_local_kv_heads_ = n_kv_heads_ / world_size_;
    n_rep_ = n_local_heads_ / n_local_kv_heads_;
    head_dim_ = args.dim() / args.n_heads();

    const int64_t dim = args.dim();
    const int64_t n_heads = args.n_heads();

    // register submodules
    wq_ = register_module(
        "wq",
        ColumnParallelLinear(dim, n_heads * head_dim_, world_size, device));
    wk_ = register_module(
        "wk",
        ColumnParallelLinear(dim, n_kv_heads_ * head_dim_, world_size, device));
    wv_ = register_module(
        "wv",
        ColumnParallelLinear(dim, n_kv_heads_ * head_dim_, world_size, device));
    wo_ = register_module(
        "wo", RowParallelLinear(n_heads * head_dim_, dim, world_size, device));

    // initialize positional embedding
    pos_emb_ = register_module("pos_emb",
                               RotaryEmbedding(args.dim() / args.n_heads(),
                                               args.max_seq_len(),
                                               /*interleaved=*/true,
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
    // process sequences with prompt tokens (prefill)
    if (num_prompt_tokens > 0) {
      auto sliced_output =
          output.slice(/*dim=*/0, /*start=*/0, /*end=*/num_prompt_tokens);
      attention::varlen_masked_self_attention(
          query, key, value, input_params.cu_seq_lens, sliced_output);
    }

    if (num_prompt_tokens < num_tokens) {
      // process sequences without prompt tokens (generate)
      auto sliced_output = output.slice(/*dim=*/0, /*start=*/num_prompt_tokens);
      attention::single_token_masked_self_attention(
          kv_cache,
          query.slice(/*dim=*/0, /*start=*/num_prompt_tokens),
          input_params.block_tables,
          input_params.context_lens,
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

  // configs
  int64_t world_size_;
  int64_t n_kv_heads_;
  int64_t n_local_heads_;
  int64_t n_local_kv_heads_;
  int64_t n_rep_;
  int64_t head_dim_;
};

TORCH_MODULE(Attention);

}  // namespace llm
