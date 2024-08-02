#pragma once

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_set>

#include "common/macros.h"

namespace llm {

struct ModelArgs {
  DEFINE_ARG(std::string, model_type);

  DEFINE_ARG(std::string, dtype);

  // dimension of the encoder layer.
  DEFINE_ARG(int64_t, hidden_size) = 0;

  DEFINE_ARG(std::string, hidden_act);

  // dimension of the 'intermediate' (aka feed-forward) layer.
  DEFINE_ARG(int64_t, intermediate_size) = 0;

  // number of hidden layers in the encoder.
  DEFINE_ARG(int64_t, n_layers) = 0;

  // dimension of the attention head.
  DEFINE_ARG(int64_t, head_dim) = 0;

  // number of attention heads.
  DEFINE_ARG(int64_t, n_heads) = 0;

  // number of attention heads for key/value.
  DEFINE_ARG(std::optional<int64_t>, n_kv_heads);

  // number of tokens in the vocabulary.
  DEFINE_ARG(int64_t, vocab_size) = -1;

  // whether to use rms norm.
  DEFINE_ARG(bool, use_rms_norm) = false;

  // the epsilon value to use for rms norm.
  DEFINE_ARG(float, rms_norm_eps) = 0.0f;

  // the epsilon value to use for layer norm.
  DEFINE_ARG(float, layer_norm_eps) = 0.0f;

  // args for rotary position embeddings
  DEFINE_ARG(int64_t, rotary_dim) = 0;

  // the base period of the rotary position embeddings.
  DEFINE_ARG(float, rope_theta) = 10000.0f;

  // rope_scaling related args
  DEFINE_ARG(std::string, rope_scaling_rope_type);
  DEFINE_ARG(float, rope_scaling_factor) = 0.0f;
  DEFINE_ARG(float, rope_scaling_low_freq_factor) = 0.0f;
  DEFINE_ARG(float, rope_scaling_high_freq_factor) = 0.0f;
  DEFINE_ARG(int64_t, rope_scaling_original_max_position_embeddings) = 0;

  // percentage of hidden dimension to allocate to rotary position embeddings.
  DEFINE_ARG(float, rotary_pct) = 1.0f;

  // the maximum sequence length to use for rotary position embeddings.
  DEFINE_ARG(int64_t, max_position_embeddings) = 0;

  // token id for beginning of sentence.
  DEFINE_ARG(int32_t, bos_token_id) = 0;

  // token id for end of sentence.
  DEFINE_ARG(int32_t, eos_token_id) = 0;

  // configs for gpt_neox
  // whether to use a 'parallel' formulation in each transformer layer, which
  // can provide a slight training speedup at large scales (e.g. 20B).
  DEFINE_ARG(bool, use_parallel_residual) = false;

  // configs for MPT attention
  // the clip value for qkv which limits the range of qkv values within
  // [-qkv_clip, qkv_clip]
  DEFINE_ARG(std::optional<float>, attn_qkv_clip);

  // whether to use layer norm for qk
  DEFINE_ARG(bool, attn_qk_ln) = false;

  // whether to use alibi
  DEFINE_ARG(bool, attn_alibi) = false;

  // max alibi bias
  DEFINE_ARG(float, alibi_bias_max) = 0.0f;

  // whether to use bias. only used for mpt models
  DEFINE_ARG(bool, no_bias) = false;

  // whether to use bias for linear.
  DEFINE_ARG(bool, linear_bias) = false;

  // whether to use bias for qkv.
  DEFINE_ARG(bool, qkv_bias) = false;

  // whether to apply residual connection post layernorm
  DEFINE_ARG(bool, residual_post_layernorm) = false;

  // whether to use layer norm on final layer.
  DEFINE_ARG(bool, post_layernorm) = false;

  // Stop token ids for decoding.
  DEFINE_ARG(std::unordered_set<int32_t>, stop_token_ids);

  // sliding window for attention
  DEFINE_ARG(bool, use_sliding_window) = false;
  DEFINE_ARG(int32_t, sliding_window) = -1;
  DEFINE_ARG(int32_t, max_window_layers) = 0;
};

inline std::ostream& operator<<(std::ostream& os, const ModelArgs& args) {
  os << "ModelArgs: [model_type: " << args.model_type();
  os << ", dtype: " << args.dtype();
  os << ", hidden_size: " << args.hidden_size();
  os << ", hidden_act: " << args.hidden_act();
  os << ", intermediate_size: " << args.intermediate_size();
  os << ", n_layers: " << args.n_layers();
  os << ", head_dim: " << args.head_dim();
  os << ", n_heads: " << args.n_heads();
  os << ", n_kv_heads: " << args.n_kv_heads().value_or(-1);
  os << ", vocab_size: " << args.vocab_size();
  os << ", rms_norm_eps: " << args.rms_norm_eps();
  os << ", layer_norm_eps: " << args.layer_norm_eps();
  os << ", rotary_dim: " << args.rotary_dim();
  os << ", rope_theta: " << args.rope_theta();
  os << ", rope_scaling_rope_type: " << args.rope_scaling_rope_type();
  os << ", rope_scaling_factor: " << args.rope_scaling_factor();
  os << ", rope_scaling_low_freq_factor: "
     << args.rope_scaling_low_freq_factor();
  os << ", rope_scaling_high_freq_factor: "
     << args.rope_scaling_high_freq_factor();
  os << ", rope_scaling_original_max_position_embeddings: "
     << args.rope_scaling_original_max_position_embeddings();
  os << ", rotary_pct: " << args.rotary_pct();
  os << ", max_position_embeddings: " << args.max_position_embeddings();
  os << ", bos_token_id: " << args.bos_token_id();
  os << ", eos_token_id: " << args.eos_token_id();
  os << ", use_parallel_residual: " << args.use_parallel_residual();
  os << ", attn_qkv_clip: " << args.attn_qkv_clip().value_or(0.0f);
  os << ", attn_qk_ln: " << args.attn_qk_ln();
  os << ", attn_alibi: " << args.attn_alibi();
  os << ", alibi_bias_max: " << args.alibi_bias_max();
  os << ", no_bias: " << args.no_bias();
  os << ", linear_bias: " << args.linear_bias();
  os << ", qkv_bias: " << args.qkv_bias();
  os << ", residual_post_layernorm: " << args.residual_post_layernorm();
  os << "]";
  return os;
}

}  // namespace llm
