#pragma once

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

  // rope scaling factor.
  DEFINE_ARG(float, rope_scaling) = 0.0f;

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

  // Vision model's dropout
  DEFINE_ARG(float, mm_dropout) = 0.0f;

  // Vision model's hidden_act
  DEFINE_ARG(std::string, mm_hidden_act);

  // Vision model's mm_hidden_size
  DEFINE_ARG(int64_t, mm_hidden_size) = 0;

  // Vision model's mm_image_size
  DEFINE_ARG(int64_t, mm_image_size) = 0;

  // Vision model's mm_intermediate_size
  DEFINE_ARG(int64_t, mm_intermediate_size) = 0;

  // Vision model's mm_num_channels
  DEFINE_ARG(int64_t, mm_num_channels) = 0;

  // Vision model's mm_initializer_range
  DEFINE_ARG(float, mm_initializer_range) = 0.0f;

  // Vision model's mm_layer_norm_eps
  DEFINE_ARG(float, mm_layer_norm_eps) = 0;

  // Vision model's mm_num_attention_heads
  DEFINE_ARG(int64_t, mm_num_attention_heads) = 0;

  // Vision model's mm_num_beam_groups
  DEFINE_ARG(int64_t, mm_num_beam_groups) = 0;

  // Vision model's mm_num_beams
  DEFINE_ARG(int64_t, mm_num_beams) = 0;

  // Vision model's mm_num_hidden_layers
  DEFINE_ARG(int64_t, mm_num_hidden_layers) = 0;

  // Vision model's mm_num_return_sequences
  DEFINE_ARG(int64_t, mm_num_return_sequences) = 0;

  // Vision model's mm_output_attentions
  DEFINE_ARG(bool, mm_output_attentions) = false;

  // Vision model's mm_output_hidden_states
  DEFINE_ARG(bool, mm_output_hidden_states) = false;

  // Vision model's mm_output_scores
  DEFINE_ARG(bool, mm_output_scores) = false;

  // Vision model's mm_patch_size
  DEFINE_ARG(int64_t, mm_patch_size) = 0;

  // Vision model's mm_projection_dim
  DEFINE_ARG(int64_t, mm_projection_dim) = 0;

  // Vision model's mm_remove_invalid_values
  DEFINE_ARG(bool, mm_remove_invalid_values) = false;

  // Vision model's mm_repetition_penalty
  DEFINE_ARG(float, mm_repetition_penalty) = 0.0f;

  // Vision model's mm_return_dict
  DEFINE_ARG(bool, mm_return_dict) = false;

  // Vision model's mm_return_dict_in_generate
  DEFINE_ARG(bool, mm_return_dict_in_generate) = false;

  // Vision model's mm_temperature
  DEFINE_ARG(float, mm_temperature) = 0.0f;

  // Vision model's mm_tie_encoder_decoder
  DEFINE_ARG(bool, mm_tie_encoder_decoder) = false;

  // Vision model's mm_tie_word_embeddings
  DEFINE_ARG(bool, mm_tie_word_embeddings) = false;

  // Vision model's mm_top_k
  DEFINE_ARG(int64_t, mm_top_k) = 0;

  // Vision model's mm_top_p
  DEFINE_ARG(float, mm_top_p) = 0.0f;

  // Vision model's mm_torchscript
  DEFINE_ARG(bool, mm_torchscript) = false;

  // Vision model's mm_use_bfloat16
  DEFINE_ARG(bool, mm_use_bfloat16) = false;

  // Vision model's mm_head_dim
  DEFINE_ARG(int64_t, mm_head_dim) = 0;

  // Vision model's mm_vocab_size
  DEFINE_ARG(int64_t, mm_vocab_size) = 0;

  // VLM model projector's mm_projector_type
  DEFINE_ARG(std::string, mm_projector_type);

  // VLM model projector's mm_projector_hidden_act
  DEFINE_ARG(std::string, mm_projector_hidden_act);

  // VLM model projector's mm_projector_n_layers
  DEFINE_ARG(int64_t, mm_projector_n_layers) = 0;

  // VLM model projector's mm_vision_feature_layer
  DEFINE_ARG(int64_t, mm_vision_feature_layer) = 0;

  // VLM model projector's mm_vision_feature_select_strategy
  DEFINE_ARG(std::string, mm_vision_feature_select_strategy);
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
  os << ", rope_scaling: " << args.rope_scaling();
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
