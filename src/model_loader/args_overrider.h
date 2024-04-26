#pragma once

#include <gflags/gflags.h>

#include "models/model_args.h"
#include "quantization/quant_args.h"
#include "tokenizer/tokenizer_args.h"

// Model args flags
DECLARE_string(model_type);
DECLARE_string(dtype);
DECLARE_string(hidden_size);
DECLARE_string(hidden_act);
DECLARE_string(intermediate_size);
DECLARE_string(n_layers);
DECLARE_string(n_heads);
DECLARE_string(n_kv_heads);
DECLARE_string(vocab_size);
DECLARE_string(rms_norm_eps);
DECLARE_string(layer_norm_eps);
DECLARE_string(rotary_dim);
DECLARE_string(rope_theta);
DECLARE_string(rope_scaling);
DECLARE_string(rotary_pct);
DECLARE_string(max_position_embeddings);
DECLARE_string(bos_token_id);
DECLARE_string(eos_token_id);
DECLARE_string(use_parallel_residual);
DECLARE_string(attn_qkv_clip);
DECLARE_string(attn_qk_ln);
DECLARE_string(attn_alibi);
DECLARE_string(alibi_bias_max);
DECLARE_string(no_bias);
DECLARE_string(residual_post_layernorm);

// Quantization flags
DECLARE_string(quant_method);
DECLARE_string(bits);
DECLARE_string(group_size);
DECLARE_string(desc_act);
DECLARE_string(true_sequential);

// tokenizer flags
DECLARE_string(tokenizer_type);
DECLARE_string(vocab_file);
// DECLARE_string(special_tokens);
DECLARE_string(pattern);
DECLARE_string(prefix_tokens);
DECLARE_string(chat_template);

namespace llm {

// a utility function to override model args from gflag
void override_args_from_gflag(ModelArgs& args,
                              QuantArgs& quant_args,
                              TokenizerArgs& tokenizer_args);

}  // namespace llm