#include "args_overrider.h"

#include <gflags/gflags.h>

#include "common/logging.h"

// define gflags for all model args defined in src/models/args.h
DEFINE_string(model_type, "", "model type, e.g. llama2, llama, gpt_neox");
DEFINE_string(dtype, "", "dtype, e.g. float32, float16");
DEFINE_string(hidden_size, "", "dimension of the encoder layer");
DEFINE_string(hidden_act, "", "hidden activation function");
DEFINE_string(intermediate_size,
              "",
              "dimension of the 'intermediate' (aka feed-forward) layer");
DEFINE_string(n_layers, "", "number of hidden layers in the encoder");
DEFINE_string(n_heads, "", "number of attention heads");
DEFINE_string(n_kv_heads, "", "number of attention heads for key/value");
DEFINE_string(vocab_size, "", "number of tokens in the vocabulary");
DEFINE_string(rms_norm_eps, "", "the epsilon value to use for rms norm");
DEFINE_string(layer_norm_eps, "", "the epsilon value to use for layer norm");
DEFINE_string(rotary_dim, "", "args for rotary position embeddings");
DEFINE_string(rope_theta,
              "",
              "the base period of the rotary position embeddings");
DEFINE_string(rope_scaling, "", "rope scaling factor");
DEFINE_string(
    rotary_pct,
    "",
    "percentage of hidden dimension to allocate to rotary position embeddings");
DEFINE_string(
    max_position_embeddings,
    "",
    "the maximum sequence length to use for rotary position embeddings");
DEFINE_string(bos_token_id, "", "token id for beginning of sentence");
DEFINE_string(eos_token_id, "", "token id for end of sentence");
DEFINE_string(
    use_parallel_residual,
    "",
    "whether to use a 'parallel' formulation in each transformer layer");
DEFINE_string(attn_qkv_clip, "", "clip the qkv matrix in attention layer");
DEFINE_string(attn_qk_ln,
              "",
              "whether to apply layer norm to qk in attention layer");
DEFINE_string(attn_alibi, "", "whether to use alibi attention");
DEFINE_string(alibi_bias_max, "", "max value of bias in alibi attention");
DEFINE_string(no_bias, "", "whether to use bias in attention layer");
DEFINE_string(residual_post_layernorm,
              "",
              "whether to apply residual after layernorm");

// define gflags for all quant args defined in src/models/args.h
DEFINE_string(quant_method, "", "quantization method, e.g. awq, gptq");
DEFINE_string(bits, "", "number of bits for quantization");
DEFINE_string(group_size, "", "group size for quantization");
DEFINE_string(desc_act, "", "desc_act for quantization");
DEFINE_string(true_sequential, "", "true_sequential for quantization");

// In gflag, we can't tell the difference between a flag not set and a flag set
// to default value. Instead, we use string to represent the flag value, and use
// empty string to represent the flag not set. this leads to a lot of
// boilerplate code to convert string to the actual type. we use the following
// macros to reduce the boilerplate.
template <typename T>
std::optional<T> convert_from_string(const std::string& str) {
  try {
    if constexpr (std::is_same<T, std::string>::value) {
      return str;
    } else if constexpr (std::is_same<T, bool>::value) {
      // return true if match "1", "t", "true", "y", "yes"
      return str == "1" || str == "true" || str == "t" || str == "y" ||
             str == "yes";
    } else if constexpr (std::is_same<T, float>::value) {
      return std::stof(str);
    } else if constexpr (std::is_same<T, double>::value) {
      return std::stod(str);
    } else if constexpr (std::is_same<T, int32_t>::value) {
      return std::stoi(str);
    } else if constexpr (std::is_same<T, int64_t>::value) {
      return std::stoll(str);
    } else if constexpr (std::is_same<T, uint32_t>::value) {
      return std::stoul(str);
    } else if constexpr (std::is_same<T, uint64_t>::value) {
      return std::stoull(str);
    } else {
      static_assert(std::is_same<T, void>::value,
                    "unsupported type for convert_from_string");
    }
  } catch (const std::invalid_argument& e) {
  }
  return std::nullopt;
}

#define OVERRIDE_ARG_FROM_GFLAG(args, arg_name)                            \
  if (!FLAGS_##arg_name.empty()) {                                         \
    auto value = args.arg_name();                                          \
    auto arg_val = convert_from_string<decltype(value)>(FLAGS_##arg_name); \
    if (arg_val.has_value()) {                                             \
      args.arg_name() = arg_val.value();                                   \
      GLOG(WARNING) << "Overwriting " << #arg_name << " from " << value    \
                    << " to " << arg_val.value();                          \
    } else {                                                               \
      GLOG(WARNING) << "Ignoring invalid value for " << #arg_name << ": "  \
                    << FLAGS_##arg_name;                                   \
    }                                                                      \
  }

#define OVERRIDE_OPTIONAL_ARG_FROM_GFLAG(args, arg_name)                    \
  if (!FLAGS_##arg_name.empty()) {                                          \
    auto value = args.arg_name();                                           \
    auto arg_val =                                                          \
        convert_from_string<decltype(value)::value_type>(FLAGS_##arg_name); \
    if (arg_val.has_value()) {                                              \
      args.arg_name() = arg_val.value();                                    \
      if (value.has_value()) {                                              \
        GLOG(WARNING) << "Overwriting " << #arg_name << " from "            \
                      << value.value() << " to " << arg_val.value();        \
      } else {                                                              \
        GLOG(WARNING) << "Overwriting " << #arg_name << " from nullopt"     \
                      << " to " << arg_val.value();                         \
      }                                                                     \
    } else {                                                                \
      GLOG(WARNING) << "Ignoring invalid value for " << #arg_name << ": "   \
                    << FLAGS_##arg_name;                                    \
    }                                                                       \
  }

namespace llm {

// a utility function to override model args from gflag
void override_args_from_gflag(ModelArgs& args, QuantArgs& quant_args) {
  // override args from gflag
  OVERRIDE_ARG_FROM_GFLAG(args, model_type);
  OVERRIDE_ARG_FROM_GFLAG(args, dtype);
  OVERRIDE_ARG_FROM_GFLAG(args, hidden_size);
  OVERRIDE_ARG_FROM_GFLAG(args, hidden_act);
  OVERRIDE_ARG_FROM_GFLAG(args, intermediate_size);
  OVERRIDE_ARG_FROM_GFLAG(args, n_layers);
  OVERRIDE_ARG_FROM_GFLAG(args, n_heads);
  OVERRIDE_OPTIONAL_ARG_FROM_GFLAG(args, n_kv_heads);
  OVERRIDE_ARG_FROM_GFLAG(args, vocab_size);
  OVERRIDE_ARG_FROM_GFLAG(args, rms_norm_eps);
  OVERRIDE_ARG_FROM_GFLAG(args, layer_norm_eps);
  OVERRIDE_ARG_FROM_GFLAG(args, rotary_dim);
  OVERRIDE_ARG_FROM_GFLAG(args, rope_theta);
  OVERRIDE_ARG_FROM_GFLAG(args, rope_scaling);
  OVERRIDE_ARG_FROM_GFLAG(args, rotary_pct);
  OVERRIDE_ARG_FROM_GFLAG(args, max_position_embeddings);
  OVERRIDE_ARG_FROM_GFLAG(args, bos_token_id);
  OVERRIDE_ARG_FROM_GFLAG(args, eos_token_id);
  OVERRIDE_ARG_FROM_GFLAG(args, use_parallel_residual);
  OVERRIDE_OPTIONAL_ARG_FROM_GFLAG(args, attn_qkv_clip);
  OVERRIDE_ARG_FROM_GFLAG(args, attn_qk_ln);
  OVERRIDE_ARG_FROM_GFLAG(args, attn_alibi);
  OVERRIDE_ARG_FROM_GFLAG(args, alibi_bias_max);
  OVERRIDE_ARG_FROM_GFLAG(args, no_bias);
  OVERRIDE_ARG_FROM_GFLAG(args, residual_post_layernorm);

  // override quant args from gflag
  OVERRIDE_ARG_FROM_GFLAG(quant_args, quant_method);
  OVERRIDE_ARG_FROM_GFLAG(quant_args, bits);
  OVERRIDE_ARG_FROM_GFLAG(quant_args, group_size);
  OVERRIDE_ARG_FROM_GFLAG(quant_args, desc_act);
  OVERRIDE_ARG_FROM_GFLAG(quant_args, true_sequential);
}

}  // namespace llm