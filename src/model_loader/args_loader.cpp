#include "args_loader.h"

#include <glog/logging.h>

#include <nlohmann/json.hpp>

#include "models/args.h"

namespace llm {

bool load_meta_llama_model_args(const nlohmann::json& data, ModelArgs* args) {
  // example config:
  // https://huggingface.co/meta-llama/Llama-2-7b/blob/main/params.json
  args->model_type() = "llama2";
  args->vocab_size() = 32000;
  args->hidden_size() = 4096;
  args->n_layers() = 32;
  args->n_heads() = 32;
  args->intermediate_size() = 11008;
  args->hidden_act() = "silu";
  args->max_position_embeddings() = 4096;
  args->rope_theta() = 10000.0f;
  args->rms_norm_eps() = 1e-5;
  args->bos_token_id() = 1;
  args->eos_token_id() = 2;

  if (data.contains("dim")) {
    args->hidden_size() = data["dim"].get<int64_t>();
  }
  if (data.contains("n_layers")) {
    args->n_layers() = data["n_layers"].get<int64_t>();
  }
  if (data.contains("n_heads")) {
    args->n_heads() = data["n_heads"].get<int64_t>();
  }
  if (data.contains("n_kv_heads")) {
    args->n_kv_heads() = data["n_kv_heads"].get<int64_t>();
  }
  if (data.contains("vocab_size")) {
    args->vocab_size() = data["vocab_size"].get<int64_t>();
  }
  if (data.contains("norm_eps")) {
    args->rms_norm_eps() = data["norm_eps"].get<float>();
  }
  if (data.contains("rope_theta")) {
    args->rope_theta() = data["rope_theta"].get<float>();
  }

  // calculate intermediate_size from hidden_size
  int64_t multiple_of = 256;
  float ffn_dim_multiplier = 1.0f;
  if (data.contains("multiple_of")) {
    multiple_of = data["multiple_of"].get<int64_t>();
  }
  if (data.contains("ffn_dim_multiplier")) {
    ffn_dim_multiplier = data["ffn_dim_multiplier"].get<float>();
  }

  // calculate hidden_dim from dim
  int64_t intermediate_size = args->hidden_size() * 4;
  intermediate_size = 2 * intermediate_size / 3;
  // custom dim factor multiplier
  intermediate_size *= ffn_dim_multiplier;
  // round up to make hidden layer size multiple of large power of 2
  intermediate_size =
      multiple_of * ((intermediate_size + multiple_of - 1) / multiple_of);
  args->intermediate_size() = intermediate_size;
  return true;
}

bool load_llama_model_args(const nlohmann::json& data, ModelArgs* args) {
  // example config:
  // https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json set
  // default values for args explicitly with values from:
  // https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py#L112
  args->vocab_size() = 32000;
  args->hidden_size() = 4096;
  args->n_layers() = 32;
  args->n_heads() = 32;
  args->intermediate_size() = 11008;
  args->hidden_act() = "silu";
  args->max_position_embeddings() = 2048;
  args->rms_norm_eps() = 1e-5;
  args->bos_token_id() = 1;
  args->eos_token_id() = 2;
  args->rope_theta() = 10000.0f;

  if (data.contains("vocab_size")) {
    args->vocab_size() = data["vocab_size"].get<int64_t>();
  }
  if (data.contains("hidden_size")) {
    args->hidden_size() = data["hidden_size"].get<int64_t>();
  }
  if (data.contains("num_hidden_layers")) {
    args->n_layers() = data["num_hidden_layers"].get<int64_t>();
  }
  if (data.contains("num_attention_heads")) {
    args->n_heads() = data["num_attention_heads"].get<int64_t>();
  }
  if (data.contains("num_key_value_heads")) {
    args->n_kv_heads() = data["num_key_value_heads"].get<int64_t>();
  }
  if (data.contains("intermediate_size")) {
    args->intermediate_size() = data["intermediate_size"].get<int64_t>();
  } else {
    LOG(ERROR) << "Failed to find intermediate_size in config.json";
    return false;
  }
  if (data.contains("max_position_embeddings")) {
    args->max_position_embeddings() =
        data["max_position_embeddings"].get<int64_t>();
  }
  if (data.contains("rms_norm_eps")) {
    args->rms_norm_eps() = data["rms_norm_eps"].get<float>();
  }
  if (data.contains("bos_token_id")) {
    args->bos_token_id() = data["bos_token_id"].get<int32_t>();
  }
  if (data.contains("eos_token_id")) {
    args->eos_token_id() = data["eos_token_id"].get<int32_t>();
  }
  if (data.contains("hidden_act")) {
    args->hidden_act() = data["hidden_act"].get<std::string>();
  }
  if (data.contains("rope_theta")) {
    args->rope_theta() = data["rope_theta"].get<float>();
  }
  if (data.contains("rope_scaling") && data["rope_scaling"].is_number_float()) {
    args->rope_scaling() = data["rope_scaling"].get<float>();
  }
  return true;
}

// model args loader for supported models
bool load_gpt2_model_args(const nlohmann::json& data, ModelArgs* args) {
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

bool load_gpt_neox_model_args(const nlohmann::json& data, ModelArgs* args) {
  // example config:
  // https://huggingface.co/EleutherAI/gpt-neox-20b/blob/main/config.json set
  // set default values for args explicitly with values from:
  // https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/configuration_gpt_neox.py#L106
  args->vocab_size() = 50432;
  args->hidden_size() = 6144;
  args->n_layers() = 44;
  args->n_heads() = 64;
  args->intermediate_size() = 24576;
  args->hidden_act() = "gelu";
  args->rotary_pct() = 0.25;
  args->rope_theta() = 10000.0f;
  args->max_position_embeddings() = 2048;
  args->layer_norm_eps() = 1e-5;
  args->bos_token_id() = 0;
  args->eos_token_id() = 2;
  args->use_parallel_residual() = true;

  if (data.contains("vocab_size")) {
    args->vocab_size() = data["vocab_size"].get<int64_t>();
  }
  if (data.contains("hidden_size")) {
    args->hidden_size() = data["hidden_size"].get<int64_t>();
  }
  if (data.contains("num_hidden_layers")) {
    args->n_layers() = data["num_hidden_layers"].get<int64_t>();
  }
  if (data.contains("num_attention_heads")) {
    args->n_heads() = data["num_attention_heads"].get<int64_t>();
  }
  if (data.contains("intermediate_size")) {
    args->intermediate_size() = data["intermediate_size"].get<int64_t>();
  }
  if (data.contains("hidden_act")) {
    args->hidden_act() = data["hidden_act"].get<std::string>();
  }
  if (data.contains("rotary_pct")) {
    args->rotary_pct() = data["rotary_pct"].get<float>();
  }
  if (data.contains("rotary_emb_base")) {
    args->rope_theta() = data["rotary_emb_base"].get<float>();
  }
  if (data.contains("rope_scaling") && data["rope_scaling"].is_number_float()) {
    args->rope_scaling() = data["rope_scaling"].get<float>();
  }
  if (data.contains("max_position_embeddings")) {
    args->max_position_embeddings() =
        data["max_position_embeddings"].get<int64_t>();
  }
  if (data.contains("layer_norm_eps")) {
    args->layer_norm_eps() = data["layer_norm_eps"].get<float>();
  }
  if (data.contains("bos_token_id")) {
    args->bos_token_id() = data["bos_token_id"].get<int32_t>();
  }
  if (data.contains("eos_token_id")) {
    args->eos_token_id() = data["eos_token_id"].get<int32_t>();
  }
  if (data.contains("use_parallel_residual")) {
    args->use_parallel_residual() =
        data["use_parallel_residual"].get<bool>();
  }
  return true;
}

}  // namespace llm
