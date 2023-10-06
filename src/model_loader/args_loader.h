#pragma once

#include <nlohmann/json.hpp>

#include "models/args.h"

namespace llm {

// model args loader for supported models
bool load_meta_llama_model_args(const nlohmann::json& data, ModelArgs* args);

bool load_llama_model_args(const nlohmann::json& data, ModelArgs* args);

bool load_gpt2_model_args(const nlohmann::json& data, ModelArgs* args);

bool load_gptj_model_args(const nlohmann::json& data, ModelArgs* args);

bool load_gpt_neox_model_args(const nlohmann::json& data, ModelArgs* args);

bool load_mistral_model_args(const nlohmann::json& data, ModelArgs* args);

bool load_aquila_model_args(const nlohmann::json& data, ModelArgs* args);

}  // namespace llm
