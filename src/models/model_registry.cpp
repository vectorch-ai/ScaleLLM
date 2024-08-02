#include "model_registry.h"

#include <glog/logging.h>

// list all registered models here
#include "huggingface/aquila.h"    // IWYU pragma: keep
#include "huggingface/baichuan.h"  // IWYU pargma: keep
#include "huggingface/bloom.h"     // IWYU pragma: keep
#include "huggingface/chatglm.h"   // IWYU pragma: keep
#include "huggingface/gemma.h"     // IWYU pragma: keep
#include "huggingface/gpt2.h"      // IWYU pragma: keep
#include "huggingface/gpt_j.h"     // IWYU pragma: keep
#include "huggingface/gpt_neox.h"  // IWYU pragma: keep
#include "huggingface/internlm.h"  // IWYU pragma: keep
#include "huggingface/llama.h"     // IWYU pragma: keep
#include "huggingface/mistral.h"   // IWYU pragma: keep
#include "huggingface/mpt.h"       // IWYU pragma: keep
#include "huggingface/phi.h"       // IWYU pragma: keep
#include "huggingface/qwen.h"      // IWYU pragma: keep
#include "huggingface/qwen2.h"      // IWYU pragma: keep
#include "llama.h"                 // IWYU pragma: keep

namespace llm {

ModelRegistry* ModelRegistry::get_instance() {
  static ModelRegistry registry;
  return &registry;
}

void ModelRegistry::register_causallm_factory(const std::string& name,
                                              CausalLMFactory factory) {
  ModelRegistry* instance = get_instance();
  if (instance->model_registry_[name].causal_lm_factory != nullptr) {
    LOG(WARNING) << "causal lm factory for " << name << "already registered.";
  } else {
    instance->model_registry_[name].causal_lm_factory = factory;
  }
}

void ModelRegistry::register_model_args_loader(const std::string& name,
                                               ModelArgsLoader loader) {
  ModelRegistry* instance = get_instance();
  if (instance->model_registry_[name].model_args_loader != nullptr) {
    LOG(WARNING) << "model args loader for " << name << "already registered.";
  } else {
    instance->model_registry_[name].model_args_loader = loader;
  }
}

void ModelRegistry::register_quant_args_loader(const std::string& name,
                                               QuantArgsLoader loader) {
  ModelRegistry* instance = get_instance();
  if (instance->model_registry_[name].quant_args_loader != nullptr) {
    LOG(WARNING) << "quant args loader for " << name << "already registered.";
  } else {
    instance->model_registry_[name].quant_args_loader = loader;
  }
}

void ModelRegistry::register_tokenizer_args_loader(const std::string& name,
                                                   TokenizerArgsLoader loader) {
  ModelRegistry* instance = get_instance();
  if (instance->model_registry_[name].tokenizer_args_loader != nullptr) {
    LOG(WARNING) << "tokenizer args loader for " << name
                 << "already registered.";
  } else {
    instance->model_registry_[name].tokenizer_args_loader = loader;
  }
}

void ModelRegistry::register_default_chat_template_factory(
    const std::string& name,
    ChatTemplateFactory factory) {
  ModelRegistry* instance = get_instance();
  if (instance->model_registry_[name].chat_template_factory != nullptr) {
    LOG(WARNING) << "conversation template for " << name
                 << "already registered.";
  } else {
    instance->model_registry_[name].chat_template_factory = factory;
  }
}

CausalLMFactory ModelRegistry::get_causallm_factory(const std::string& name) {
  ModelRegistry* instance = get_instance();
  return instance->model_registry_[name].causal_lm_factory;
}

ModelArgsLoader ModelRegistry::get_model_args_loader(const std::string& name) {
  ModelRegistry* instance = get_instance();
  return instance->model_registry_[name].model_args_loader;
}

QuantArgsLoader ModelRegistry::get_quant_args_loader(const std::string& name) {
  ModelRegistry* instance = get_instance();
  return instance->model_registry_[name].quant_args_loader;
}

TokenizerArgsLoader ModelRegistry::get_tokenizer_args_loader(
    const std::string& name) {
  ModelRegistry* instance = get_instance();
  return instance->model_registry_[name].tokenizer_args_loader;
}

ChatTemplateFactory ModelRegistry::get_default_chat_template_factory(
    const std::string& name) {
  ModelRegistry* instance = get_instance();
  return instance->model_registry_[name].chat_template_factory;
}

}  // namespace llm
