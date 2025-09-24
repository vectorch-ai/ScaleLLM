#include "model_registry.h"

#include <glog/logging.h>

#include "registered_models.h"  // IWYU pragma: keep

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
