#include "model_registry.h"

#include "common/logging.h"

// list all registered models here
#include "huggingface/aquila.h"
#include "huggingface/bloom.h"
#include "huggingface/chatglm.h"
#include "huggingface/gpt2.h"
#include "huggingface/gpt_j.h"
#include "huggingface/gpt_neox.h"
#include "huggingface/internlm.h"
#include "huggingface/llama.h"
#include "huggingface/mistral.h"
#include "huggingface/mpt.h"
#include "huggingface/qwen.h"
#include "huggingface/yi.h"
#include "llama.h"

namespace llm {

ModelRegistry* ModelRegistry::get_instance() {
  static ModelRegistry registry;
  return &registry;
}

void ModelRegistry::register_causallm_factory(const std::string& name,
                                              CausalLMFactory factory) {
  ModelRegistry* instance = get_instance();
  if (instance->model_registry_[name].causal_lm_factory != nullptr) {
    GLOG(WARNING) << "causal lm factory for " << name << "already registered.";
  } else {
    instance->model_registry_[name].causal_lm_factory = factory;
  }
}

void ModelRegistry::register_model_args_loader(const std::string& name,
                                               ModelArgsLoader loader) {
  ModelRegistry* instance = get_instance();
  if (instance->model_registry_[name].model_args_loader != nullptr) {
    GLOG(WARNING) << "model args loader for " << name << "already registered.";
  } else {
    instance->model_registry_[name].model_args_loader = loader;
  }
}

void ModelRegistry::register_quant_args_loader(const std::string& name,
                                               QuantArgsLoader loader) {
  ModelRegistry* instance = get_instance();
  if (instance->model_registry_[name].quant_args_loader != nullptr) {
    GLOG(WARNING) << "quant args loader for " << name << "already registered.";
  } else {
    instance->model_registry_[name].quant_args_loader = loader;
  }
}

void ModelRegistry::register_conversation_template(
    const std::string& name,
    ConversationTemplate factory) {
  ModelRegistry* instance = get_instance();
  if (instance->model_registry_[name].conversation_template != nullptr) {
    GLOG(WARNING) << "conversation template for " << name
                  << "already registered.";
  } else {
    instance->model_registry_[name].conversation_template = factory;
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

ConversationTemplate ModelRegistry::get_conversation_template(
    const std::string& name) {
  ModelRegistry* instance = get_instance();
  return instance->model_registry_[name].conversation_template;
}

}  // namespace llm
