#pragma once
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

#include "models/args.h"
#include "models/causal_lm.h"

namespace llm {

using CausalLMFactory =
    std::function<std::unique_ptr<CausalLM>(const ModelArgs& args,
                                            const QuantizationArgs& quant_args,
                                            const ParallelArgs& parallel_args,
                                            torch::ScalarType dtype,
                                            const torch::Device& device)>;
using ModelArgsLoader =
    std::function<bool(const nlohmann::json& data, ModelArgs* args)>;

using QuantizationArgsLoader =
    std::function<bool(const nlohmann::json& data, QuantizationArgs* args)>;

// TODO: add default args loader.
struct ModelMeta {
  CausalLMFactory causal_lm_factory;
  ModelArgsLoader model_args_loader;
  QuantizationArgsLoader quant_args_loader;
};

// Model registry is a singleton class that registers all models with the
// ModelFactory, ModelArgParser to facilitate model loading.
class ModelRegistry {
 public:
  static ModelRegistry* get() {
    static ModelRegistry registry;
    return &registry;
  }

  void register_causallm_factory(const std::string& name,
                                 CausalLMFactory factory) {
    model_registry_[name].causal_lm_factory = factory;
  }

  void register_model_args_loader(const std::string& name,
                                  ModelArgsLoader loader) {
    model_registry_[name].model_args_loader = loader;
  }

  void register_quant_args_loader(const std::string& name,
                                  QuantizationArgsLoader loader) {
    model_registry_[name].quant_args_loader = loader;
  }

  CausalLMFactory get_causallm_factory(const std::string& name) {
    return model_registry_[name].causal_lm_factory;
  }

  ModelArgsLoader get_model_args_loader(const std::string& name) {
    return model_registry_[name].model_args_loader;
  }

  QuantizationArgsLoader get_quant_args_loader(const std::string& name) {
    return model_registry_[name].quant_args_loader;
  }

 private:
  std::map<std::string, ModelMeta> model_registry_;
};

// Macro to register a model with the ModelRegistry
#define REGISTER_CAUSAL_MODEL(ModelType, ModelClass)                        \
  const bool ModelType##_registered = []() {                                \
    ModelRegistry::get()->register_causallm_factory(                        \
        #ModelType,                                                         \
        [](const ModelArgs& args,                                           \
           const QuantizationArgs& quant_args,                              \
           const ParallelArgs& parallel_args,                               \
           torch::ScalarType dtype,                                         \
           const torch::Device& device) {                                   \
          ModelClass model(args, quant_args, parallel_args, dtype, device); \
          model->eval();                                                    \
          return std::make_unique<llm::CausalLMImpl<ModelClass>>(           \
              std::move(model));                                            \
        });                                                                 \
    return true;                                                            \
  }();

// Macro to register a model args loader with the ModelRegistry
#define REGISTER_MODEL_ARGS_LOADER(Name, Loader)                    \
  const bool Name##_args_loader_registered = []() {                 \
    ModelRegistry::get()->register_model_args_loader(Name, Loader); \
    return true;                                                    \
  }();

// Macro to register a quantization args loader with the ModelRegistry
#define REGISTER_QUANT_ARGS_LOADER(Name, Loader)                    \
  const bool Name##_quant_args_loader_registered = []() {           \
    ModelRegistry::get()->register_quant_args_loader(Name, Loader); \
    return true;                                                    \
  }();

}  // namespace llm
