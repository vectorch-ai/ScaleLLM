#pragma once
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "args.h"
#include "causal_lm.h"
#include "common/json_reader.h"
#include "conversation.h"

namespace llm {

using CausalLMFactory =
    std::function<std::unique_ptr<CausalLM>(const ModelArgs& args,
                                            const QuantArgs& quant_args,
                                            const ParallelArgs& parallel_args,
                                            torch::ScalarType dtype,
                                            const torch::Device& device)>;

using ConversationTemplate = std::function<std::unique_ptr<Conversation>()>;

using ModelArgsLoader =
    std::function<bool(const JsonReader& json, ModelArgs* args)>;

using QuantArgsLoader =
    std::function<bool(const JsonReader& json, QuantArgs* args)>;

// TODO: add default args loader.
struct ModelMeta {
  CausalLMFactory causal_lm_factory;
  ConversationTemplate conversation_template;
  ModelArgsLoader model_args_loader;
  QuantArgsLoader quant_args_loader;
};

// Model registry is a singleton class that registers all models with the
// ModelFactory, ModelArgParser to facilitate model loading.
class ModelRegistry {
 public:
  static ModelRegistry* get_instance();

  static void register_causallm_factory(const std::string& name,
                                        CausalLMFactory factory);

  static void register_model_args_loader(const std::string& name,
                                         ModelArgsLoader loader);

  static void register_quant_args_loader(const std::string& name,
                                         QuantArgsLoader loader);

  static void register_conversation_template(const std::string& name,
                                             ConversationTemplate factory);

  static CausalLMFactory get_causallm_factory(const std::string& name);

  static ModelArgsLoader get_model_args_loader(const std::string& name);

  static QuantArgsLoader get_quant_args_loader(const std::string& name);

  static ConversationTemplate get_conversation_template(
      const std::string& name);

 private:
  std::unordered_map<std::string, ModelMeta> model_registry_;
};

// Macro to register a model with the ModelRegistry
#define REGISTER_CAUSAL_MODEL_WITH_VARNAME(VarName, ModelType, ModelClass)  \
  const bool VarName##_registered = []() {                                  \
    ModelRegistry::register_causallm_factory(                               \
        #ModelType,                                                         \
        [](const ModelArgs& args,                                           \
           const QuantArgs& quant_args,                                     \
           const ParallelArgs& parallel_args,                               \
           torch::ScalarType dtype,                                         \
           const torch::Device& device) {                                   \
          ModelClass model(args, quant_args, parallel_args, dtype, device); \
          model->eval();                                                    \
          return std::make_unique<llm::CausalLMImpl<ModelClass>>(           \
              std::move(model));                                            \
        });                                                                 \
    return true;                                                            \
  }()

#define REGISTER_CAUSAL_MODEL(ModelType, ModelClass) \
  REGISTER_CAUSAL_MODEL_WITH_VARNAME(ModelType, ModelType, ModelClass)

#define REGISTER_CONVERSATION_TEMPLATE_WITH_VARNAME(                  \
    VarName, ModelType, ModelClass)                                   \
  const bool VarName##_dialog_registered = []() {                     \
    ModelRegistry::register_conversation_template(                    \
        #ModelType, []() { return std::make_unique<ModelClass>(); }); \
    return true;                                                      \
  }()

#define REGISTER_CONVERSATION_TEMPLATE(ModelType, ModelClass) \
  REGISTER_CONVERSATION_TEMPLATE_WITH_VARNAME(ModelType, ModelType, ModelClass)

// Macro to register a model args loader with the ModelRegistry
#define REGISTER_MODEL_ARGS_LOADER_WITH_VARNAME(VarName, ModelType, Loader) \
  const bool VarName##_args_loader_registered = []() {                      \
    ModelRegistry::register_model_args_loader(#ModelType, Loader);          \
    return true;                                                            \
  }()

#define REGISTER_MODEL_ARGS_LOADER(ModelType, Loader) \
  REGISTER_MODEL_ARGS_LOADER_WITH_VARNAME(ModelType, ModelType, Loader)

#define REGISTER_MODEL_ARGS_WITH_VARNAME(VarName, ModelType, ...)       \
  REGISTER_MODEL_ARGS_LOADER_WITH_VARNAME(                              \
      VarName, ModelType, [](const JsonReader& json, ModelArgs* args) { \
        __VA_ARGS__();                                                  \
        return true;                                                    \
      })

#define REGISTER_MODEL_ARGS(ModelType, ...) \
  REGISTER_MODEL_ARGS_WITH_VARNAME(ModelType, ModelType, __VA_ARGS__)

// Macro to register a quantization args loader with the ModelRegistry
#define REGISTER_QUANT_ARGS_LOADER_WITH_VARNAME(VarName, ModelType, Loader) \
  const bool VarName##_quant_args_loader_registered = []() {                \
    ModelRegistry::register_quant_args_loader(#ModelType, Loader);          \
    return true;                                                            \
  }()

#define REGISTER_QUANT_ARGS_LOADER(ModelType, Loader) \
  REGISTER_QUANT_ARGS_LOADER_WITH_VARNAME(ModelType, ModelType, Loader)

template <typename type>
struct RemoveOptional {
  using value_type = type;
};

// specialization for optional
template <typename type>
struct RemoveOptional<std::optional<type>> {
  using value_type = type;
};

#define LOAD_ARG_OR(arg_name, json_name, default_value)                      \
  [&] {                                                                      \
    auto value = args->arg_name();                                           \
    using value_type = typename RemoveOptional<decltype(value)>::value_type; \
    args->arg_name() = json.value_or<value_type>(json_name, default_value);  \
  }()

#define LOAD_ARG(arg_name, json_name)                                        \
  [&] {                                                                      \
    auto value = args->arg_name();                                           \
    using value_type = typename RemoveOptional<decltype(value)>::value_type; \
    if (auto data_value = json.value<value_type>(json_name)) {               \
      args->arg_name() = data_value.value();                                 \
    }                                                                        \
  }()

#define LOAD_ARG_OR_FUNC(arg_name, json_name, ...)                           \
  [&] {                                                                      \
    auto value = args->arg_name();                                           \
    using value_type = typename RemoveOptional<decltype(value)>::value_type; \
    if (auto data_value = json.value<value_type>(json_name)) {               \
      args->arg_name() = data_value.value();                                 \
    } else {                                                                 \
      args->arg_name() = __VA_ARGS__();                                      \
    }                                                                        \
  }()

}  // namespace llm
