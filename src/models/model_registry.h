#pragma once
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "causal_lm.h"
#include "chat_template/chat_template.h"
#include "common/json_reader.h"
#include "common/type_traits.h"  // IWYU pragma: keep
#include "model_args.h"
#include "model_parallel/parallel_args.h"
#include "quantization/quant_args.h"
#include "tokenizer/tokenizer_args.h"

namespace llm {

using CausalLMFactory = std::function<std::unique_ptr<CausalLM>(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)>;

using ChatTemplateFactory = std::function<std::unique_ptr<ChatTemplate>()>;

using ModelArgsLoader =
    std::function<bool(const JsonReader& json, ModelArgs* args)>;

using QuantArgsLoader =
    std::function<bool(const JsonReader& json, QuantArgs* args)>;

using TokenizerArgsLoader =
    std::function<bool(const JsonReader& json, TokenizerArgs* args)>;

// TODO: add default args loader.
struct ModelMeta {
  CausalLMFactory causal_lm_factory;
  ChatTemplateFactory chat_template_factory;
  ModelArgsLoader model_args_loader;
  QuantArgsLoader quant_args_loader;
  TokenizerArgsLoader tokenizer_args_loader;
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

  static void register_tokenizer_args_loader(const std::string& name,
                                             TokenizerArgsLoader loader);

  static void register_default_chat_template_factory(
      const std::string& name,
      ChatTemplateFactory factory);

  static CausalLMFactory get_causallm_factory(const std::string& name);

  static ModelArgsLoader get_model_args_loader(const std::string& name);

  static QuantArgsLoader get_quant_args_loader(const std::string& name);

  static TokenizerArgsLoader get_tokenizer_args_loader(const std::string& name);

  static ChatTemplateFactory get_default_chat_template_factory(
      const std::string& name);

 private:
  std::unordered_map<std::string, ModelMeta> model_registry_;
};

// Macro to register a model with the ModelRegistry
#define REGISTER_CAUSAL_MODEL_WITH_VARNAME(VarName, ModelType, ModelClass) \
  const bool VarName##_registered = []() {                                 \
    ModelRegistry::register_causallm_factory(                              \
        #ModelType,                                                        \
        [](const ModelArgs& args,                                          \
           const QuantArgs& quant_args,                                    \
           const ParallelArgs& parallel_args,                              \
           const torch::TensorOptions& options) {                          \
          ModelClass model(args, quant_args, parallel_args, options);      \
          model->eval();                                                   \
          return std::make_unique<llm::CausalLMImpl<ModelClass>>(          \
              std::move(model), options);                                  \
        });                                                                \
    return true;                                                           \
  }()

#define REGISTER_CAUSAL_MODEL(ModelType, ModelClass) \
  REGISTER_CAUSAL_MODEL_WITH_VARNAME(ModelType, ModelType, ModelClass)

#define REGISTER_DEFAULT_CHAT_TEMPLATE_WITH_VARNAME(                         \
    VarName, ModelType, ChatTemplateClass)                                   \
  const bool VarName##_chat_template_registered = []() {                     \
    ModelRegistry::register_default_chat_template_factory(                   \
        #ModelType, []() { return std::make_unique<ChatTemplateClass>(); }); \
    return true;                                                             \
  }()

#define REGISTER_DEFAULT_CHAT_TEMPLATE(ModelType, ChatTemplateClass) \
  REGISTER_DEFAULT_CHAT_TEMPLATE_WITH_VARNAME(                       \
      ModelType, ModelType, ChatTemplateClass)

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
        UNUSED_PARAMETER(json);                                         \
        UNUSED_PARAMETER(args);                                         \
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

// Macro to register a tokenizer args loader with the ModelRegistry
#define REGISTER_TOKENIZER_ARGS_LOADER_WITH_VARNAME(                   \
    VarName, ModelType, Loader)                                        \
  const bool VarName##_tokenizer_args_loader_registered = []() {       \
    ModelRegistry::register_tokenizer_args_loader(#ModelType, Loader); \
    return true;                                                       \
  }()

#define REGISTER_TOKENIZER_ARGS_LOADER(ModelType, Loader) \
  REGISTER_TOKENIZER_ARGS_LOADER_WITH_VARNAME(ModelType, ModelType, Loader)

#define REGISTER_TOKENIZER_ARGS_WITH_VARNAME(VarName, ModelType, ...)       \
  REGISTER_TOKENIZER_ARGS_LOADER_WITH_VARNAME(                              \
      VarName, ModelType, [](const JsonReader& json, TokenizerArgs* args) { \
        UNUSED_PARAMETER(json);                                             \
        UNUSED_PARAMETER(args);                                             \
        __VA_ARGS__();                                                      \
        return true;                                                        \
      })

#define REGISTER_TOKENIZER_ARGS(ModelType, ...) \
  REGISTER_TOKENIZER_ARGS_WITH_VARNAME(ModelType, ModelType, __VA_ARGS__)

#define LOAD_ARG(arg_name, json_name)                          \
  [&] {                                                        \
    auto value = args->arg_name();                             \
    using value_type = remove_optional_t<decltype(value)>;     \
    if (auto data_value = json.value<value_type>(json_name)) { \
      args->arg_name() = data_value.value();                   \
    }                                                          \
  }()

#define LOAD_ARG_OR(arg_name, json_name, default_value)                     \
  [&] {                                                                     \
    auto value = args->arg_name();                                          \
    using value_type = remove_optional_t<decltype(value)>;                  \
    args->arg_name() = json.value_or<value_type>(json_name, default_value); \
  }()

#define LOAD_ARG_OR_FUNC(arg_name, json_name, ...)             \
  [&] {                                                        \
    auto value = args->arg_name();                             \
    using value_type = remove_optional_t<decltype(value)>;     \
    if (auto data_value = json.value<value_type>(json_name)) { \
      args->arg_name() = data_value.value();                   \
    } else {                                                   \
      args->arg_name() = __VA_ARGS__();                        \
    }                                                          \
  }()

#define SET_ARG(arg_name, value) [&] { args->arg_name() = value; }()

}  // namespace llm
