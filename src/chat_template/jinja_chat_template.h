#pragma once

#include <jinja2cpp/template.h>

#include <nlohmann/json.hpp>
#include <optional>
#include <string>

#include "chat_template.h"

namespace llm {

// A chat template implementation that uses jinja2 as the template engine.
class JinjaChatTemplate : public ChatTemplate {
 public:
  JinjaChatTemplate(const std::string& template_str,
                    bool add_generation_prompt);

  std::optional<std::string> apply(const ChatMessages& messages) const override;

  // expose this function for testing
  // apply the template to the values in the json object
  std::optional<std::string> apply(nlohmann::json& messages) const;

 private:
  mutable jinja2::Template template_;
  bool add_generation_prompt_;
};

}  // namespace llm