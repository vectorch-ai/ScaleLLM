#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "chat_template/coded_chat_template.h"

namespace llm {

// dialog conversation for llama2 model
class Llama2ChatTemplate final : public CodedChatTemplate {
 public:
  // generate prompt from dialogs
  std::optional<std::string> get_prompt(
      const std::string_view& system_message,
      const std::vector<std::string_view>& messages) const override;
};

class Llama3ChatTemplate final : public CodedChatTemplate {
 public:
  // generate prompt from dialogs
  std::optional<std::string> get_prompt(
      const std::string_view& system_message,
      const std::vector<std::string_view>& messages) const override;
};

}  // namespace llm