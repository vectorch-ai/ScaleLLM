#pragma once

#include <optional>
#include <string>

#include "chat_template.h"

namespace llm {

// A chat template implementation that embeds template logic in the code.
class CodedChatTemplate : public ChatTemplate {
 public:
  CodedChatTemplate() = default;

  std::optional<std::string> apply(const ChatMessages& messages) const override;

  // generate prompt from dialogs
  virtual std::optional<std::string> get_prompt(
      const std::string_view& system_message,
      const std::vector<std::string_view>& messages) const = 0;
};

}  // namespace llm