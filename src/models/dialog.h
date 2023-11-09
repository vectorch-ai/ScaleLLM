#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llm {

// dialog only supports 'system', 'user' and 'assistant' roles.
// start with system message, then user and assistant message. (u/a/u/a/u...)
class Dialog {
 public:
  Dialog() = default;
  virtual ~Dialog() = default;

  enum class Role : int8_t {
    User = 0,
    Assistant = 1,
    System = 2,
  };

  // add all messages to the dialog one by one
  void add_message(Role role, const std::string_view& message);

  // generate prompt from dialogs
  // return empty optional if no prompt is generated
  virtual std::optional<std::string> get_prompt() const = 0;

 protected:
  Role last_message_role_ = Role::System;

  // accumulate system, user and assistant messages
  std::vector<std::string_view> messages_;
  std::string_view system_message_;
};

// dialog conversation for llama2 model
class Llama2Dialog final : public Dialog {
 public:
  // generate prompt from dialogs
  std::optional<std::string> get_prompt() const override;
};

}  // namespace llm