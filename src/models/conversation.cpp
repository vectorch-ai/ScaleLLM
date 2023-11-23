#include "conversation.h"

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace llm {

// add all messages to the conversation one by one
void Conversation::add_message(Role role, const std::string_view& message) {
  switch (role) {
    case Role::User:
      if (last_message_role_ == Role::User) {
        // multiple user messages in a row
        // add a empty assistant message to separate them
        messages_.emplace_back("");
        messages_.push_back(message);
      } else {
        // user message after system/assistant message
        messages_.push_back(message);
      }
      last_message_role_ = role;
      break;
    case Role::Assistant:
      if (last_message_role_ == Role::Assistant) {
        // multiple assistant messages in a row
        // add a empty user message to separate them
        messages_.emplace_back("");
        messages_.push_back(message);
      } else {
        // assistant message after system/user message
        messages_.push_back(message);
      }
      last_message_role_ = role;
      break;
    case Role::System:
      // just track last system messages
      system_message_ = message;
      break;
  }
}

// generate prompt from Conversation
std::optional<std::string> Llama2Conversation::get_prompt() const {
  // at least one user message
  if (messages_.size() % 2 == 0) {
    return std::nullopt;
  }

  std::stringstream ss;
  // start with system message
  // N.B. tokenizer would add <s> to the beginning of the prompt
  if (!system_message_.empty()) {
    ss << "[INST] <<SYS>>\n" << system_message_ << "\n<</SYS>>\n\n";
  } else {
    // no system message
    ss << "[INST] ";
  }

  // then user and assistant message pairs (u/a/u/a/u...)
  for (size_t i = 0; i < messages_.size(); ++i) {
    const char* role = i % 2 == 0 ? "[INST] " : "[/INST] ";
    const char* seps = i % 2 == 0 ? " " : " </s><s>";
    if (i == 0) {
      ss << messages_[i] << " ";
    } else {
      ss << role << messages_[i] << seps;
    }
  }
  // end with assistant message
  ss << "[/INST]";
  return ss.str();
}

}  // namespace llm