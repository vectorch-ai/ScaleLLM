#include "common_chat_template.h"

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace llm {

// generate prompt from ChatTemplate
std::optional<std::string> Llama2ChatTemplate::get_prompt(
    const std::string_view& system_message,
    const std::vector<std::string_view>& messages) const {
  // at least one user message
  if (messages.size() % 2 == 0) {
    return std::nullopt;
  }

  std::stringstream ss;
  // start with system message
  // N.B. tokenizer would add <s> to the beginning of the prompt
  if (!system_message.empty()) {
    ss << "[INST] <<SYS>>\n" << system_message << "\n<</SYS>>\n\n";
  } else {
    // no system message
    ss << "[INST] ";
  }

  // then user and assistant message pairs (u/a/u/a/u...)
  for (size_t i = 0; i < messages.size(); ++i) {
    const char* role = i % 2 == 0 ? "[INST] " : "[/INST] ";
    const char* seps = i % 2 == 0 ? " " : " </s><s>";
    if (i == 0) {
      ss << messages[i] << " ";
    } else {
      ss << role << messages[i] << seps;
    }
  }
  // end with assistant message
  ss << "[/INST]";
  return ss.str();
}

}  // namespace llm