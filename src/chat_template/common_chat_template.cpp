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

// generate prompt from ChatTemplate
std::optional<std::string> Llama3ChatTemplate::get_prompt(
    const std::string_view& system_message,
    const std::vector<std::string_view>& messages) const {
  // at least one user message
  if (messages.size() % 2 == 0) {
    return std::nullopt;
  }

  std::stringstream ss;
  ss << "<|begin_of_text|>";
  auto add_message = [&ss](const std::string_view& role,
                           const std::string_view& message) {
    ss << "<|start_header_id|>" << role << "<|end_header_id|>\n\n";
    ss << message << "<|eot_id|>";
  };

  // start with system message
  if (!system_message.empty()) {
    add_message("system", system_message);
  }

  // then user and assistant message pairs (u/a/u/a/u...)
  for (size_t i = 0; i < messages.size(); ++i) {
    const char* role = i % 2 == 0 ? "user" : "assistant";
    add_message(role, messages[i]);
  }
  // end with assistant message
  ss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
  return ss.str();
}

}  // namespace llm