#include "coded_chat_template.h"

#include <glog/logging.h>

#include <boost/algorithm/string.hpp>
#include <optional>
#include <string>

#include "chat_template.h"

namespace llm {

std::optional<std::string> CodedChatTemplate::apply(
    const ChatMessages& messages) const {
  // construct messages from the chat messages
  enum class Role : int8_t {
    User = 0,
    Assistant = 1,
    System = 2,
  };

  std::vector<std::string_view> msgs;
  std::string_view system_message;
  Role last_message_role = Role::System;
  for (const auto& message : messages) {
    if (boost::iequals(message.role(), "system")) {
      system_message = message.content();
      last_message_role = Role::System;
    } else if (boost::iequals(message.role(), "user")) {
      if (last_message_role == Role::User) {
        // multiple user messages in a row
        // add a empty assistant message to separate them
        msgs.emplace_back("");
        msgs.push_back(message.content());
      } else {
        // user message after system/assistant message
        msgs.push_back(message.content());
      }
      last_message_role = Role::User;
    } else if (boost::iequals(message.role(), "assistant")) {
      if (last_message_role == Role::Assistant) {
        // multiple assistant messages in a row
        // add a empty user message to separate them
        msgs.emplace_back("");
        msgs.push_back(message.content());
      } else {
        // assistant message after system/user message
        msgs.push_back(message.content());
      }
      last_message_role = Role::Assistant;
    } else {
      LOG(ERROR) << "Unknown message role: " << message.role();
      return nullptr;
    }
  }
  // generate prompt from the messages
  return get_prompt(system_message, msgs);
}

}  // namespace llm