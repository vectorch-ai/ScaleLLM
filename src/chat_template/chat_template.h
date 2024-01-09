#pragma once
#include <optional>
#include <string>

#include "chat.pb.h"

namespace llm {

// It is not ideal to use protobuf as the interface between the server and the
// model. However, it is the easiest way to do it. We can change it later if
// needed.
using ChatMessages = google::protobuf::RepeatedPtrField<ChatMessage>;

// ChatTemplate only supports 'system', 'user' and 'assistant' roles.
// start with system message, then user and assistant message. (u/a/u/a/u...)
class ChatTemplate {
 public:
  virtual ~ChatTemplate() = default;

  // apply the chat template to the messages
  // return std::nullopt if the template is not applicable.
  virtual std::optional<std::string> apply(const ChatMessages& messages) const = 0;
};

}  // namespace llm