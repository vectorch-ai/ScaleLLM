#include "jinja_chat_template.h"

#include <jinja2cpp/binding/nlohmann_json.h>
#include <jinja2cpp/value.h>

#include <optional>
#include <string>

#include "common/logging.h"

namespace llm {

JinjaChatTemplate::JinjaChatTemplate(const std::string& template_str,
                                     bool add_generation_prompt)
    : add_generation_prompt_(add_generation_prompt) {
  if (!template_.Load(template_str)) {
    GLOG(FATAL) << "Failed to load template: " << template_str;
  }
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages) const {
  // convert the messages to json object
  nlohmann::json messages_json = nlohmann::json::array();
  for (const auto& message : messages) {
    nlohmann::json message_json;
    message_json["role"] = message.role();
    message_json["content"] = message.content();
    messages_json.push_back(message_json);
  }
  // apply the template
  return apply(messages_json);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::json& messages) const {
  jinja2::ValuesMap values;
  // add the messages to the values
  values["messages"] = jinja2::Reflect(messages);
  // add the generation prompt
  values["add_generation_prompt"] = add_generation_prompt_;
  // render the template
  return template_.RenderAsString(values).value();
}

}  // namespace llm