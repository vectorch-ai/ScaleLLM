#include "jinja_chat_template.h"

#include <gtest/gtest.h>

namespace llm {

TEST(JinjaChatTemplate, YiModel) {
  // clang-format off
  const std::string template_str =
      "{% if add_generation_prompt is undefined %}{% set add_generation_prompt = false %}{% endif %}"
      "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
      "{% endfor %}"
      "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
  nlohmann::json messages = {
      {{"role", "system"}, {"content", "you are a helpful assistant."}},
      {{"role", "user"}, {"content", "hi"}},
      {{"role", "assistant"}, {"content", "what i can do for you?"}},
      {{"role", "user"}, {"content", "how are you?"}}};
  const std::string expected = 
    "<|im_start|>system\n"
    "you are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "hi<|im_end|>\n"
    "<|im_start|>assistant\n"
    "what i can do for you?<|im_end|>\n"
    "<|im_start|>user\n"
    "how are you?<|im_end|>\n"
    "<|im_start|>assistant\n";
  // clang-format on

  JinjaChatTemplate template_(template_str, true);
  auto result = template_.apply(messages);
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result.value(), expected);
}

TEST(JinjaChatTemplate, OpenChatModel) {
  // clang-format off
  const std::string template_str =
      "<s>"
      "{% for message in messages %}"
        "{{ 'GPT4 Correct ' + message['role'] + ': ' + message['content'] + '<|end_of_turn|>'}}"
      "{% endfor %}"
      "{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}";

  nlohmann::json messages = {
      {{"role", "system"}, {"content", "you are a helpful assistant."}},
      {{"role", "user"}, {"content", "hi"}},
      {{"role", "assistant"}, {"content", "what i can do for you?"}},
      {{"role", "user"}, {"content", "how are you?"}}};
  const std::string expected = 
    "<s>"
    "GPT4 Correct system: you are a helpful assistant.<|end_of_turn|>"
    "GPT4 Correct user: hi<|end_of_turn|>"
    "GPT4 Correct assistant: what i can do for you?<|end_of_turn|>"
    "GPT4 Correct user: how are you?<|end_of_turn|>"
    "GPT4 Correct Assistant:";
  // clang-format on

  JinjaChatTemplate template_(template_str, true);
  auto result = template_.apply(messages);
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result.value(), expected);
}

}  // namespace llm
