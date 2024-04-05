#pragma once

#include <string_view>

#include "engine.h"

namespace llm {

class EngineFactory {
 public:
  static std::unique_ptr<Engine> create(
      const std::string& model_path,
      const std::string& devices_str,
      const std::string& draft_model_path,
      const std::string& draft_devices_str);

  static std::unique_ptr<Engine> create(
      const std::string& model_path,
      const std::string& devices_str);
};

}  // namespace llm
