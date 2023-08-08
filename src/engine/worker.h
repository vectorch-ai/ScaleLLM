#pragma once

#include <string>
#include "executor.h"

namespace llm {
class CacheManager;

class Worker final {
 public:
  Worker(const std::string& model_path, uint32_t rank)
      : model_path_(model_path) {}

  ~Worker() = default;

  // Load the model from the given path. blocking call
  void load_model();

  // Run the model on the given input. async call
  void run_model();

  // initialize cache manager. async call
  void init_cache_manager();

 private:
  // model path
  std::string model_path_;

  // working thread
  Executor executor_;

  // cache manager
  std::unique_ptr<CacheManager> cache_manager_;
};

}  // namespace llm
