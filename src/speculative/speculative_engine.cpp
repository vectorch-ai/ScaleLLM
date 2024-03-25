#include "speculative_engine.h"

#include <gflags/gflags_declare.h>
#include <glog/logging.h>

#include <memory>

namespace llm {

SpeculativeEngine::SpeculativeEngine(
    const std::vector<torch::Device>& devices,
    const std::vector<torch::Device>& draft_devices) {
  engine_ = std::make_unique<LLMEngine>(devices);
  draft_engine_ = std::make_unique<LLMEngine>(draft_devices);
}

bool SpeculativeEngine::init(const std::string& model_weights_path,
                             const std::string& draft_model_weights_path) {
  if (!init_model(model_weights_path, draft_model_weights_path)) {
    return false;
  }

  // const int64_t kv_cache_size_in_bytes = profile_memory_for_kv_cache();
  // if (!init_kv_cache(kv_cache_size_in_bytes)) {
  //   LOG(ERROR) << "Failed to initialize kv cache";
  //   return false;
  // }
  return true;
}

bool SpeculativeEngine::init_model(
    const std::string& model_weights_path,
    const std::string& draft_model_weights_path) {
  if (!engine_->init_model(model_weights_path)) {
    return false;
  }
  if (!draft_engine_->init_model(draft_model_weights_path)) {
    return false;
  }
  return true;
}

bool SpeculativeEngine::init_kv_cache(int64_t cache_size_in_bytes) {
  // TODO: share block allocation between engine and draft_engine
  return false;
}

void SpeculativeEngine::execute_model(Batch& batch) {
  LOG(FATAL) << "Not implemented";
}

void SpeculativeEngine::validate(Batch& batch) {
  LOG(FATAL) << "Not implemented";
}

}  // namespace llm
