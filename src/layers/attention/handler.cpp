#include "handler.h"

#include <gflags/gflags.h>
#include <torch/torch.h>

#include <memory>

namespace llm {

// create an attention handler with alibi
std::unique_ptr<AttentionHandler> AttentionHandler::create_handler(
    float scale,
    torch::optional<torch::Tensor> alibi_slopes) {
  return nullptr;
}

}  // namespace llm
