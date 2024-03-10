#include "handler.h"

#include <gflags/gflags.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <memory>

#include "flash_attn_handler.h"
#include "flash_infer_handler.h"
#include "ref_handler.h"

// decide which attention implementation to use
DEFINE_string(attention_handler,
              "auto",
              "attention handler, e.g. auto, flash_attn, flash_infer");

namespace llm {

std::unique_ptr<AttentionHandler> AttentionHandler::create(
    const ModelArgs& args,
    const torch::Device& device,
    torch::optional<torch::Tensor> alibi_slopes) {
  const int64_t head_dim = args.hidden_size() / args.n_heads();
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // check if the user specified the attention handler
  if (boost::iequals(FLAGS_attention_handler, "flash_attn")) {
    CHECK(device.is_cuda()) << "flash_attn only supports cuda device";
    return std::make_unique<FlashAttnHandler>(scale, alibi_slopes);
  }
  if (boost::iequals(FLAGS_attention_handler, "flash_infer")) {
    CHECK(device.is_cuda()) << "flash_infer only supports cuda device";
    return std::make_unique<FlashInferHandler>(scale, alibi_slopes);
  }

  if (device.is_cuda()) {
    // use flash_attn for cuda device
    return std::make_unique<FlashAttnHandler>(scale, alibi_slopes);
  }

  // use slower ref handler for other devices for now.
  return std::make_unique<RefHandler>(scale, alibi_slopes);
}

}  // namespace llm
