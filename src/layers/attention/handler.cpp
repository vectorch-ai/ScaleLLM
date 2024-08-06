#include "handler.h"

#include <c10/core/TensorOptions.h>
#include <gflags/gflags.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <memory>

#include "flash_attn_handler.h"
#include "flash_infer_handler.h"
#include "layers/pos_embedding.h"
#include "ref_handler.h"

// decide which attention implementation to use
DEFINE_string(attention_handler,
              "auto",
              "attention handler, e.g. auto, pytorch, flash_attn");

namespace llm {

namespace {
torch::Tensor compute_inv_freq(int64_t rotary_dim, const ModelArgs& args) {
  auto inv_freq =
      detail::compute_default_inv_freq(rotary_dim, args.rope_theta());
  if (boost::iequals(args.rope_scaling_rope_type(), "llama3")) {
    return detail::apply_llama3_rope_scaling(
        inv_freq,
        args.rope_scaling_factor(),
        args.rope_scaling_low_freq_factor(),
        args.rope_scaling_high_freq_factor(),
        args.rope_scaling_original_max_position_embeddings());
  }

  if (!args.rope_scaling_rope_type().empty()) {
    LOG(FATAL) << "Unsupported rope scaling type: "
               << args.rope_scaling_rope_type();
  }
  return inv_freq;
}

}  // namespace

// create an attention handler with alibi slopes
std::unique_ptr<AttentionHandler> AttentionHandler::create_handler_with_alibi(
    const ModelArgs& args,
    torch::optional<torch::Tensor> alibi_slopes,
    const torch::TensorOptions& options) {
  const int64_t head_dim = args.hidden_size() / args.n_heads();

  const float attn_scale =
      args.attn_scalar().value_or(static_cast<float>(head_dim));
  const float sm_scale = 1.0f / std::sqrt(attn_scale);

  if (alibi_slopes.has_value()) {
    // move alibi slopes to the same device as the model
    alibi_slopes = alibi_slopes.value().to(options.device());
  }

  // check if the user specified the attention handler
  if (boost::iequals(FLAGS_attention_handler, "pytorch")) {
    return std::make_unique<RefHandler>(
        sm_scale, args.attn_logit_soft_cap(), alibi_slopes);
  }

  const bool is_cuda = options.device().is_cuda();
  if (boost::iequals(FLAGS_attention_handler, "flash_attn")) {
    CHECK(is_cuda) << "flash_attn only supports cuda device";
    return std::make_unique<FlashAttnHandler>(
        sm_scale, args.attn_logit_soft_cap(), alibi_slopes);
  }

  // choose the best handler based on device type
  if (is_cuda) {
    // use flash_attn for cuda device
    return std::make_unique<FlashAttnHandler>(
        sm_scale, args.attn_logit_soft_cap(), alibi_slopes);
  }

  // use slower ref handler for other devices for now.
  return std::make_unique<RefHandler>(
      sm_scale, args.attn_logit_soft_cap(), alibi_slopes);
}

// create an attention handler with ROPE
std::unique_ptr<AttentionHandler> AttentionHandler::create_handler_with_rope(
    const ModelArgs& args,
    bool interleaved,
    const torch::TensorOptions& options) {
  const int64_t head_dim = args.head_dim();
  // default to use head_dim if rotary_dim is not specified
  int64_t rotary_dim = args.rotary_dim() > 0 ? args.rotary_dim() : head_dim;
  // apply rotary_dim percentage
  rotary_dim = static_cast<int64_t>(rotary_dim * args.rotary_pct());

  const float attn_scale =
      args.attn_scalar().value_or(static_cast<float>(head_dim));
  const float sm_scale = 1.0f / std::sqrt(attn_scale);

  const auto inv_freq = compute_inv_freq(rotary_dim, args);

  // check if the user specified the attention handler
  if (boost::iequals(FLAGS_attention_handler, "pytorch")) {
    return std::make_unique<RefHandler>(sm_scale,
                                        args.attn_logit_soft_cap(),
                                        rotary_dim,
                                        args.max_position_embeddings(),
                                        inv_freq,
                                        interleaved,
                                        options);
  }

  const bool is_cuda = options.device().is_cuda();
  if (boost::iequals(FLAGS_attention_handler, "flash_attn")) {
    CHECK(is_cuda) << "flash_attn only supports cuda device";
    return std::make_unique<FlashAttnHandler>(sm_scale,
                                              args.attn_logit_soft_cap(),
                                              rotary_dim,
                                              args.max_position_embeddings(),
                                              inv_freq,
                                              interleaved,
                                              options);
  }

  // choose the best handler based on device type
  if (is_cuda) {
    // use flash_attn for cuda device
    return std::make_unique<FlashAttnHandler>(sm_scale,
                                              args.attn_logit_soft_cap(),
                                              rotary_dim,
                                              args.max_position_embeddings(),
                                              inv_freq,
                                              interleaved,
                                              options);
  }

  // use slower ref handler for other devices for now.
  return std::make_unique<RefHandler>(sm_scale,
                                      args.attn_logit_soft_cap(),
                                      rotary_dim,
                                      args.max_position_embeddings(),
                                      inv_freq,
                                      interleaved,
                                      options);
}

}  // namespace llm
