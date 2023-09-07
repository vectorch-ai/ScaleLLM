#include "pos_embedding.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>

namespace llm {

namespace {
// create right instance based on params
std::shared_ptr<RotaryEmbeddingImpl> create(int64_t rotary_dim,
                                            int64_t max_seq_len,
                                            float scaling_factor,
                                            bool interleaved,
                                            const torch::Device& device) {
  if (interleaved) {
    return std::make_shared<InterleavedRotaryEmbedding>(
        rotary_dim, max_seq_len, scaling_factor, device);
  }
  return std::make_shared<RotatedRotaryEmbedding>(
      rotary_dim, max_seq_len, scaling_factor, device);
}
}  // namespace

RotaryEmbedding::RotaryEmbedding(int64_t rotary_dim,
                                 int64_t max_seq_len,
                                 float scaling_factor,
                                 bool interleaved,
                                 const torch::Device& device)
    : ModuleHolder(create(rotary_dim,
                          max_seq_len,
                          scaling_factor,
                          interleaved,
                          device)) {}

}  // namespace llm
