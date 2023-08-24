#include "pos_embedding.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>

namespace llm {

namespace {
// create right instance based on params
std::shared_ptr<RotaryEmbeddingImpl> create(int64_t rotary_dim,
                                            int64_t max_seq_len,
                                            bool interleaved,
                                            const torch::Device& device) {
  if (interleaved) {
    return std::make_shared<InterleavedRotaryEmbedding>(
        rotary_dim, max_seq_len, device);
  }
  return std::make_shared<RotatedRotaryEmbedding>(
      rotary_dim, max_seq_len, device);
}
}  // namespace

RotaryEmbedding::RotaryEmbedding(int64_t rotary_dim,
                                 int64_t max_seq_len,
                                 bool interleaved,
                                 const torch::Device& device)
    : ModuleHolder(create(rotary_dim, max_seq_len, interleaved, device)) {}

}  // namespace llm