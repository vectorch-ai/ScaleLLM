#include "pos_embedding.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>

namespace llm {

// static factory method
std::shared_ptr<RotaryEmbedding> RotaryEmbedding::create(
    int64_t rotary_dim,
    int64_t max_seq_len,
    bool interleaved) {
  if (interleaved) {
    return std::make_shared<InterleavedRotaryEmbedding>(rotary_dim,
                                                        max_seq_len);
  }
  return std::make_shared<RotatedRotaryEmbedding>(rotary_dim, max_seq_len);
}

}  // namespace llm
