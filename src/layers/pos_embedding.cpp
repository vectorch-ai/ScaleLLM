#include "pos_embedding.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>

namespace llm {

namespace {
// create right instance based on params
std::shared_ptr<RotaryEmbeddingImpl> create(int64_t rotary_dim,
                                            int64_t max_seq_len,
                                            bool interleaved) {
  if (interleaved) {
    return std::make_shared<InterleavedRotaryEmbedding>(rotary_dim,
                                                        max_seq_len);
  }
  return std::make_shared<RotatedRotaryEmbedding>(rotary_dim, max_seq_len);
}
}  // namespace

RotaryEmbedding::RotaryEmbedding(int64_t rotary_dim,
                                 int64_t max_seq_len,
                                 bool interleaved)
    : ModuleHolder(create(rotary_dim, max_seq_len, interleaved)) {}

}  // namespace llm
