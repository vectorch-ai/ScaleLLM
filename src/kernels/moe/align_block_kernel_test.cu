#include <ATen/ops/allclose.h>
#include <ATen/ops/equal.h>
#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cute/layout.hpp>

#include "cute/int_tuple.hpp"
#include "gtest/gtest.h"

namespace llm {

namespace kernel::moe {
void permute_align_block(torch::Tensor topk_ids,
                         int64_t num_experts,
                         int64_t block_size,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor experts_ids,
                         torch::Tensor num_tokens_post_pad,
                         torch::Tensor cu_sum);

}  // namespace kernel::moe

namespace {
bool prefix_equal(torch::Tensor val, torch::Tensor ref) {
  const int64_t expected_size = ref.numel();
  if (val.numel() < expected_size) {
    return false;
  }
  // prefix equal
  return torch::equal(
      ref, val.flatten().slice(/*dim=*/0, /*start=*/0, /*end=*/expected_size));
}
// reference implementation
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> permute_align_block_ref(
    torch::Tensor topk_ids,  // [n_tokens, topk]
    int64_t n_experts,
    int64_t block_size) {
  const int64_t n_tokens = topk_ids.size(0);
  const int64_t topk = topk_ids.size(1);
  const int64_t n_flatten_tokens = topk_ids.numel();

  auto topk_ids_cpu = topk_ids.cpu().contiguous();
  const int32_t* topk_ids_ptr = topk_ids_cpu.data_ptr<int32_t>();

  std::map<int32_t, std::vector<int32_t>> expert_to_idxes;
  for (int i = 0; i < n_flatten_tokens; ++i) {
    const int32_t expert_id = topk_ids_ptr[i];
    expert_to_idxes[expert_id].push_back(i);
  }
  // LOG(ERROR) << "expert_to_idxes: " << expert_to_idxes;

  std::vector<int32_t> sorted_token_ids;
  std::vector<int32_t> experts_ids;
  int32_t n_padded_tokens = 0;
  for (int e_idx = 0; e_idx < n_experts; ++e_idx) {
    // flatten indices for each expert, sorted by token id
    const auto& idxes = expert_to_idxes[e_idx];
    // LOG(ERROR) << "expert_id: " << e_idx << ", idxes: " << idxes;
    if (idxes.empty()) {
      continue;
    }
    const auto count = idxes.size();
    const auto n_blocks = cute::ceil_div(count, block_size);
    n_padded_tokens += (n_blocks * block_size);
    // fill flatten indices for each block
    for (int b_idx = 0; b_idx < n_blocks; ++b_idx) {
      // expert id for each block
      experts_ids.push_back(e_idx);
      for (int offset = 0; offset < block_size; ++offset) {
        auto idx = (b_idx * block_size) + offset;
        if (idx < count) {
          // fill flatten indices
          sorted_token_ids.push_back(idxes[idx]);
        } else {
          // fill padding
          sorted_token_ids.push_back(n_flatten_tokens);
        }
      }
    }
  }

  // construct tensor and return
  const auto options = topk_ids.options();
  return {torch::tensor(sorted_token_ids, options),
          torch::tensor(experts_ids, options),
          torch::tensor({n_padded_tokens}, options)};
}

}  // namespace

class AlignBlockTest
    : public ::testing::TestWithParam<std::tuple<torch::ScalarType /*q_dtype*/,
                                                 int64_t /*n_tokens*/,
                                                 int64_t /*dim*/,
                                                 int64_t /*n_experts*/,
                                                 int64_t /*topk*/,
                                                 int64_t /*block_size*/>> {
 public:
  void SetUp() override {
    // Set random seed for test stability
    torch::manual_seed(0);
  }
};

TEST_P(AlignBlockTest, AlignBlock) {
  const auto [dtype, n_tokens, dim, n_experts, topk, block_size] = GetParam();

  const auto options = torch::dtype(dtype).device(torch::kCUDA);
  const auto options_int32 = options.dtype(torch::kInt32);
  const auto gating_logit = torch::randn({n_tokens, n_experts}, options);

  auto [topk_weights, topk_ids] = gating_logit.topk(topk, /*dim=*/-1);
  // LOG(ERROR) << "block_size: " << block_size;
  // LOG(ERROR) << "topk_ids: " << topk_ids;

  // auto probs = weights.softmax(/*dim=*/-1);
  auto [sorted_token_ids_ref, experts_ids_ref, n_padded_tokens_ref] =
      permute_align_block_ref(
          topk_ids.to(torch::kInt32), n_experts, block_size);
  // LOG(ERROR) << "sorted_token_ids_ref: " << sorted_token_ids_ref;
  // LOG(ERROR) << "experts_ids_ref: " << experts_ids_ref;
  // LOG(ERROR) << "n_padded_tokens_ref: " << n_padded_tokens_ref;
  // EXPECT_TRUE(false);

  // allocate tensors
  // at most each expert has up to block_size - 1 padding tokens
  const int64_t n_flatten_tokens = n_tokens * topk;
  const int64_t max_padded_tokens =
      n_flatten_tokens + (n_experts * (block_size - 1));
  auto sorted_token_ids = torch::zeros({max_padded_tokens}, options_int32);
  // fill padding token id
  sorted_token_ids.fill_(n_flatten_tokens);
  // max number of blocks
  const int64_t max_blocks = cute::ceil_div(max_padded_tokens, block_size);
  auto experts_ids = torch::zeros({max_blocks}, options_int32);
  auto n_padded_tokens = torch::zeros({1}, options_int32);
  auto cu_sum = torch::zeros({n_experts + 1}, options_int32);
  kernel::moe::permute_align_block(topk_ids.to(torch::kInt32),
                                   n_experts,
                                   block_size,
                                   sorted_token_ids,
                                   experts_ids,
                                   n_padded_tokens,
                                   cu_sum);
  LOG(ERROR) << "block_size: " << block_size;
  LOG(ERROR) << "topk_ids: " << topk_ids;
  LOG(ERROR) << "sorted_token_ids: " << sorted_token_ids;
  LOG(ERROR) << "experts_ids: " << experts_ids;
  LOG(ERROR) << "n_padded_tokens: " << n_padded_tokens;

  LOG(ERROR) << "sorted_token_ids_ref: " << sorted_token_ids_ref;
  LOG(ERROR) << "experts_ids_ref: " << experts_ids_ref;
  LOG(ERROR) << "n_padded_tokens_ref: " << n_padded_tokens_ref;

  EXPECT_TRUE(prefix_equal(sorted_token_ids, sorted_token_ids_ref));
  EXPECT_TRUE(prefix_equal(experts_ids, experts_ids_ref));
  EXPECT_TRUE(torch::equal(n_padded_tokens, n_padded_tokens_ref));
}

INSTANTIATE_TEST_SUITE_P(
    Moe,
    AlignBlockTest,
    ::testing::Combine(::testing::Values(torch::kFloat),  // dtype
                       ::testing::Values(2),              // n_tokens
                       ::testing::Values(16),             // dim
                       ::testing::Values(64),             // n_experts
                       ::testing::Values(64),             // topk
                       ::testing::Values(2)               // block_size
                       ));

}  // namespace llm
