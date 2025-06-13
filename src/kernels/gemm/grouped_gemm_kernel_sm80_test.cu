#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>

#include "grouped_gemm_kernel_sm80.cuh"  // IWYU pragma: keep

namespace llm {

namespace {

// reference implementation
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> permute_align_block(
    torch::Tensor topk_ids,  // [n_tokens, topk]
    int64_t n_experts,
    int64_t block_size) {
  const int64_t n_tokens = topk_ids.size(0);
  const int64_t topk = topk_ids.size(1);
  const int64_t n_flatten_tokens = topk_ids.numel();

  auto topk_ids_cpu = topk_ids.cpu().contiguous();
  const int32_t* topk_ids_ptr = topk_ids_cpu.data_ptr<int32_t>();

  std::vector<std::vector<int32_t>> expert_to_idxes(n_experts);
  for (int i = 0; i < n_flatten_tokens; ++i) {
    const int32_t expert_id = topk_ids_ptr[i];
    assert(expert_id >= 0 && expert_id < n_experts);
    expert_to_idxes[expert_id].push_back(i);
  }

  std::vector<int32_t> sorted_token_idxes;
  std::vector<int32_t> expert_ids;
  int32_t n_padded_tokens = 0;
  for (int e_idx = 0; e_idx < n_experts; ++e_idx) {
    // flatten indices for each expert, sorted by token id
    const auto& idxes = expert_to_idxes[e_idx];
    if (idxes.empty()) {
      continue;
    }
    const auto count = idxes.size();
    const auto n_blocks = cute::ceil_div(count, block_size);
    n_padded_tokens += (n_blocks * block_size);
    // fill flatten indices for each block
    for (int b_idx = 0; b_idx < n_blocks; ++b_idx) {
      // expert id for each block
      expert_ids.push_back(e_idx);
      for (int offset = 0; offset < block_size; ++offset) {
        auto idx = (b_idx * block_size) + offset;
        if (idx < count) {
          // fill flatten indices
          sorted_token_idxes.push_back(idxes[idx]);
        } else {
          // fill padding
          sorted_token_idxes.push_back(n_flatten_tokens);
        }
      }
    }
  }

  // construct tensor and return
  const auto options = topk_ids.options();
  return {torch::tensor(sorted_token_idxes, options),
          torch::tensor(expert_ids, options),
          torch::tensor({n_padded_tokens}, options)};
}

torch::Tensor grouped_gemm_sm80(const torch::Tensor& a,        // (m, k)
                                const torch::Tensor& w,        // (e, n, k)
                                const torch::Tensor& topk_ids  // (m, topk)
) {
  const auto m = a.size(0);
  const auto k = a.size(1);
  const auto n = w.size(1);
  const auto n_experts = w.size(0);
  const auto topk = topk_ids.size(1);

  // construct aligned
  auto [sorted_token_idex, expert_ids, n_tokens_padded] = permute_align_block(
      topk_ids.to(torch::kInt32), n_experts, /*block_size=*/64);

  // LOG(ERROR) << "sorted_token_idex: " << sorted_token_idex;
  // LOG(ERROR) << "expert_ids: " << expert_ids;
  // LOG(ERROR) << "n_padded_tokens: " << n_tokens_padded;

  // (m * topk, n)
  auto out = torch::zeros({m * topk, n}, a.options());

  using Traits = GEMMTraitsSM80<cute::half_t, /*DTYPE*/
                                64,           /*DIM*/
                                64,           /*BLK_M*/
                                64,           /*BLK_N*/
                                64,           /*BLK_K*/
                                2>;           /*STAGES*/

  // construct params
  GEMMParams params;
  params.a_ptr = a.const_data_ptr();
  params.a_stride = make_stride(a.stride(0));
  params.b_ptr = w.const_data_ptr();
  params.b_stride = make_stride(w.stride(0), w.stride(1));
  params.c_ptr = out.mutable_data_ptr();
  params.c_stride = make_stride(out.stride(0));

  params.sorted_token_idxes_ptr = sorted_token_idex.const_data_ptr<int32_t>();
  params.expert_ids_ptr = expert_ids.const_data_ptr<int32_t>();
  params.n_tokens_padded = n_tokens_padded.const_data_ptr<int32_t>();

  params.m = m;
  params.n = n;
  params.k = k;
  params.topk = topk;

  params.m_blocks = expert_ids.size(0);
  params.n_blocks = cute::ceil_div(n, 64);

  launch_grouped_gemm_kernel_sm80<Traits>(params, nullptr);

  // (m * topk, n) => (m, topk, n)
  return out.view({m, topk, n});
}

// returns (m, topk, n)
torch::Tensor grouped_gemm_ref(const torch::Tensor& a,        // (m, k)
                               const torch::Tensor& w,        // (e, n, k)
                               const torch::Tensor& topk_ids  // (m, topk)

) {
  const auto m = a.size(0);
  const auto k = a.size(1);
  const auto n = w.size(1);
  const auto n_experts = w.size(0);
  const auto topk = topk_ids.size(1);

  // (m * topk, n)
  auto out = torch::zeros({m * topk, n}, a.options());

  // (m, k) => (m, topk, k) => (m * topk, k)
  auto a_expanded_flat =
      a.unsqueeze(/*dim=*/1).expand({-1, topk, -1}).reshape({-1, k});
  // (m, topk) => (m * topk)
  auto topk_ids_flat = topk_ids.reshape({-1});

  // process each expert
  for (int64_t e = 0; e < n_experts; ++e) {
    // 1D indices for the current expert
    auto indices = torch::nonzero(topk_ids_flat == e).squeeze();
    // select corresponding tokens
    auto a_selected = a_expanded_flat.index_select(/*dim=*/0, indices);
    // perform the GEMM operation for this expert
    auto e_out = torch::matmul(a_selected, w[e].transpose(0, 1));
    // copy the results into the output tensor
    out.index_copy_(/*dim=*/0, indices, e_out);
  }
  // (m * topk, n) => (m, topk, n)
  return out.view({m, topk, n});
}

}  // namespace

class GroupedGemmKernelTest
    : public ::testing::TestWithParam<std::tuple<torch::ScalarType /*dtype*/,
                                                 int64_t /*m*/,
                                                 int64_t /*n*/,
                                                 int64_t /*k*/,
                                                 int64_t /*n_experts*/,
                                                 int64_t /*topk*/>> {
 public:
  void SetUp() override {
    // Set random seed for test stability
    torch::manual_seed(0);
  }
};

TEST_P(GroupedGemmKernelTest, GEMM) {
  const auto [dtype, m, n, k, n_experts, topk] = GetParam();

  const auto options = torch::dtype(dtype).device(torch::kCUDA);

  // Create input tensors
  auto a = torch::randn({m, k}, options);
  auto w = torch::randn({n_experts, n, k}, options);

  // Get top-k indices
  auto logits = torch::randn({m, n_experts}, options).softmax(/*dim=*/1);
  auto [topk_weights, topk_ids] = logits.topk(topk, /*dim=*/1);

  auto ref_out = grouped_gemm_ref(a, w, topk_ids);
  // LOG(ERROR) << "ref_out: " << ref_out;
  auto out = grouped_gemm_sm80(a, w, topk_ids);

  EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-3, /*atol=*/1e-3));

  // auto max_diff = (out - ref_out).abs().max();
  // LOG(ERROR) << "Max diff: " << max_diff;
  // LOG(ERROR) << "ref_out: " << ref_out;
  // LOG(ERROR) << "out: " << out;
}

INSTANTIATE_TEST_SUITE_P(
    GEMM,
    GroupedGemmKernelTest,
    ::testing::Combine(::testing::Values(torch::kHalf),  // dtype
                       ::testing::Values(64, 128),       // m
                       ::testing::Values(64, 128),       // n
                       ::testing::Values(64, 128),       // k
                       ::testing::Values(1),             // n_experts
                       ::testing::Values(1)              // topk
                       ));

}  // namespace llm
