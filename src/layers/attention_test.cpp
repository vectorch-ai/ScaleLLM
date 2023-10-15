#include "attention.h"

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <memory>
#include <thread>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/HashStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

namespace llm {

TEST(AttentionTest, VarlenMaskedSelfAttention) {
  const int64_t num_tokens = 400;
  const int64_t n_heads = 40;
  const int64_t n_kv_heads = 20;
  const int64_t head_dim = 250;
  const int32_t max_seq_len = num_tokens;

  torch::Device device(torch::kCUDA);
  torch::ScalarType dtype(torch::kHalf);

  torch::Tensor query = torch::rand({num_tokens, n_heads, head_dim},
                                    torch::dtype(dtype).device(device));
  torch::Tensor key = torch::rand({num_tokens, n_kv_heads, head_dim},
                                  torch::dtype(dtype).device(device));
  torch::Tensor value = torch::rand({num_tokens, n_kv_heads, head_dim},
                                    torch::dtype(dtype).device(device));

  torch::Tensor alibi_slopes =
      torch::rand({n_heads}, torch::dtype(torch::kFloat32).device(device));
  torch::Tensor cu_seq_lens = torch::tensor(
      {0, max_seq_len}, torch::dtype(torch::kInt32).device(device));
  torch::Tensor none_tensor;

  torch::Tensor output = torch::empty_like(query);
  varlen_masked_self_attention_slow(query,
                                    key,
                                    value,
                                    alibi_slopes,
                                    cu_seq_lens,
                                    max_seq_len,
                                    /*scale=*/1.0,
                                    output);

  torch::Tensor output_cuda = torch::empty_like(query);
  varlen_masked_self_attention_cuda(query,
                                    key,
                                    value,
                                    alibi_slopes,
                                    cu_seq_lens,
                                    max_seq_len,
                                    /*scale=*/1.0,
                                    output_cuda);
  EXPECT_TRUE(
      torch::allclose(output, output_cuda, /*rtol=*/1e-2, /*atol=*/1e-2));
}

}  // namespace llm
