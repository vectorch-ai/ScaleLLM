#include "pack_utils.h"

#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>
#include <sys/types.h>
#include <torch/torch.h>

namespace llm {

TEST(PackUtilsTest, Basic) {
  torch::ScalarType dtype(torch::kInt32);
  torch::Device device(torch::kCPU);
  auto options = torch::dtype(dtype).device(device);

  const int64_t size_m = 300;
  const int64_t size_n = 400;
  const int64_t num_bits = 8;

  torch::Tensor qweight = torch::randn({size_m, size_n}).to(options);
  // clamp to [0, 255]
  const int32_t clamp_max = (1 << num_bits) - 1;
  qweight = torch::clamp(qweight, 0, clamp_max);

  torch::Tensor packed_qweight = pack_utils::pack_cols(qweight, num_bits);

  torch::Tensor unpacked_qweight =
      pack_utils::unpack_cols(packed_qweight, num_bits);
  EXPECT_TRUE(unpacked_qweight.equal(qweight));
}

}  // namespace llm
