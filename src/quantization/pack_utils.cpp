#include "pack_utils.h"

#include <glog/logging.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

namespace llm::pack_utils {

// returns int32 packed qweight: (m, n) -> (m, n/pack_factor)
torch::Tensor pack_cols(const torch::Tensor& qweight,  // (m, n)
                        int64_t num_bits) {
  CHECK_EQ(qweight.dim(), 2);
  CHECK_EQ(qweight.scalar_type(), torch::kInt);
  CHECK(qweight.is_contiguous());

  const int64_t size_m = qweight.size(0);
  const int64_t size_n = qweight.size(1);
  const int64_t pack_factor = 32 / num_bits;
  CHECK_EQ(size_n % pack_factor, 0);

  torch::Tensor qweight_cpu = qweight.cpu().contiguous();
  const uint32_t* qweight_ptr = qweight_cpu.const_data_ptr<uint32_t>();

  torch::Tensor packed_qweight =
      torch::empty({size_m, size_n / pack_factor},
                   torch::dtype(torch::kUInt32).device(torch::kCPU));
  uint32_t* packed_qweight_ptr = packed_qweight.mutable_data_ptr<uint32_t>();

  const uint32_t mask = (1 << num_bits) - 1;
  for (int64_t i = 0; i < size_m; ++i) {
    for (int64_t j = 0; j < size_n; j += pack_factor) {
      uint32_t packed_val = 0;
      for (int64_t k = 0; k < pack_factor; ++k) {
        packed_val |= (qweight[i][j + k] & mask) << (num_bits * k);
      }
      packed_qweight[i][j / pack_factor] = packed_val;
    }
  }
  return packed_qweight.to(qweight);
}

// returns int32 qweight: (m, n/pack_factor) -> (m, n)
torch::Tensor unpack_cols(
    const torch::Tensor& packed_qweight,  // (m, n/pack_factor)
    int64_t num_bits) {
  return packed_qweight;
}

}  // namespace llm::pack_utils