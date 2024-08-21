#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

namespace llm::pack_utils {

// returns int32 packed qweight: (m, n) -> (m, n/pack_factor)
torch::Tensor pack_cols(const torch::Tensor& qweight,  // (m, n)
                        int64_t num_bits);

// returns int32 qweight: (m, n/pack_factor) -> (m, n)
torch::Tensor unpack_cols(
    const torch::Tensor& packed_qweight,  // (m, n/pack_factor)
    int64_t num_bits);

}  // namespace llm::pack_utils