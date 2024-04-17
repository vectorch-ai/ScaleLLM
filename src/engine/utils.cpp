#include "utils.h"

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <vector>

#include "models/parameters.h"

namespace llm {

// Only support all decode sequences
void Utils::prepare_capture_inputs(int64_t max_seq_len,
                                   int64_t batch_size,
                                   torch::Tensor* flatten_token_ids,
                                   torch::Tensor* flatten_positions,
                                   InputParameters* input_params) {
  // create flatten tensor with shape [batch_size], only support decode.
  *flatten_token_ids = torch::ones({batch_size}, torch::kInt32);
  *flatten_positions = torch::ones({batch_size}, torch::kInt32);
  torch::Tensor cu_seq_lens = torch::range(
      /*start=*/0, /*end=*/batch_size + 1, /*step=*/1, torch::kInt32);

  InputParameters params;
  input_params->empty_kv_cache = false;
  input_params->num_sequences = batch_size;
  input_params->q_max_seq_len = 1;
  input_params->kv_max_seq_len = max_seq_len;
  input_params->q_cu_seq_lens = cu_seq_lens;
  input_params->kv_cu_seq_lens = cu_seq_lens;

  input_params->new_cache_slots = torch::zeros({batch_size}, torch::kInt);
  input_params->block_tables =
      torch::zeros({batch_size, max_seq_len}, torch::kInt);
}

void Utils::prepare_profile_inputs(int64_t max_num_tokens,
                                   int64_t max_num_seqs,
                                   torch::Tensor* flatten_token_ids,
                                   torch::Tensor* flatten_positions,
                                   InputParameters* input_params) {
  const int64_t max_seq_len = max_num_tokens / max_num_seqs;
  std::vector<int32_t> positions;
  std::vector<int32_t> cu_lens = {0};
  std::vector<int32_t> last_token_idxes;
  for (int64_t i = 0; i < max_num_seqs; ++i) {
    cu_lens.push_back(cu_lens.back() + max_seq_len);
    for (int64_t j = 0; j < max_seq_len; ++j) {
      positions.push_back(j);
    }
    last_token_idxes.push_back(positions.size() - 1);
  }

  // dummy tensor with shape [max_batch_size * max_seq_len]
  *flatten_token_ids = torch::ones({max_num_seqs * max_seq_len}, torch::kInt32);
  *flatten_positions = torch::tensor(positions, torch::kInt32);
  torch::Tensor cu_seq_lens = torch::tensor(cu_lens, torch::kInt32);

  InputParameters params;
  input_params->empty_kv_cache = true;
  input_params->num_sequences = 1;
  input_params->q_max_seq_len = max_seq_len;
  input_params->kv_max_seq_len = max_seq_len;
  input_params->q_cu_seq_lens = cu_seq_lens;
  input_params->kv_cu_seq_lens = cu_seq_lens;

  // following parameters can be empty since we don't use kv-cache
  // input_params->new_cache_slots = torch::empty({0}, torch::kInt);
  // input_params->block_tables = torch::empty({0, 0}, torch::kInt);
}

}  // namespace llm
