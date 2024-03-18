#include "utils.h"

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <vector>

#include "models/input_parameters.h"

namespace llm {
namespace {

std::vector<int32_t> cache_slots_for_pos(const std::vector<Block>& blocks,
                                         int32_t block_size,
                                         int32_t start,
                                         int32_t end) {
  std::vector<int32_t> slots;
  slots.reserve(end - start);
  for (int32_t i = start; i < end; ++i) {
    const int32_t block_id = blocks[i / block_size].id();
    const int32_t block_offset = i % block_size;
    slots.push_back(block_id * block_size + block_offset);
  }
  return slots;
}

bool has_enough_cache_slots(const Sequence& sequence, int32_t block_size) {
  const size_t num_tokens = sequence.num_tokens();
  const size_t num_blocks = sequence.num_blocks();
  return num_tokens <= num_blocks * block_size;
}

template <typename T>
torch::Tensor create_2d_tensor(std::vector<std::vector<T>>& vec,
                               size_t max_col_size,
                               torch::ScalarType dtype,
                               T pad_value = 0) {
  const int64_t n_rows = vec.size();
  auto tensor =
      torch::empty({n_rows, static_cast<int64_t>(max_col_size)}, dtype);
  for (int64_t i = 0; i < n_rows; ++i) {
    vec[i].resize(max_col_size, pad_value);
    tensor[i] = torch::tensor(vec[i], dtype);
  }
  return tensor;
}

}  // namespace

// Only support all decode sequences
void Utils::prepare_capture_inputs(int64_t max_seq_len,
                                   int64_t batch_size,
                                   torch::Tensor* flatten_token_ids,
                                   torch::Tensor* flatten_positions,
                                   InputParameters* input_params) {
  // create position
  std::vector<int32_t> positions;
  std::vector<int32_t> cu_lens = {0};
  std::vector<int32_t> last_token_idxes;
  for (int64_t i = 0; i < batch_size; ++i) {
    cu_lens.push_back(cu_lens.back() + 1);
    positions.push_back(0);
    last_token_idxes.push_back(positions.size() - 1);
  }

  // create flatten tensor with shape [batch_size], only support decode.
  *flatten_token_ids = torch::ones({batch_size}, torch::kInt32);
  *flatten_positions = torch::tensor(positions, torch::kInt32);
  torch::Tensor cu_seq_lens = torch::tensor(cu_lens, torch::kInt32);

  InputParameters params;
  input_params->all_prefill_sequences = false;
  input_params->num_sequences = batch_size;
  input_params->q_max_seq_len = 1;
  input_params->kv_max_seq_len = max_seq_len;
  input_params->q_cu_seq_lens = cu_seq_lens;
  input_params->kv_cu_seq_lens = cu_seq_lens;

  input_params->new_cache_slots = torch::ones({batch_size}, torch::kInt);
  input_params->block_tables =
      torch::empty({batch_size, max_seq_len}, torch::kInt);
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
  input_params->all_prefill_sequences = true;
  input_params->num_sequences = 1;
  input_params->q_max_seq_len = max_seq_len;
  input_params->kv_max_seq_len = max_seq_len;
  input_params->q_cu_seq_lens = cu_seq_lens;
  input_params->kv_cu_seq_lens = cu_seq_lens;

  // following parameters can be empty since we don't use kv-cache
  // input_params->new_cache_slots = torch::empty({0}, torch::kInt);
  // input_params->block_tables = torch::empty({0, 0}, torch::kInt);
}

void Utils::prepare_inputs(const std::vector<Sequence*>& batch,
                           int32_t block_size,
                           torch::Tensor* flatten_token_ids,
                           torch::Tensor* flatten_positions,
                           InputParameters* input_params,
                           SamplingParameters* sampling_params) {
  // flatten the token ids and positions
  std::vector<int32_t> flatten_tokens_vec;
  std::vector<int32_t> flatten_positions_vec;
  // track the last token index in the flattened tokens
  std::vector<int32_t> last_token_idxes;

  // track the token ids and counts in the batch
  std::vector<std::vector<int64_t>> token_ids_vec;
  std::vector<int32_t> token_ids_lens_vec;
  std::vector<std::vector<int32_t>> token_counts_vec;
  size_t max_unique_tokens = 0;

  // process prefill requests
  bool all_prefill_sequences = true;
  int32_t max_seq_len = 0;
  int32_t q_max_seq_len = 0;
  std::vector<int32_t> cu_seq_lens = {0};
  std::vector<int32_t> q_cu_seq_lens = {0};
  // slot ids for new token
  std::vector<int32_t> new_token_slot_ids;
  std::vector<std::vector<int32_t>> block_tables_vec;
  int32_t max_block_table_len = 0;
  const int32_t num_sequences = static_cast<int32_t>(batch.size());
  for (int32_t i = 0; i < num_sequences; ++i) {
    const auto* sequence = batch[i];
    CHECK(!sequence->is_finished());
    CHECK(has_enough_cache_slots(*sequence, block_size));

    all_prefill_sequences &= sequence->is_prefill();

    const auto& seq_token_ids = sequence->token_ids();
    const int32_t seq_len = static_cast<int32_t>(seq_token_ids.size());
    const int32_t kvcache_seq_len = sequence->num_tokens_in_cache();
    const int32_t q_seq_len = seq_len - kvcache_seq_len;
    // pack the token ids and positions into one-dimensional tensors
    for (int32_t i = kvcache_seq_len; i < seq_len; ++i) {
      flatten_tokens_vec.push_back(seq_token_ids[i]);
      flatten_positions_vec.push_back(static_cast<int32_t>(i));
    }
    last_token_idxes.push_back(
        static_cast<int32_t>(flatten_tokens_vec.size() - 1));

    // add token id and count for each sequence
    const auto& seq_token_counts = sequence->token_to_count_map();
    const auto unique_tokens = seq_token_counts.size();

    auto& ids = token_ids_vec.emplace_back();
    auto& counts = token_counts_vec.emplace_back();
    ids.reserve(unique_tokens);
    counts.reserve(unique_tokens);
    for (const auto& [token_id, count] : seq_token_counts) {
      ids.push_back(token_id);
      counts.push_back(count);
    }
    token_ids_lens_vec.push_back(static_cast<int32_t>(unique_tokens));
    max_unique_tokens = std::max(max_unique_tokens, unique_tokens);

    max_seq_len = std::max(max_seq_len, seq_len);
    q_max_seq_len = std::max(q_max_seq_len, q_seq_len);
    cu_seq_lens.push_back(cu_seq_lens.back() + seq_len);
    q_cu_seq_lens.push_back(q_cu_seq_lens.back() + q_seq_len);

    // add sampling parameters
    sampling_params->add(sequence->sampling_param());

    // assign slot ids for new tokens [n_tokens_in_kvcache, total_tokens)
    const auto& blocks = sequence->blocks();
    const auto slot_ids =
        cache_slots_for_pos(blocks, block_size, kvcache_seq_len, seq_len);
    new_token_slot_ids.insert(
        new_token_slot_ids.end(), slot_ids.begin(), slot_ids.end());

    // construct block ids for each sequence
    std::vector<int32_t> block_ids;
    block_ids.reserve(blocks.size());
    for (const auto& block : blocks) {
      block_ids.push_back(block.id());
    }
    block_tables_vec.push_back(block_ids);

    max_block_table_len =
        std::max(max_block_table_len, static_cast<int32_t>(blocks.size()));
  }

  using torch::indexing::Slice;
  // construct two-dimensional tensors for token ids and counts
  auto token_ids = create_2d_tensor(token_ids_vec,
                                    max_unique_tokens,
                                    torch::kInt64,
                                    /*pad_value=*/int64_t(0));
  auto token_counts = create_2d_tensor(
      token_counts_vec, max_unique_tokens, torch::kInt, /*pad_value=*/0);

  auto block_tables = create_2d_tensor(
      block_tables_vec, max_block_table_len, torch::kInt, /*pad_value=*/0);

  *flatten_token_ids = torch::tensor(flatten_tokens_vec, torch::kInt);
  *flatten_positions = torch::tensor(flatten_positions_vec, torch::kInt);

  input_params->all_prefill_sequences = all_prefill_sequences;
  input_params->num_sequences = num_sequences;
  input_params->kv_max_seq_len = max_seq_len;
  input_params->q_max_seq_len = q_max_seq_len;
  input_params->kv_cu_seq_lens = torch::tensor(cu_seq_lens, torch::kInt);
  input_params->q_cu_seq_lens = torch::tensor(q_cu_seq_lens, torch::kInt);
  input_params->new_cache_slots =
      torch::tensor(new_token_slot_ids, torch::kInt);
  input_params->block_tables = block_tables;

  sampling_params->last_token_idxes =
      torch::tensor(last_token_idxes, torch::kInt);
  sampling_params->token_ids = token_ids;
  sampling_params->token_counts = token_counts;
  sampling_params->token_ids_lens =
      torch::tensor(token_ids_lens_vec, torch::kInt);
}

void Utils::prepare_validate_inputs(const std::vector<Sequence*>& batch,
                                    int32_t block_size,
                                    torch::Tensor* flatten_token_ids,
                                    torch::Tensor* flatten_positions,
                                    torch::Tensor* seq_idxes,
                                    InputParameters* input_params,
                                    SamplingParameters* sampling_params) {
  // TODO(michael): update this function to use the new input parameters
  /*std::vector<int32_t> flatten_tokens_vec;
  std::vector<int32_t> flatten_positions_vec;
  std::vector<int32_t> last_token_idxes;

  std::vector<std::vector<int64_t>> token_ids_vec;
  std::vector<int32_t> token_ids_lens_vec;
  std::vector<std::vector<int32_t>> token_counts_vec;
  size_t max_unique_tokens = 0;

  std::vector<int32_t> seq_idxes_vec(batch.size());

  int32_t num_total_tokens = 0;
  int32_t max_seq_len = 0;
  std::vector<int32_t> cu_seq_lens = {0};
  std::vector<int32_t> slot_ids;

  for (int32_t i = 0; i < static_cast<int32_t>(batch.size()); ++i) {
    const auto* sequence = batch[i];
    CHECK(has_enough_cache_slots(*sequence, block_size));

    seq_idxes_vec[i] = static_cast<int32_t>(token_ids_vec.size());

    const auto& seq_token_ids = sequence->token_ids();
    const int32_t num_tokens = static_cast<int32_t>(seq_token_ids.size());
    for (int32_t i = 0; i < num_tokens; ++i) {
      flatten_tokens_vec.emplace_back(seq_token_ids[i]);
      flatten_positions_vec.emplace_back(i);
    }

    auto& ids = token_ids_vec.emplace_back();
    auto& counts = token_counts_vec.emplace_back();

    const auto& seq_token_counts = sequence->token_to_count_map();
    const auto unique_tokens = seq_token_counts.size();
    ids.reserve(unique_tokens);
    counts.reserve(unique_tokens);
    for (const auto& [token_id, count] : seq_token_counts) {
      ids.emplace_back(token_id);
      counts.emplace_back(count);
    }
    token_ids_lens_vec.emplace_back(static_cast<int32_t>(unique_tokens));
    max_unique_tokens = std::max(max_unique_tokens, unique_tokens);

    num_total_tokens += num_tokens;
    max_seq_len = std::max(max_seq_len, num_tokens);
    cu_seq_lens.emplace_back(num_total_tokens);
    sampling_params->add(sequence->sampling_param());
    last_token_idxes.emplace_back(
        static_cast<int32_t>(flatten_tokens_vec.size() - 1));

    const auto slots = cache_slots(*sequence, block_size);
    slot_ids.insert(slot_ids.end(), slots.begin(), slots.end());
  }

  using torch::indexing::Slice;
  auto token_ids_tensor =
      torch::empty({static_cast<int64_t>(token_ids_vec.size()),
                    static_cast<int64_t>(max_unique_tokens)},
                   torch::kInt64);
  auto token_counts_tensor =
      torch::empty({static_cast<int64_t>(token_counts_vec.size()),
                    static_cast<int64_t>(max_unique_tokens)},
                   torch::kInt);
  for (int64_t i = 0; i < token_ids_vec.size(); ++i) {
    auto& ids = token_ids_vec[i];
    ids.resize(max_unique_tokens, 0);
    token_ids_tensor.index_put_({i, Slice()},
                                torch::tensor(ids, torch::kInt64));

    auto& counts = token_counts_vec[i];
    counts.resize(max_unique_tokens, 0);
    token_counts_tensor.index_put_({i, Slice()},
                                   torch::tensor(counts, torch::kInt));
  }

  std::vector<std::vector<int32_t>> block_tables;
  int32_t max_block_table_len = 0;
  auto block_tables_tensor = torch::empty(
      {static_cast<int64_t>(block_tables.size()), max_block_table_len},
      torch::kInt);

  for (int64_t i = 0; i < block_tables.size(); ++i) {
    auto& block_table = block_tables[i];
    block_table.resize(max_block_table_len, 0);
    block_tables_tensor.index_put_({i, Slice()},
                                   torch::tensor(block_table, torch::kInt));
  }

  *flatten_token_ids = torch::tensor(flatten_tokens_vec, torch::kInt);
  *flatten_positions = torch::tensor(flatten_positions_vec, torch::kInt);
  *seq_idxes = torch::tensor(seq_idxes_vec, torch::kInt);

  input_params->num_prompt_tokens = num_total_tokens;
  input_params->max_seq_len = max_seq_len;
  // input_params->max_context_len = 0;
  input_params->cu_seq_lens = torch::tensor(cu_seq_lens, torch::kInt);
  input_params->slot_ids = torch::tensor(slot_ids, torch::kInt);
  input_params->block_tables = block_tables_tensor;
  // input_params->context_lens = torch::tensor({}, torch::kInt);
  input_params->last_token_idxes = torch::tensor(last_token_idxes, torch::kInt);
  input_params->token_ids = token_ids_tensor;
  input_params->token_counts = token_counts_tensor;
  input_params->token_ids_lens = torch::tensor(token_ids_lens_vec, torch::kInt);
  */
}

}  // namespace llm
