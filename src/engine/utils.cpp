#include "utils.h"

#include <torch/torch.h>
#include <torch/types.h>

#include <vector>

#include "common/logging.h"
#include "models/input_parameters.h"

namespace llm {
namespace {

std::vector<int32_t> cache_slots(const Sequence& sequence, int32_t block_size) {
  const size_t num_tokens = sequence.num_tokens();
  const auto& blocks = sequence.blocks();
  std::vector<int32_t> slots;
  slots.reserve(num_tokens);
  for (const auto& block_id : blocks) {
    for (int32_t i = 0; i < block_size; ++i) {
      slots.push_back(block_id * block_size + i);
      if (slots.size() == num_tokens) {
        break;
      }
    }
  }
  return slots;
}

int32_t last_slot_id(const Sequence& sequence, int32_t block_size) {
  const size_t num_tokens = sequence.num_tokens();
  const auto& blocks = sequence.blocks();

  const int32_t block_offset =
      static_cast<int32_t>(num_tokens - 1) % block_size;
  const int32_t last_block_id = blocks.back();
  return last_block_id * block_size + block_offset;
}

bool has_enough_cache_slots(const Sequence& sequence, int32_t block_size) {
  const size_t num_tokens = sequence.num_tokens();
  const size_t num_blocks = sequence.num_blocks();
  return num_tokens <= num_blocks * block_size;
}

}  // namespace

void Utils::prepare_inputs(const std::vector<Sequence*>& batch,
                           int32_t block_size,
                           torch::Tensor* flatten_token_ids,
                           torch::Tensor* flatten_positions,
                           torch::Tensor* seq_idxes,
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

  // track the sequence indices in the batch
  // from original sequence index to the new index in the batch
  std::vector<int32_t> seq_idxes_vec(batch.size());

  // process prefill requests
  int32_t num_prompt_tokens = 0;
  int32_t max_seq_len = 0;
  std::vector<int32_t> cu_seq_lens = {0};
  std::vector<int32_t> slot_ids;
  for (int32_t i = 0; i < static_cast<int32_t>(batch.size()); ++i) {
    const auto* sequence = batch[i];
    GCHECK(!sequence->is_finished());
    GCHECK(has_enough_cache_slots(*sequence, block_size));

    if (!sequence->is_prefill()) {
      continue;
    }
    seq_idxes_vec[i] = static_cast<int32_t>(token_ids_vec.size());

    const auto& seq_token_ids = sequence->token_ids();
    const int32_t num_tokens = static_cast<int32_t>(seq_token_ids.size());
    for (int32_t i = 0; i < num_tokens; ++i) {
      flatten_tokens_vec.push_back(seq_token_ids[i]);
      flatten_positions_vec.push_back(i);
    }

    auto& ids = token_ids_vec.emplace_back();
    auto& counts = token_counts_vec.emplace_back();

    const auto& seq_token_counts = sequence->token_to_count_map();
    const auto unique_tokens = seq_token_counts.size();
    ids.reserve(unique_tokens);
    counts.reserve(unique_tokens);
    for (const auto& [token_id, count] : seq_token_counts) {
      ids.push_back(token_id);
      counts.push_back(count);
    }
    token_ids_lens_vec.push_back(static_cast<int32_t>(unique_tokens));
    max_unique_tokens = std::max(max_unique_tokens, unique_tokens);

    num_prompt_tokens += num_tokens;
    max_seq_len = std::max(max_seq_len, num_tokens);
    cu_seq_lens.push_back(num_prompt_tokens);
    sampling_params->add(sequence->sampling_param());
    last_token_idxes.push_back(
        static_cast<int32_t>(flatten_tokens_vec.size() - 1));

    // assign slot ids for each token
    const auto slots = cache_slots(*sequence, block_size);
    slot_ids.insert(slot_ids.end(), slots.begin(), slots.end());
  }

  // process decode requests
  int32_t max_context_len = 0;
  std::vector<int32_t> context_lens;
  std::vector<std::vector<int32_t>> block_tables;
  int32_t max_block_table_len = 0;
  for (int32_t i = 0; i < static_cast<int32_t>(batch.size()); ++i) {
    const auto* sequence = batch[i];
    if (sequence->is_prefill()) {
      continue;
    }
    seq_idxes_vec[i] = static_cast<int32_t>(token_ids_vec.size());

    const auto& seq_token_ids = sequence->token_ids();
    const int32_t num_tokens = static_cast<int32_t>(seq_token_ids.size());
    flatten_tokens_vec.push_back(seq_token_ids.back());
    flatten_positions_vec.push_back(num_tokens - 1);

    auto& ids = token_ids_vec.emplace_back();
    auto& counts = token_counts_vec.emplace_back();

    const auto& seq_token_counts = sequence->token_to_count_map();
    const auto unique_tokens = seq_token_counts.size();
    ids.reserve(unique_tokens);
    counts.reserve(unique_tokens);
    for (const auto& [token_id, count] : seq_token_counts) {
      ids.push_back(token_id);
      counts.push_back(count);
    }
    token_ids_lens_vec.push_back(static_cast<int32_t>(unique_tokens));
    max_unique_tokens = std::max(max_unique_tokens, unique_tokens);

    context_lens.push_back(num_tokens);
    max_context_len = std::max(max_context_len, num_tokens);
    sampling_params->add(sequence->sampling_param());
    last_token_idxes.push_back(
        static_cast<int32_t>(flatten_tokens_vec.size() - 1));
    const auto& seq_blocks = sequence->blocks();
    block_tables.push_back(seq_blocks);
    max_block_table_len =
        std::max(max_block_table_len, static_cast<int32_t>(seq_blocks.size()));

    slot_ids.push_back(last_slot_id(*sequence, block_size));
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
    // padding token ids to the same length
    ids.resize(max_unique_tokens, /*pad_id=*/0);
    token_ids_tensor.index_put_({i, Slice()},
                                torch::tensor(ids, torch::kInt64));

    auto& counts = token_counts_vec[i];
    counts.resize(max_unique_tokens, /*pad_id=*/0);
    token_counts_tensor.index_put_({i, Slice()},
                                   torch::tensor(counts, torch::kInt));
  }
  auto block_tables_tensor = torch::empty(
      {static_cast<int64_t>(block_tables.size()), max_block_table_len},
      torch::kInt);
  for (int64_t i = 0; i < block_tables.size(); ++i) {
    auto& block_table = block_tables[i];
    block_table.resize(max_block_table_len, /*pad_id=*/0);
    block_tables_tensor.index_put_({i, Slice()},
                                   torch::tensor(block_table, torch::kInt));
  }

  *flatten_token_ids = torch::tensor(flatten_tokens_vec, torch::kInt);
  *flatten_positions = torch::tensor(flatten_positions_vec, torch::kInt);
  *seq_idxes = torch::tensor(seq_idxes_vec, torch::kInt);

  input_params->num_prompt_tokens = num_prompt_tokens;
  input_params->max_seq_len = max_seq_len;
  input_params->max_context_len = max_context_len;
  input_params->cu_seq_lens = torch::tensor(cu_seq_lens, torch::kInt);
  input_params->slot_ids = torch::tensor(slot_ids, torch::kInt);
  input_params->block_tables = block_tables_tensor;
  input_params->context_lens = torch::tensor(context_lens, torch::kInt);
  input_params->last_token_idxes = torch::tensor(last_token_idxes, torch::kInt);
  input_params->token_ids = token_ids_tensor;
  input_params->token_counts = token_counts_tensor;
  input_params->token_ids_lens = torch::tensor(token_ids_lens_vec, torch::kInt);
}

void Utils::prepare_validate_inputs(const std::vector<Sequence*>& batch,
                                    int32_t block_size,
                                    torch::Tensor* flatten_token_ids,
                                    torch::Tensor* flatten_positions,
                                    torch::Tensor* seq_idxes,
                                    InputParameters* input_params,
                                    SamplingParameters* sampling_params) {
  std::vector<int32_t> flatten_tokens_vec;
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
    GCHECK(has_enough_cache_slots(*sequence, block_size));

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

  for(int64_t i = 0; i < block_tables.size(); ++i) {
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
  input_params->max_context_len = 0;
  input_params->cu_seq_lens = torch::tensor(cu_seq_lens, torch::kInt);
  input_params->slot_ids = torch::tensor(slot_ids, torch::kInt);
  input_params->block_tables = block_tables_tensor;
  input_params->context_lens = torch::tensor({}, torch::kInt);
  input_params->last_token_idxes = torch::tensor(last_token_idxes, torch::kInt);
  input_params->token_ids = token_ids_tensor;
  input_params->token_counts = token_counts_tensor;
  input_params->token_ids_lens = torch::tensor(token_ids_lens_vec, torch::kInt);
}

}  // namespace llm
