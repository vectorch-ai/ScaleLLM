#include "utils.h"

#include <torch/torch.h>

#include <vector>

#include "models/parameters.h"
#include "request/request.h"

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
                           torch::Tensor* input_token_ids,
                           torch::Tensor* input_positions,
                           torch::Tensor* seq_indices,
                           InputParameters* input_params,
                           SamplingParameters* sampling_params) {
  std::vector<int32_t> flat_tokens;
  std::vector<int32_t> flat_positions;
  std::vector<int32_t> sample_idx;
  std::vector<std::vector<int32_t>> token_ids;
  int32_t max_num_tokens = 0;

  // track the sequence indices in the batch
  // used to map the output back to the original sequence order
  std::vector<int32_t> seq_indices_vec;
  // from original sequence index to the index in the new order
  seq_indices_vec.resize(batch.size());

  // process prefill requests
  int32_t num_prompt_tokens = 0;
  int32_t max_seq_len = 0;
  std::vector<int32_t> cu_seq_lens = {0};
  std::vector<int32_t> slot_ids;
  for (int32_t i = 0; i < static_cast<int32_t>(batch.size()); ++i) {
    const auto* sequence = batch[i];
    CHECK(!sequence->is_finished());
    CHECK(has_enough_cache_slots(*sequence, block_size));

    if (!sequence->is_prefill()) {
      continue;
    }
    seq_indices_vec[i] = static_cast<int32_t>(token_ids.size());

    const auto& seq_token_ids = sequence->token_ids();
    const int32_t num_tokens = static_cast<int32_t>(seq_token_ids.size());
    max_num_tokens = std::max(max_num_tokens, num_tokens);
    token_ids.push_back(seq_token_ids);
    for (int32_t i = 0; i < num_tokens; ++i) {
      flat_tokens.push_back(seq_token_ids[i]);
      flat_positions.push_back(i);
    }

    num_prompt_tokens += num_tokens;
    max_seq_len = std::max(max_seq_len, num_tokens);
    cu_seq_lens.push_back(num_prompt_tokens);
    sampling_params->add(sequence->sampling_param());
    // only sample against the last token of the prompt
    sample_idx.push_back(static_cast<int32_t>(flat_tokens.size() - 1));

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
    seq_indices_vec[i] = static_cast<int32_t>(token_ids.size());

    const auto& seq_token_ids = sequence->token_ids();
    const int32_t num_tokens = static_cast<int32_t>(seq_token_ids.size());
    max_num_tokens = std::max(max_num_tokens, num_tokens);
    token_ids.push_back(seq_token_ids);
    flat_tokens.push_back(seq_token_ids.back());
    flat_positions.push_back(num_tokens - 1);

    context_lens.push_back(num_tokens);
    max_context_len = std::max(max_context_len, num_tokens);
    sampling_params->add(sequence->sampling_param());
    sample_idx.push_back(static_cast<int32_t>(flat_tokens.size() - 1));
    const auto& seq_blocks = sequence->blocks();
    block_tables.push_back(seq_blocks);
    max_block_table_len =
        std::max(max_block_table_len, static_cast<int32_t>(seq_blocks.size()));

    slot_ids.push_back(last_slot_id(*sequence, block_size));
  }

  using torch::indexing::Slice;
  // padding token ids to the same length
  auto token_ids_tensor = torch::empty(
      {static_cast<int64_t>(token_ids.size()), max_num_tokens}, torch::kLong);
  for (int64_t i = 0; i < token_ids.size(); ++i) {
    auto& ids = token_ids[i];
    ids.resize(max_num_tokens, /*pad_id=*/0);
    token_ids_tensor.index_put_({i, Slice()}, torch::tensor(ids, torch::kLong));
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

  *input_token_ids = torch::tensor(flat_tokens, torch::kInt);
  *input_positions = torch::tensor(flat_positions, torch::kInt);
  *seq_indices = torch::tensor(seq_indices_vec, torch::kInt);

  input_params->num_prompt_tokens = num_prompt_tokens;
  input_params->max_seq_len = max_seq_len;
  input_params->max_context_len = max_context_len;
  input_params->cu_seq_lens = torch::tensor(cu_seq_lens, torch::kInt);
  input_params->slot_ids = torch::tensor(slot_ids, torch::kInt);
  input_params->block_tables = block_tables_tensor;
  input_params->context_lens = torch::tensor(context_lens, torch::kInt);
  input_params->last_token_indicies = torch::tensor(sample_idx, torch::kInt);
  input_params->token_ids = token_ids_tensor;
}

}  // namespace llm
