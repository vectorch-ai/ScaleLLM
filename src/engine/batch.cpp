#include "batch.h"

#include <torch/torch.h>

#include <vector>

#include "common/slice.h"
#include "models/parameters.h"
#include "request/sequence.h"

namespace llm {

namespace {

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

Batch::Batch(Sequence* sequence) { add(sequence); }
Batch::Batch(const std::vector<Sequence*>& sequences) { add(sequences); }

void Batch::reset(const std::vector<Sequence*>& sequences) {
  clear();
  add(sequences);
}

void Batch::add(Sequence* sequence, uint32_t token_budget) {
  CHECK(sequence != nullptr);
  CHECK(!sequence->is_finished());
  CHECK_GT(token_budget, 0);

  sequences_.push_back(sequence);
  token_budgets_.push_back(token_budget);
}

void Batch::add(const std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    add(sequence);
  }
}

void Batch::clear() {
  sequences_.clear();
  token_budgets_.clear();
}

// prepare inputs for the batch
ModelInput Batch::prepare_model_input() {
  ModelInput model_inputs;

  // flatten the token ids and positions
  std::vector<int32_t> flatten_tokens_vec;
  std::vector<int32_t> flatten_positions_vec;

  // sleceted tokens to return logits, including generated tokens and last
  // prompt token
  std::vector<int32_t> selected_token_idxes;
  // track the last token of selected tokens for sampling
  std::vector<int32_t> sample_idxes;

  // track the token ids and counts in the batch
  std::vector<std::vector<int64_t>> token_ids_vec;
  std::vector<int32_t> token_ids_lens_vec;
  std::vector<std::vector<int32_t>> token_counts_vec;
  size_t max_unique_tokens = 0;

  bool empty_kv_cache = true;
  uint32_t max_seq_len = 0;
  uint32_t q_max_seq_len = 0;
  std::vector<int32_t> cu_seq_lens = {0};
  std::vector<int32_t> q_cu_seq_lens = {0};
  // slot ids for new token
  std::vector<int32_t> new_token_slot_ids;
  std::vector<std::vector<int32_t>> block_tables_vec;
  int32_t max_block_table_len = 0;
  const int32_t num_sequences = static_cast<int32_t>(sequences_.size());
  for (int32_t i = 0; i < num_sequences; ++i) {
    auto* sequence = sequences_[i];
    empty_kv_cache = empty_kv_cache && (sequence->kv_cache_size() == 0);

    const auto token_ids = sequence->token_ids();

    const uint32_t n_tokens = token_ids.size();
    const uint32_t n_tokens_in_kv_cache = sequence->kv_cache_size();
    const uint32_t q_seq_len =
        std::min(n_tokens - n_tokens_in_kv_cache, token_budgets_[i]);

    const uint32_t seq_len = q_seq_len + n_tokens_in_kv_cache;

    // check if the sequence has enough cache slots
    CHECK_GE(sequence->kv_cache_capacity(), seq_len);

    // at least one token to process otherwise the sequence should be finished.
    CHECK(q_seq_len != 0) << "at least one token should be processed. "
                          << "n_tokens: " << n_tokens
                          << ", n_tokens_in_kv_cache: " << n_tokens_in_kv_cache
                          << ", kv_cache_capacity: "
                          << sequence->kv_cache_capacity()
                          << ", token_budget: " << token_budgets_[i];

    // update sequence length
    max_seq_len = std::max(max_seq_len, seq_len);
    q_max_seq_len = std::max(q_max_seq_len, q_seq_len);
    cu_seq_lens.push_back(cu_seq_lens.back() + seq_len);
    q_cu_seq_lens.push_back(q_cu_seq_lens.back() + q_seq_len);

    // pack the token ids and positions into one-dimensional tensors
    // and select tokens for sampling the next token
    const uint32_t n_prompt_tokens = sequence->num_prompt_tokens();
    bool has_selected_token = false;
    for (uint32_t j = n_tokens_in_kv_cache; j < seq_len; ++j) {
      flatten_tokens_vec.push_back(token_ids[j]);
      flatten_positions_vec.push_back(static_cast<int32_t>(j));

      // skip prompt tokens except the last one
      if (j + 1 < n_prompt_tokens) {
        continue;
      }
      // select tokens for sampling the next token
      selected_token_idxes.push_back(flatten_tokens_vec.size() - 1);
      has_selected_token = true;
      // sample last token in the sequence
      if (j == seq_len - 1) {
        sample_idxes.push_back(
            static_cast<int32_t>(selected_token_idxes.size() - 1));
      }
    }

    if (has_selected_token) {
      // add token id and count for sampling
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

      // add sampling parameters
      model_inputs.sampling_params.add(sequence->sampling_param());
    }

    // commit kv cache to advance kv_cache pos in sequence
    sequence->commit_kv_cache(/*size=*/q_seq_len);

    // assign slot ids for new tokens [n_tokens_in_kvcache, total_tokens)
    const auto blocks = sequence->blocks();
    const auto slot_ids =
        sequence->kv_cache_slots(n_tokens_in_kv_cache, seq_len);
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

  auto block_tables = create_2d_tensor(
      block_tables_vec, max_block_table_len, torch::kInt, /*pad_value=*/0);

  model_inputs.token_ids = torch::tensor(flatten_tokens_vec, torch::kInt);
  model_inputs.positions = torch::tensor(flatten_positions_vec, torch::kInt);

  auto& input_params = model_inputs.input_params;
  input_params.empty_kv_cache = empty_kv_cache;
  input_params.num_sequences = num_sequences;
  input_params.kv_max_seq_len = max_seq_len;
  input_params.q_max_seq_len = q_max_seq_len;
  input_params.kv_cu_seq_lens = torch::tensor(cu_seq_lens, torch::kInt);
  input_params.q_cu_seq_lens = torch::tensor(q_cu_seq_lens, torch::kInt);
  input_params.new_cache_slots = torch::tensor(new_token_slot_ids, torch::kInt);
  input_params.block_tables = block_tables;

  auto& sampling_params = model_inputs.sampling_params;
  if (!selected_token_idxes.empty()) {
    // construct two-dimensional tensors for token ids and counts
    auto token_ids = create_2d_tensor(token_ids_vec,
                                      max_unique_tokens,
                                      torch::kInt64,
                                      /*pad_value=*/int64_t(0));
    auto token_counts = create_2d_tensor(
        token_counts_vec, max_unique_tokens, torch::kInt, /*pad_value=*/0);

    sampling_params.selected_token_idxes =
        torch::tensor(selected_token_idxes, torch::kInt);
    sampling_params.token_ids = token_ids;
    sampling_params.token_counts = token_counts;
    sampling_params.token_ids_lens =
        torch::tensor(token_ids_lens_vec, torch::kInt);
  }

  if (!sample_idxes.empty()) {
    sampling_params.sample_idxes = torch::tensor(sample_idxes, torch::kInt);
  }

  return model_inputs;
}

void Batch::process_sample_output(const SampleOutput& sample_output) {
  // it is possible that the model output is empty for prefill sequences
  if (sample_output.next_tokens.defined()) {
    const auto& next_tokens = sample_output.next_tokens.cpu();
    const int64_t num_seqs = next_tokens.numel();
    int64_t output_idx = 0;
    for (auto* seq : sequences_) {
      if (seq->is_prefill_stage()) {
        // no sampling for prefill sequences
        continue;
      }
      CHECK_LT(output_idx, num_seqs);

      // add the next token to sequence
      const int32_t next_token_id =
          static_cast<int32_t>(next_tokens[output_idx++].item<int64_t>());
      seq->append_new_token_id(next_token_id);
    }
    CHECK_EQ(output_idx, num_seqs);
  }
}

void Batch::process_validate_output(const torch::Tensor& accepted_ids) {
  const auto& token_ids = accepted_ids.cpu();
  const int64_t num_seqs = accepted_ids.size(0);
  int64_t output_idx = 0;
  for (auto* seq : sequences_) {
    if (seq->is_prefill_stage()) {
      // no sampling for prefill sequences
      continue;
    }
    CHECK_LT(output_idx, num_seqs);

    const auto ids = token_ids[output_idx++];
    const Slice<int64_t> accepted_token_ids = {
        ids.data_ptr<int64_t>(), static_cast<size_t>(ids.numel())};

    // validate the draft tokens with accepted tokens
    seq->validate_token_ids(accepted_token_ids);
  }
  CHECK_EQ(output_idx, num_seqs);
}

void Batch::set_engine_type(EngineType engine_type) {
  // set engine type for all sequences in the batch
  for (auto* sequence : sequences_) {
    sequence->set_engine_type(engine_type);
  }
}

}  // namespace llm
