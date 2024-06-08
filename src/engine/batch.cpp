#include "batch.h"

#include <c10/core/DeviceType.h>
#include <torch/torch.h>

#include <vector>

#include "common/metrics.h"
#include "common/slice.h"
#include "common/tensor_helper.h"
#include "models/parameters.h"
#include "request/sequence.h"
#include "sampling/parameters.h"

DEFINE_COUNTER(num_accepted_tokens_total,
               "Total number of accepted tokens in validation");

namespace llm {

namespace {

template <typename T>
void pad_2d_vector(std::vector<std::vector<T>>& vec, T pad_value) {
  size_t max_col_size = 0;
  for (const auto& row : vec) {
    max_col_size = std::max(max_col_size, row.size());
  }

  for (auto& row : vec) {
    row.resize(max_col_size, pad_value);
  }
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
  budget_used_.push_back(0);
}

void Batch::add(const std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    add(sequence);
  }
}

void Batch::set_engine_type(EngineType engine_type) {
  // set engine type for all sequences in the batch
  for (auto* sequence : sequences_) {
    sequence->set_engine_type(engine_type);
  }
  // reset the budget used
  std::fill(budget_used_.begin(), budget_used_.end(), 0);
}

void Batch::clear() {
  sequences_.clear();
  token_budgets_.clear();
  budget_used_.clear();
}

// prepare inputs for the batch
// NOLINTNEXTLINE
ModelInput Batch::prepare_model_input(uint32_t num_decoding_tokens,
                                      uint32_t min_decoding_bach_size) {
  // flatten the token ids and positions
  std::vector<int32_t> flatten_tokens_vec;
  std::vector<int32_t> flatten_positions_vec;

  // sleceted tokens to return logits, including generated tokens and last
  // prompt token
  std::vector<const SamplingParameter*> sampling_params;
  std::vector<int32_t> selected_token_idxes;
  // track the last token of selected tokens for sampling
  std::vector<int32_t> sample_idxes;

  // track the unique token ids and counts in the batch
  std::vector<std::vector<int64_t>> unique_token_ids_vec;
  std::vector<std::vector<int32_t>> unique_token_counts_vec;
  std::vector<int32_t> unique_token_lens_vec;

  bool empty_kv_cache = true;
  uint32_t max_seq_len = 0;
  uint32_t q_max_seq_len = 0;
  std::vector<int32_t> cu_seq_lens = {0};
  std::vector<int32_t> q_cu_seq_lens = {0};
  // slot ids for new token
  std::vector<int32_t> new_token_slot_ids;
  std::vector<std::vector<int32_t>> block_tables_vec;
  const int32_t num_sequences = static_cast<int32_t>(sequences_.size());
  for (int32_t i = 0; i < num_sequences; ++i) {
    auto* sequence = sequences_[i];
    const auto token_ids = sequence->token_ids();
    const uint32_t n_tokens = token_ids.size();
    const uint32_t n_kv_cache_tokens = sequence->num_kv_cache_tokens();

    empty_kv_cache = empty_kv_cache && (n_kv_cache_tokens == 0);

    const uint32_t remaining_token_budget = token_budgets_[i] - budget_used_[i];
    if (remaining_token_budget == 0) {
      // no token budget left for the prefill sequence
      CHECK(sequence->is_prefill_stage());
      continue;
    }

    const uint32_t q_seq_len =
        std::min(n_tokens - n_kv_cache_tokens, remaining_token_budget);

    const uint32_t seq_len = q_seq_len + n_kv_cache_tokens;

    // check if the sequence has enough cache slots
    CHECK_GE(sequence->kv_cache_capacity(), seq_len);

    // at least one token to process otherwise the sequence should be finished.
    CHECK_GT(q_seq_len, 0) << "at least one token should be processed. "
                           << "n_tokens: " << n_tokens
                           << ", n_kv_cache_tokens: " << n_kv_cache_tokens
                           << ", kv_cache_capacity: "
                           << sequence->kv_cache_capacity()
                           << ", token_budget: " << token_budgets_[i];

    // update budget used
    budget_used_[i] += q_seq_len;

    // update sequence length
    max_seq_len = std::max(max_seq_len, seq_len);
    q_max_seq_len = std::max(q_max_seq_len, q_seq_len);
    cu_seq_lens.push_back(cu_seq_lens.back() + seq_len);
    q_cu_seq_lens.push_back(q_cu_seq_lens.back() + q_seq_len);

    // pack the token ids and positions into one-dimensional tensors
    // and select tokens for sampling the next token
    const uint32_t n_prompt_tokens = sequence->num_prompt_tokens();
    std::unordered_map<int32_t, int32_t> adjusted_token_to_count_map;
    for (uint32_t j = n_kv_cache_tokens; j < seq_len; ++j) {
      // skip prompt tokens except the last one
      if (j + 1 < n_prompt_tokens) {
        continue;
      }
      ++adjusted_token_to_count_map[token_ids[j]];
    }

    bool has_selected_token = false;
    for (uint32_t j = n_kv_cache_tokens; j < seq_len; ++j) {
      flatten_tokens_vec.push_back(token_ids[j]);
      flatten_positions_vec.push_back(static_cast<int32_t>(j));

      // skip prompt tokens except the last one
      if (j + 1 < n_prompt_tokens) {
        continue;
      }

      // adjust token count for current token
      --adjusted_token_to_count_map[token_ids[j]];

      // select tokens for sampling the next token
      selected_token_idxes.push_back(flatten_tokens_vec.size() - 1);
      sampling_params.push_back(sequence->sampling_param());

      // add token id and count for sampling
      const auto& seq_token_counts = sequence->token_to_count_map();
      const auto unique_tokens = seq_token_counts.size();

      auto& ids = unique_token_ids_vec.emplace_back();
      auto& counts = unique_token_counts_vec.emplace_back();
      ids.reserve(unique_tokens);
      counts.reserve(unique_tokens);
      for (const auto& [token_id, count] : seq_token_counts) {
        const auto it = adjusted_token_to_count_map.find(token_id);
        const auto adjust_count =
            it != adjusted_token_to_count_map.end() ? it->second : 0;
        if (count > adjust_count) {
          ids.push_back(token_id);
          counts.push_back(count - adjust_count);
        }
      }
      unique_token_lens_vec.push_back(static_cast<int32_t>(ids.size()));

      // sample last token in the sequence
      if (j == seq_len - 1) {
        sample_idxes.push_back(
            static_cast<int32_t>(selected_token_idxes.size() - 1));
      }
    }

    // commit kv cache to advance kv_cache pos in sequence
    sequence->commit_kv_cache(/*size=*/q_seq_len);

    // assign slot ids for new tokens [n_tokens_in_kvcache, total_tokens)
    const auto blocks = sequence->blocks();
    const auto slot_ids = sequence->kv_cache_slots(n_kv_cache_tokens, seq_len);
    new_token_slot_ids.insert(
        new_token_slot_ids.end(), slot_ids.begin(), slot_ids.end());

    // construct block ids for each sequence
    std::vector<int32_t> block_ids;
    block_ids.reserve(blocks.size());
    for (const auto& block : blocks) {
      block_ids.push_back(block.id());
    }
    block_tables_vec.push_back(block_ids);
  }

  if (flatten_tokens_vec.empty()) {
    // no tokens to process
    return {};
  }

  // padding the batch to the minimum decoding batch size for cuda graph
  // TODO: move the logic to a better place
  if (num_sequences < min_decoding_bach_size) {
    const uint32_t n_tokens = flatten_tokens_vec.size();
    // kv_cache is not empty in decoding phase
    const bool in_decoding_phase = !empty_kv_cache;
    const bool same_num_decoding_tokens =
        q_max_seq_len == num_decoding_tokens &&
        n_tokens == num_sequences * num_decoding_tokens;
    if (in_decoding_phase && same_num_decoding_tokens) {
      // add padding tokens to the batch
      for (int32_t i = num_sequences; i < min_decoding_bach_size; ++i) {
        for (int32_t k = 0; k < num_decoding_tokens; ++k) {
          flatten_tokens_vec.push_back(0);
          flatten_positions_vec.push_back(0);
          new_token_slot_ids.push_back(0);
        }
        cu_seq_lens.push_back(cu_seq_lens.back() + num_decoding_tokens);
        q_cu_seq_lens.push_back(q_cu_seq_lens.back() + num_decoding_tokens);
        block_tables_vec.emplace_back();
      }
    }
  }

  ModelInput model_inputs;
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

  pad_2d_vector(block_tables_vec, /*pad_value=*/0);
  input_params.block_tables = create_2d_tensor(block_tables_vec, torch::kInt);

  CHECK_EQ(sampling_params.size(), selected_token_idxes.size());
  if (!selected_token_idxes.empty()) {
    pad_2d_vector<int64_t>(unique_token_ids_vec, /*pad_value=*/0);
    pad_2d_vector(unique_token_counts_vec, /*pad_value=*/0);
    model_inputs.sampling_params.init(sampling_params,
                                      selected_token_idxes,
                                      sample_idxes,
                                      unique_token_ids_vec,
                                      unique_token_counts_vec,
                                      unique_token_lens_vec);
  }

  return model_inputs;
}

void Batch::process_sample_output(const SampleOutput& sample_output) {
  const auto& next_tokens = safe_to(sample_output.next_tokens, torch::kCPU);
  // it is possible that the model output is empty for prefill sequences
  if (next_tokens.defined()) {
    const auto& logprobs = safe_to(sample_output.logprobs, torch::kCPU);
    const auto& top_tokens = safe_to(sample_output.top_tokens, torch::kCPU);
    const auto& top_logprobs = safe_to(sample_output.top_logprobs, torch::kCPU);

    const int64_t num_seqs = next_tokens.numel();
    int64_t output_idx = 0;
    for (auto* seq : sequences_) {
      if (seq->is_prefill_stage()) {
        // no sampling for prefill sequences
        continue;
      }
      CHECK_LT(output_idx, num_seqs);

      // add the next token to sequence
      const auto curr_idx = output_idx++;

      const int32_t next_token_id =
          static_cast<int32_t>(next_tokens[curr_idx].item<int64_t>());

      TokenInfo token_info(next_token_id);
      if (logprobs.defined()) {
        token_info.logprob = logprobs[curr_idx].item<float>();
      }
      if (top_tokens.defined() && top_logprobs.defined()) {
        auto topk_tokens = top_tokens[curr_idx];
        auto topk_logprobs = top_logprobs[curr_idx];
        const size_t topk = topk_tokens.numel();
        token_info.top_tokens = {topk_tokens.const_data_ptr<int64_t>(), topk};
        token_info.top_logprobs = {topk_logprobs.const_data_ptr<float>(), topk};
      }
      seq->append_token(token_info);
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
    auto num_accepted_tokens = seq->validate_tokens(accepted_token_ids);
    COUNTER_ADD(num_accepted_tokens_total, num_accepted_tokens);
  }
  CHECK_EQ(output_idx, num_seqs);
}

}  // namespace llm
