#include "continuous_batching_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/MPMCQueue.h>
#include <glog/logging.h>

#include <cstdint>
#include <memory>

#include "request/request.h"
#include "request/sequence.h"

namespace llm {

constexpr size_t kRequestQueueSize = 100000;

DEFINE_int32(streaming_token_buffer_size,
             1,
             "number of tokens to buffer before streaming to client");

DEFINE_int32(max_tokens_per_batch, 2048, "max number of tokens per batch");
DEFINE_int32(max_seqs_per_batch, 256, "max number of sequences per batch");

ContinuousBatchingScheduler::ContinuousBatchingScheduler(Engine* engine)
    : engine_(engine), request_queue_(kRequestQueueSize) {
  CHECK(engine_ != nullptr);
  block_manager_ = engine_->block_manager();
  tokenizer_ = engine_->tokenizer();
  CHECK(block_manager_ != nullptr);
  CHECK(tokenizer_ != nullptr);
}

ContinuousBatchingScheduler::~ContinuousBatchingScheduler() {
  // release all requests in the queue
  while (!request_queue_.isEmpty()) {
    Request* request = nullptr;
    request_queue_.read(request);
    std::unique_ptr<Request> request_ptr(request);
  }

  // release all requests in the priority queue
  while (!priority_queue_.empty()) {
    Request* request = priority_queue_.top();
    priority_queue_.pop();
    std::unique_ptr<Request> request_ptr(request);
  }

  // release all requests in the batch
  for (Request* request : requests_batch_) {
    std::unique_ptr<Request> request_ptr(request);
  }
  sequences_batch_.clear();
  requests_batch_.clear();
}

void ContinuousBatchingScheduler::on_request_finish(Request* request) {
  // release all blocks for the finished request
  block_manager_->release_blocks_for(request);
  // take over the ownership of the request
  std::unique_ptr<Request> finished_request(request);
  response_threadpool_.schedule([tokenizer = tokenizer_.get(),
                                 request = std::move(finished_request)]() {
    if (request->stream) {
      // just finish the request
      request->on_stream_finish(Status());
    } else {
      // summarize statistics for all sequences
      Statistics stats;
      stats.num_prompt_tokens = request->num_prompt_tokens();
      for (const Sequence& seq : request->sequences) {
        stats.num_generated_tokens += seq.num_generated_tokens();
      }
      stats.num_total_tokens =
          stats.num_prompt_tokens + stats.num_generated_tokens;

      std::vector<SequenceResult> seq_results;
      seq_results.reserve(request->sequences.size());
      for (Sequence& seq : request->sequences) {
        // generate the final output
        const auto output = seq.decode_delta_text(seq.num_tokens(), *tokenizer);
        seq_results.push_back({output, seq.finish_reason()});
      }
      request->on_finish(seq_results, Status(), stats);
    }
  });
}

void ContinuousBatchingScheduler::on_sequence_stream(Sequence* seq) {
  // check if the sequence has enough tokens to output
  const size_t num_tokens = seq->num_tokens();
  const size_t output_offset = seq->output_offset();
  const size_t num_tokens_to_output = num_tokens - output_offset;
  if (seq->is_finished() ||
      num_tokens_to_output >= FLAGS_streaming_token_buffer_size) {
    const auto finish_reason = seq->finish_reason();
    // output the delta text til the end of the sequence to the client
    response_threadpool_.schedule(
        [seq, tokenizer = tokenizer_.get(), end = num_tokens, finish_reason]() {
          const auto detla = seq->decode_delta_text(end, *tokenizer);
          if (!detla.empty() || finish_reason != FinishReason::NONE) {
            seq->stream_delta(detla, finish_reason);
          };
        });
  }
}

bool ContinuousBatchingScheduler::schedule(std::unique_ptr<Request>& request) {
  CHECK(request != nullptr);
  if (request_queue_.write(request.get())) {
    // take over the ownership of the request
    request.release();
    return true;
  }
  // queue is full
  return false;
}

void ContinuousBatchingScheduler::build_sequence_batch() {
  // propogate new requests to priority_queue_
  while (!request_queue_.isEmpty()) {
    Request* request = nullptr;
    // read from request queue then push to priority queue
    request_queue_.read(request);
    CHECK(request != nullptr);
    priority_queue_.push(request);
  }

  // access in reverse order to iterate from the lowest priority to the highest
  for (auto it = requests_batch_.rbegin(); it != requests_batch_.rend(); ++it) {
    Request* request = *it;
    if (request->is_finished()) {
      on_request_finish(request);
      continue;
    }

    // put it to the front of the preemptable queue as it has higher priority
    preemptable_candidates_.push_front(request);
    // push the request back to the priority queue
    priority_queue_.push(request);
  }

  struct SequenceData {
    Sequence* sequence = nullptr;
    // tokens to process in this iteration
    size_t token_budget = 0;
  };
  std::vector<SequenceData> sequences_batch;
  std::vector<Request*> requests_batch;

  // average number of token budget for each sequence.
  const size_t avg_sequence_token_budget =
      std::max(FLAGS_max_tokens_per_batch / FLAGS_max_seqs_per_batch, 1);

  // remaining budget for the current batch
  // at least one token per sequence
  size_t remaining_token_budget =
      std::max(FLAGS_max_tokens_per_batch, FLAGS_max_seqs_per_batch);
  // at least one sequence per batch
  size_t remaining_seq_budget = std::max(FLAGS_max_seqs_per_batch, 1);

  // schedule the requests in the priority queue until budgets are exhausted
  while (!priority_queue_.empty() && remaining_token_budget > 0 &&
         remaining_seq_budget > 0) {
    Request* request = priority_queue_.top();
    std::vector<SequenceData> candidates;
    candidates.reserve(request->sequences.size());

    bool has_enough_blocks = true;
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    for (Sequence& sequence : request->sequences) {
      // skip finished sequence.
      if (sequence.is_finished()) {
        continue;
      }
      // no budget left
      if (allocated_tokens >= remaining_token_budget ||
          allocated_seqs >= remaining_seq_budget) {
        break;
      }

      const size_t token_budget = std::min(
          avg_sequence_token_budget, remaining_token_budget - allocated_tokens);
      size_t actual_tokens = 0;
      // no blocks left
      if (!allocate_blocks_for(&sequence, token_budget, &actual_tokens)) {
        has_enough_blocks = false;
        break;
      }

      // update the allocated tokens for the sequence
      allocated_tokens += actual_tokens;
      allocated_seqs += 1;
      candidates.push_back({&sequence, actual_tokens});
    }
    CHECK(allocated_tokens <= remaining_token_budget);
    CHECK(allocated_seqs <= remaining_seq_budget);

    // schedule candidates in the request if there are enough blocks
    if (has_enough_blocks) {
      // remove the request from the priority queue
      priority_queue_.pop();
      // add the request to the batch
      requests_batch.push_back(request);
      sequences_batch.insert(
          sequences_batch.end(), candidates.begin(), candidates.end());
      remaining_token_budget -= allocated_tokens;
      remaining_seq_budget -= allocated_seqs;

      // the request has been scheduled and can't be preempted
      if (!preemptable_candidates_.empty() &&
          request == preemptable_candidates_.front()) {
        preemptable_candidates_.pop_front();
      }
      continue;
    }

    // otherwise, preempt lowest priority request and retry
    if (!preemptable_candidates_.empty()) {
      Request* request_to_preempt = preemptable_candidates_.back();
      preemptable_candidates_.pop_back();

      // avoid preempting the candidate itself
      if (request_to_preempt != request) {
        block_manager_->release_blocks_for(request_to_preempt);
      }
      continue;
    }

    // no requests left to preempt, partially schedule the request
    if (!candidates.empty()) {
      priority_queue_.pop();
      requests_batch.push_back(request);
      sequences_batch.insert(
          sequences_batch.end(), candidates.begin(), candidates.end());
      remaining_token_budget -= allocated_tokens;
      remaining_seq_budget -= allocated_seqs;
    }
    break;
  }

  // adjust the token number for each sequence if still have token budget left
  if (remaining_token_budget > 0) {
    for (SequenceData& seq_data : sequences_batch) {
      // add previous allocated tokens back
      remaining_token_budget += seq_data.token_budget;
      size_t actual_tokens = 0;
      // no memory left
      if (!allocate_blocks_for(
              seq_data.sequence, remaining_token_budget, &actual_tokens)) {
        break;
      }
      // update the allocated tokens for the sequence
      seq_data.token_budget = actual_tokens;
      CHECK(remaining_token_budget >= actual_tokens);
      remaining_token_budget -= actual_tokens;

      // no budget left
      if (remaining_token_budget == 0) {
        break;
      }
    }
  }

  // update the batch
  requests_batch_ = std::move(requests_batch);
  sequences_batch_.clear();
  for (const SequenceData& seq_data : sequences_batch) {
    sequences_batch_.add(seq_data.sequence, seq_data.token_budget);
  }

  if (sequences_batch_.empty() && !priority_queue_.empty()) {
    LOG(ERROR) << "No enough memory to schedule single sequence";
    // no enough memory to schedule single sequence, just finish the request
    Request* request = priority_queue_.top();
    priority_queue_.pop();
    on_request_finish(request);
  }
}

// step the scheduler forward by one step
// may get blocked if there are no requests to process
void ContinuousBatchingScheduler::step(const absl::Duration& timeout) {
  // get a new batch of requests
  const auto deadline = absl::Now() + timeout;
  while (true) {
    build_sequence_batch();
    if (!sequences_batch_.empty()) {
      // find one batch of requests to process
      break;
    }
    const auto now = absl::Now();
    if (now > deadline) {
      // no requests to process
      return;
    }
    // wait for new requests to arrive
    constexpr uint64_t kStepSleepTimeMs = 10;
    const auto time_to_sleep =
        std::min(absl::Milliseconds(kStepSleepTimeMs), deadline - now);
    absl::SleepFor(time_to_sleep);
  }

  CHECK(!sequences_batch_.empty());
  auto model_output = engine_->execute_model(sequences_batch_);

  const auto& next_tokens = model_output.next_tokens;
  const int64_t num_seqs = next_tokens.numel();
  CHECK(num_seqs == sequences_batch_.size());

  const int64_t* new_token_ids = next_tokens.data_ptr<int64_t>();
  // process sequence in batch
  for (int64_t i = 0; i < num_seqs; ++i) {
    Sequence* seq = sequences_batch_[i];
    const int32_t next_token_id = static_cast<int32_t>(new_token_ids[i]);
    // add the next token to sequence and check if the sequence is finished
    seq->append_new_token_id(next_token_id);

    // stream delta to client if streaming is enabled
    if (seq->is_streaming()) {
      on_sequence_stream(seq);
    }
  }
}

bool ContinuousBatchingScheduler::allocate_blocks_for(Sequence* sequence,
                                                      size_t token_budget,
                                                      size_t* actual_tokens) {
  CHECK(token_budget > 0);
  // number of tokens in the kv cache, which are already processed
  const size_t num_tokens_in_kv_cache = sequence->num_tokens_in_kv_cache();
  // the number tokens can be allocated for the sequence, honoring the
  // token budget.
  const size_t num_tokens =
      std::min(num_tokens_in_kv_cache + token_budget, sequence->num_tokens());
  CHECK(num_tokens >= num_tokens_in_kv_cache);

  // the actual allocated tokens is the difference between the total
  // number of tokens and the number of tokens already processed
  *actual_tokens = num_tokens - num_tokens_in_kv_cache;
  // allocate blocks for the sequence
  return block_manager_->allocate_blocks_for(sequence, num_tokens);
}

}  // namespace llm
