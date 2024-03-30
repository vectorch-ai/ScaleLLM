#include "continuous_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/MPMCQueue.h>
#include <glog/logging.h>

#include <cstdint>
#include <memory>

#include "engine/engine.h"
#include "request/request.h"
#include "request/sequence.h"

DEFINE_int32(max_tokens_per_batch, 1024, "max number of tokens per batch");
DEFINE_int32(max_seqs_per_batch, 128, "max number of sequences per batch");
DEFINE_int32(num_speculative_steps, 0, "number of speculative steps");

DECLARE_bool(enable_prefix_cache);

namespace llm {

constexpr size_t kRequestQueueSize = 100000;

ContinuousScheduler::ContinuousScheduler(Engine* engine)
    : engine_(engine), request_queue_(kRequestQueueSize) {
  CHECK(engine_ != nullptr);
  block_manager_ = engine_->block_manager();
  tokenizer_ = engine_->tokenizer();
  CHECK(block_manager_ != nullptr);
  CHECK(tokenizer_ != nullptr);

  response_handler_ =
      std::make_unique<ResponseHandler>(block_manager_, tokenizer_.get());
}

ContinuousScheduler::~ContinuousScheduler() {
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

  // release all running requests
  for (Request* request : running_requests_) {
    std::unique_ptr<Request> request_ptr(request);
  }
  running_requests_.clear();
}

bool ContinuousScheduler::schedule(std::unique_ptr<Request>& request) {
  CHECK(request != nullptr);
  CHECK(!request->sequences.empty());

  if (request_queue_.write(request.get())) {
    // take over the ownership of the request
    request.release();
    return true;
  }
  // queue is full
  return false;
}

Batch ContinuousScheduler::build_sequence_batch() {
  // propogate new requests to priority_queue_
  while (!request_queue_.isEmpty()) {
    Request* request = nullptr;
    // read from request queue then push to priority queue
    request_queue_.read(request);
    CHECK(request != nullptr);

    // expand sequences to the target number if prefix cache is disabled.
    if (!FLAGS_enable_prefix_cache) {
      // expand sequences to the target number
      request->expand_sequences();
    }

    priority_queue_.push(request);
  }

  // insert running requests back to the priority queue, iterating from the
  // lowest priority to the highest
  for (auto it = running_requests_.rbegin(); it != running_requests_.rend();
       ++it) {
    Request* request = *it;
    if (request->is_finished()) {
      // release the ownership of the request
      response_handler_->on_request_finish(std::unique_ptr<Request>(request));
      continue;
    }

    // check if the request can be expanded
    if (request->should_expand_sequences()) {
      // cache the blocks to share among the sequences
      block_manager_->cache_blocks_for(&request->sequences[0]);
      // expand sequences to the target number
      request->expand_sequences();
    }

    // put it to the front of the preemptable queue as it has higher priority
    preemptable_requests_.push_front(request);
    // push the request back to the priority queue
    priority_queue_.push(request);
  }
  running_requests_.clear();

  struct SequenceData {
    Sequence* sequence = nullptr;
    // tokens to process in this iteration
    size_t token_budget = 0;
  };
  std::vector<SequenceData> new_batch;

  // at least one sequence per batch
  const size_t max_seqs_per_batch = std::max(FLAGS_max_seqs_per_batch, 1);

  // average number of token budget for each sequence.
  const size_t avg_sequence_token_budget =
      std::max<size_t>(FLAGS_max_tokens_per_batch / max_seqs_per_batch,
                       1 + FLAGS_num_speculative_steps);

  // remaining budget for the current batch
  // at least avg_sequence_token_budget token per sequence
  size_t remaining_token_budget =
      std::max<size_t>(FLAGS_max_tokens_per_batch,
                       max_seqs_per_batch * avg_sequence_token_budget);
  size_t remaining_seq_budget = max_seqs_per_batch;

  // schedule the requests in the priority queue until budgets are exhausted
  while (!priority_queue_.empty() &&
         remaining_token_budget > FLAGS_num_speculative_steps &&
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
      if (allocated_tokens + FLAGS_num_speculative_steps >=
              remaining_token_budget ||
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
      running_requests_.push_back(request);
      new_batch.insert(new_batch.end(), candidates.begin(), candidates.end());
      remaining_token_budget -= allocated_tokens;
      remaining_seq_budget -= allocated_seqs;

      // the request has been scheduled and can't be preempted
      if (!preemptable_requests_.empty() &&
          request == preemptable_requests_.front()) {
        preemptable_requests_.pop_front();
      }
      continue;
    }

    // otherwise, preempt lowest priority request and retry
    if (!preemptable_requests_.empty()) {
      Request* request_to_preempt = preemptable_requests_.back();
      preemptable_requests_.pop_back();

      // avoid preempting the candidate itself
      if (request_to_preempt != request) {
        block_manager_->release_blocks_for(request_to_preempt);
      }
      continue;
    }

    // no requests left to preempt, partially schedule the request
    if (!candidates.empty()) {
      priority_queue_.pop();
      running_requests_.push_back(request);
      new_batch.insert(new_batch.end(), candidates.begin(), candidates.end());
      remaining_token_budget -= allocated_tokens;
      remaining_seq_budget -= allocated_seqs;
    }
    break;
  }

  // adjust the token number for each sequence if still have token budget left
  if (remaining_token_budget > 0) {
    for (SequenceData& seq_data : new_batch) {
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

  if (new_batch.empty() && !priority_queue_.empty()) {
    LOG(ERROR) << "No enough memory to schedule single sequence";
    // no enough memory to schedule single sequence, just finish the request
    Request* request = priority_queue_.top();
    priority_queue_.pop();
    // release the ownership of the request
    response_handler_->on_request_finish(std::unique_ptr<Request>(request));
  }

  // update the batch
  Batch batch;
  for (const SequenceData& seq_data : new_batch) {
    batch.add(seq_data.sequence, seq_data.token_budget);
  }
  return batch;
}

// step the scheduler forward by one step
// may get blocked if there are no requests to process
void ContinuousScheduler::step(const absl::Duration& timeout) {
  // get a new batch of requests
  const auto deadline = absl::Now() + timeout;
  Batch batch;
  while (true) {
    batch = build_sequence_batch();
    if (!batch.empty()) {
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

  engine_->execute_model(batch);

  // process sequence in batch
  for (int64_t i = 0; i < batch.size(); ++i) {
    Sequence* seq = batch[i];
    // stream delta to client if streaming is enabled
    if (seq->is_streaming()) {
      response_handler_->on_sequence_stream(seq);
    }
  }
}

bool ContinuousScheduler::allocate_blocks_for(Sequence* sequence,
                                              size_t token_budget,
                                              size_t* actual_tokens) {
  CHECK_GT(token_budget, FLAGS_num_speculative_steps);

  // need to allocate shared blocks explicitly to avoid kv_cache_pos change
  if (sequence->num_blocks() == 0) {
    block_manager_->allocate_shared_blocks_for(sequence);
  }

  // number of tokens in the kv cache, which are already processed
  const size_t num_tokens_in_kv_cache = sequence->num_tokens_in_kv_cache();
  // the number tokens can be allocated for the sequence, honoring the
  // token budget.
  size_t num_tokens =
      std::min(num_tokens_in_kv_cache + token_budget, sequence->num_tokens());

  // make sure sequence either in prefill or decode phase to allign
  // speculative decoding progress
  if (num_tokens >= sequence->num_prompt_tokens()) {
    // check if over budget
    if (num_tokens + FLAGS_num_speculative_steps >
        num_tokens_in_kv_cache + token_budget) {
      // over budget, force to process the prefill phase only
      num_tokens = sequence->num_prompt_tokens() - 1;
    } else {
      // decode phase
      num_tokens += FLAGS_num_speculative_steps;
    }
  }

  // make sure the sequence proceeds
  CHECK(num_tokens > num_tokens_in_kv_cache);

  // the actual allocated tokens is the difference between the total
  // number of tokens and the number of tokens already processed
  *actual_tokens = num_tokens - num_tokens_in_kv_cache;
  // allocate blocks for the sequence
  return block_manager_->allocate_blocks_for(sequence, num_tokens);
}

}  // namespace llm
