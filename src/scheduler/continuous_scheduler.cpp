#include "continuous_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/MPMCQueue.h>
#include <glog/logging.h>

#include <atomic>
#include <cstdint>
#include <memory>

#include "common/metrics.h"
#include "common/timer.h"
#include "engine/engine.h"
#include "request/request.h"
#include "request/sequence.h"

// metrics
DEFINE_GAUGE(num_pending_requests, "Number of pending requests in scheduler");
DEFINE_GAUGE(num_running_requests, "Number of running requests in scheduler");
DEFINE_GAUGE(num_waiting_requests, "Number of waiting requests in scheduler");
DEFINE_GAUGE(num_preempted_requests,
             "Number of preempted requests in scheduler");

DEFINE_GAUGE(kv_cache_utilization_perc,
             "Utilization of the kv cache in percentage");
DEFINE_GAUGE(num_blocks_in_prefix_cache,
             "Number of blocks in the prefix cache");
DEFINE_GAUGE(num_free_blocks, "Number of free blocks in the block allocator");
DEFINE_GAUGE(num_blocks_in_use, "Effective number of blocks in use");

DEFINE_COUNTER(scheduling_latency_seconds, "Latency of scheduling in seconds");

DEFINE_COUNTER_FAMILY(num_processing_tokens_total,
                      "Total number of processing tokens");
DEFINE_COUNTER_INSTANCE(num_prompt_tokens_total,
                        num_processing_tokens_total,
                        {{"type", "prompt"}});
DEFINE_COUNTER_INSTANCE(num_generated_tokens_total,
                        num_processing_tokens_total,
                        {{"type", "generated"}});

namespace llm {

constexpr size_t kRequestQueueSize = 100000;

ContinuousScheduler::ContinuousScheduler(Engine* engine, const Options& options)
    : options_(options), engine_(engine), request_queue_(kRequestQueueSize) {
  CHECK(engine_ != nullptr);
  block_manager_ = engine_->block_manager();
  CHECK(block_manager_ != nullptr);

  enable_prefix_cache_ = block_manager_->options().enable_prefix_cache();

  response_handler_ = std::make_unique<ResponseHandler>(engine_->tokenizer());
}

ContinuousScheduler::~ContinuousScheduler() {
  // release all requests in the queue
  Request* request = nullptr;
  while (request_queue_.read(request)) {
    CHECK(request != nullptr);
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
  Timer timer;

  // propogate new requests to priority_queue_
  Request* request = nullptr;
  // read from request queue then push to priority queue
  while (request_queue_.read(request)) {
    CHECK(request != nullptr);

    // expand sequences to the target number if prefix cache is disabled.
    if (!enable_prefix_cache_) {
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
    if (request->is_finished() || request->is_cancelled()) {
      block_manager_->release_blocks_for(request);
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

    // release blocks for finished sequences here
    for (Sequence& sequence : request->sequences) {
      if (sequence.is_finished()) {
        block_manager_->release_blocks_for(&sequence);
      }
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
  const size_t max_seqs_per_batch = std::max(options_.max_seqs_per_batch(), 1);

  // average number of token budget for each sequence.
  const size_t avg_sequence_token_budget =
      std::max<size_t>(options_.max_tokens_per_batch() / max_seqs_per_batch,
                       1 + options_.num_speculative_tokens());

  // remaining budget for the current batch
  // at least avg_sequence_token_budget token per sequence
  size_t remaining_token_budget =
      std::max<size_t>(options_.max_tokens_per_batch(),
                       max_seqs_per_batch * avg_sequence_token_budget);
  size_t remaining_seq_budget = max_seqs_per_batch;

  size_t num_preempted_requests = 0;
  // schedule the requests in the priority queue until budgets are exhausted
  while (!priority_queue_.empty() &&
         remaining_token_budget > options_.num_speculative_tokens() &&
         remaining_seq_budget > 0) {
    Request* request = priority_queue_.top();
    // TODO: check if request is timeout

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
      if (allocated_tokens + options_.num_speculative_tokens() >=
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
        ++num_preempted_requests;
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
    block_manager_->release_blocks_for(request);
    // release the ownership of the request
    response_handler_->on_request_finish(std::unique_ptr<Request>(request));
  }

  // update the batch
  size_t num_prompt_tokens = 0;
  size_t num_generated_tokens = 0;
  Batch batch;
  for (const SequenceData& seq_data : new_batch) {
    const size_t token_budget = seq_data.token_budget;
    auto* sequence = seq_data.sequence;

    const size_t remaining_prompt_tokens =
        sequence->num_prompt_tokens() > sequence->num_kv_cache_tokens()
            ? sequence->num_prompt_tokens() - sequence->num_kv_cache_tokens()
            : 0;
    const size_t prompt_tokens =
        std::min(remaining_prompt_tokens, token_budget);
    const size_t generated_tokens = token_budget - prompt_tokens;
    num_prompt_tokens += prompt_tokens;
    num_generated_tokens += generated_tokens;

    batch.add(sequence, token_budget);
  }

  // update metrics before returning
  if (!batch.empty()) {
    // only update the scheduling latency when there are requests to process
    COUNTER_ADD(scheduling_latency_seconds, timer.elapsed_seconds());
  }

  COUNTER_ADD(num_prompt_tokens_total, num_prompt_tokens);
  COUNTER_ADD(num_generated_tokens_total, num_generated_tokens);

  GAUGE_SET(num_pending_requests,
            pending_requests_.load(std::memory_order_relaxed));
  GAUGE_SET(num_running_requests, running_requests_.size());
  GAUGE_SET(num_waiting_requests, priority_queue_.size());
  GAUGE_SET(num_preempted_requests, num_preempted_requests);

  GAUGE_SET(kv_cache_utilization_perc, block_manager_->kv_cache_utilization());
  GAUGE_SET(num_blocks_in_prefix_cache,
            block_manager_->num_blocks_in_prefix_cache());
  GAUGE_SET(num_free_blocks, block_manager_->num_free_blocks());
  GAUGE_SET(num_blocks_in_use, block_manager_->num_blocks_in_use());
  return batch;
}

Batch ContinuousScheduler::wait_for_batch(const absl::Duration& timeout) {
  const auto deadline = absl::Now() + timeout;
  while (true) {
    Batch batch = build_sequence_batch();
    if (!batch.empty()) {
      return batch;
    }
    const auto now = absl::Now();
    if (now > deadline) {
      break;
    }
    // wait for new requests to arrive
    constexpr uint64_t kStepSleepTimeMs = 10;
    const auto time_to_sleep =
        std::min(absl::Milliseconds(kStepSleepTimeMs), deadline - now);
    absl::SleepFor(time_to_sleep);
  }
  // return an empty batch
  return {};
}

// step the scheduler forward by one step
// may get blocked if there are no requests to process
void ContinuousScheduler::step(const absl::Duration& timeout) {
  // get a new batch of requests
  Batch batch = wait_for_batch(timeout);
  if (batch.empty()) {
    return;
  }

  engine_->execute_model(batch);

  // process request output in batch
  process_batch_output();
}

void ContinuousScheduler::run_until_complete() {
  while (true) {
    // build a batch of requests/sequences
    auto batch = build_sequence_batch();
    if (batch.empty()) {
      if (pending_requests_.load(std::memory_order_relaxed) > 0) {
        // wait for new requests to arrive
        continue;
      }

      // no more requests to process
      break;
    }

    // run inference for the batch
    engine_->execute_model(batch);

    // process request output in batch
    process_batch_output();
  }

  // wait for all responses to be processed
  response_handler_->wait_for_complete();
}

void ContinuousScheduler::process_batch_output() {
  // process request output in batch
  for (Request* request : running_requests_) {
    if (request->is_streaming()) {
      response_handler_->on_request_stream(request);
    }
  }
}

bool ContinuousScheduler::allocate_blocks_for(Sequence* sequence,
                                              size_t token_budget,
                                              size_t* actual_tokens) {
  // token budget should be large enough for one speculative decoding step
  CHECK_GT(token_budget, options_.num_speculative_tokens());

  if (sequence->num_blocks() == 0) {
    // need to allocate shared blocks explicitly to avoid kv_cache_pos change
    block_manager_->allocate_shared_blocks_for(sequence);
  }

  // number of tokens in the kv cache, which are already processed
  const size_t num_kv_cache_tokens = sequence->num_kv_cache_tokens();
  // the total number tokens for the sequence
  size_t num_tokens =
      std::min(num_kv_cache_tokens + token_budget, sequence->num_tokens());

  // speculative decoding specific logic
  // make sure sequence either in prefill or decode phase in one step
  const size_t num_prompt_tokens = sequence->num_prompt_tokens();
  if (options_.num_speculative_tokens() > 0 &&
      num_tokens >= num_prompt_tokens) {
    // reach decode phase, try to allocate slots for speculative tokens
    const size_t adjusted_num_tokens =
        num_tokens + options_.num_speculative_tokens();
    if (adjusted_num_tokens > num_kv_cache_tokens + token_budget) {
      // over budget, force the sequence in prefill phase
      num_tokens = num_prompt_tokens - 1;
    } else {
      // decode phase
      num_tokens = adjusted_num_tokens;
    }
  }

  // make sure the sequence proceeds forward
  CHECK_GT(num_tokens, num_kv_cache_tokens);

  // the actual allocated tokens is the difference between the total
  // number of tokens and the number of tokens already processed
  *actual_tokens = num_tokens - num_kv_cache_tokens;
  // allocate blocks for the sequence
  return block_manager_->allocate_blocks_for(sequence, num_tokens);
}

}  // namespace llm
