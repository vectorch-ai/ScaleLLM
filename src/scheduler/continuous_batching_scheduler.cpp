#include "continuous_batching_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/MPMCQueue.h>
#include <glog/logging.h>

#include <cstdint>
#include <memory>

#include "request/request.h"

namespace llm {
constexpr size_t kRequestQueueSize = 100000;
// TODO: reader from config
constexpr size_t kMaxBatchSize = 100;

constexpr uint64_t kStepSleepTimeMs = 10;

DEFINE_int32(streaming_token_buffer_size,
             1,
             "number of tokens to buffer before streaming to client");

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
  for (Request* request : request_batch_) {
    std::unique_ptr<Request> request_ptr(request);
  }
  sequences_batch_.clear();
  request_batch_.clear();
}

void ContinuousBatchingScheduler::on_request_finish(Request* request) {
  // release all blocks for the finished request
  block_manager_->release_slots_for_request(request);
  // take over the ownership of the request
  std::unique_ptr<Request> finished_request(request);
  response_executor_.schedule([request = std::move(finished_request)]() {
    // TODO: add finish handling logic
    request->on_finish("", Status());
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
    response_executor_.schedule(
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

  // access in reverse order
  for (auto it = request_batch_.rbegin(); it != request_batch_.rend(); ++it) {
    Request* request = *it;
    if (request->is_finished()) {
      on_request_finish(request);
      continue;
    }

    // the request is still holding cache slots
    preemptable_candidates_.push_front(request);
    // push the request back to the priority queue
    priority_queue_.push(request);
  }

  // clear previous batch
  sequences_batch_.clear();
  request_batch_.clear();

  // schedule sequence by sequence but preempt whole request if necessary
  while (!priority_queue_.empty()) {
    Request* candidate = priority_queue_.top();
    bool has_enough_slots = true;
    std::vector<Sequence*> sequence_candiadtes;
    sequence_candiadtes.reserve(candidate->sequences.size());
    for (Sequence& sequence : candidate->sequences) {
      if (sequence.is_finished()) {
        // skip finished sequence.
        continue;
      }
      if (block_manager_->allocate_slots_for_sequence(&sequence)) {
        sequence_candiadtes.push_back(&sequence);
      } else {
        has_enough_slots = false;
      }
    }

    // all sequences in the request have enough slots to schedule
    if (has_enough_slots) {
      // add request to new batch
      priority_queue_.pop();
      request_batch_.push_back(candidate);
      sequences_batch_.insert(sequences_batch_.end(),
                              sequence_candiadtes.begin(),
                              sequence_candiadtes.end());
      if (!preemptable_candidates_.empty() &&
          candidate == preemptable_candidates_.front()) {
        // the request has been scheduled and can't be preempted
        preemptable_candidates_.pop_front();
      }
      continue;
    }

    // try to preempt lowest priority request in current batch
    if (!preemptable_candidates_.empty()) {
      Request* request_to_preempt = preemptable_candidates_.back();
      preemptable_candidates_.pop_back();
      // avoid preempting the candidate request
      if (request_to_preempt != candidate) {
        block_manager_->release_slots_for_request(request_to_preempt);
      }
      continue;
    }

    // no requests left to preempt, partially schedule the request
    if (!sequence_candiadtes.empty()) {
      priority_queue_.pop();
      request_batch_.push_back(candidate);
      sequences_batch_.insert(sequences_batch_.end(),
                              sequence_candiadtes.begin(),
                              sequence_candiadtes.end());
    }
    break;
  }

  if (sequences_batch_.empty() && !priority_queue_.empty()) {
    // don't have enough memory to schedule one sequence
    LOG(ERROR) << "Not enough memory to schedule one sequence";
    Request* request = priority_queue_.top();
    priority_queue_.pop();

    // TODO: optimize the logic to only release blocks for sequences one by one
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
    const auto time_to_sleep =
        std::min(absl::Milliseconds(kStepSleepTimeMs), deadline - now);
    absl::SleepFor(time_to_sleep);
  }

  CHECK(!sequences_batch_.empty());
  auto output_parameters = engine_->execute_model(sequences_batch_);

  const auto& next_tokens = output_parameters.next_tokens;
  const int64_t num_seqs = next_tokens.numel();
  CHECK(num_seqs == sequences_batch_.size());

  const int64_t* new_token_ids = next_tokens.data_ptr<int64_t>();
  // process sequence in batch
  for (int64_t i = 0; i < num_seqs; ++i) {
    Sequence* seq = sequences_batch_[i];
    // append new token id to the sequence
    seq->append_new_token_id(static_cast<int>(new_token_ids[i]));

    // check if the sequence is finished and update its status
    seq->check_stopping_creteria();

    // stream delta to client if streaming is enabled
    if (seq->is_streaming()) {
      on_sequence_stream(seq);
    }
  }
}

}  // namespace llm
