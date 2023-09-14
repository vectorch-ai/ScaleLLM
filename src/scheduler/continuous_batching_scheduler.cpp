#include "continuous_batching_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/MPMCQueue.h>

#include <cstdint>
#include <memory>

#include "request/request.h"

namespace llm {
constexpr size_t kRequestQueueSize = 100000;
// TODO: reader from config
constexpr size_t kMaxBatchSize = 100;

constexpr uint64_t kStepSleepTimeMs = 10;

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
  for (Request* request : running_) {
    std::unique_ptr<Request> request_ptr(request);
  }
  running_.clear();
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

std::vector<Sequence*> ContinuousBatchingScheduler::create_sequence_batch() {
  // propogate new requests to priority_queue_
  while (!request_queue_.isEmpty()) {
    Request* request = nullptr;
    // read from request queue then push to priority queue
    request_queue_.read(request);
    CHECK(request != nullptr);
    priority_queue_.push(request);
  }

  std::vector<Request*> preemptable_candidates;
  for (Request* request : running_) {
    if (!request->is_finished()) {
      // the request is still holding cache slots
      preemptable_candidates.emplace_back(request);
      // push the request back to the priority queue
      priority_queue_.push(request);
      continue;
    }

    // release all blocks for the finished request
    block_manager_->release_slots_for_request(request);
    // take over the ownership of the request
    std::unique_ptr<Request> finished_request(request);
    response_executor_.schedule([request = std::move(finished_request)]() {
      // TODO: add finish handling logic
      request->on_finish("", Status());
    });
  }

  // schedule sequence by sequence but preempt whole request if necessary
  std::vector<Sequence*> sequences_batch;
  std::vector<Request*> request_batch;
  size_t begin_idx = 0;
  size_t end_idx = preemptable_candidates.size();
  while (!priority_queue_.empty()) {
    Request* candidate = priority_queue_.top();
    bool has_enough_slots = true;
    std::vector<Sequence*> sequence_candiadtes;
    sequence_candiadtes.reserve(candidate->sequences.size());
    for (Sequence& sequence : candidate->sequences) {
      if (block_manager_->allocate_slots_for_sequence(&sequence)) {
        sequence_candiadtes.push_back(&sequence);
      }
    }

    // all sequences in the request have enough slots
    if (sequence_candiadtes.size() == candidate->sequences.size()) {
      // add request to new batch
      priority_queue_.pop();
      request_batch.push_back(candidate);
      sequences_batch.insert(sequences_batch.end(),
                             sequence_candiadtes.begin(),
                             sequence_candiadtes.end());
      if (begin_idx < end_idx &&
          candidate == preemptable_candidates[begin_idx]) {
        // the request has been scheduled and can't be preempted
        ++begin_idx;
      }
      continue;
    }

    // try to preempt lowest priority request in current batch
    if (begin_idx < end_idx) {
      CHECK(end_idx > begin_idx);
      Request* request_to_preempt = preemptable_candidates[--end_idx];
      // avoid preempting the candidate request
      if (request_to_preempt != candidate) {
        block_manager_->release_slots_for_request(request_to_preempt);
        CHECK(begin_idx == end_idx);
      }
      continue;
    }

    // no requests left to preempt
    CHECK(begin_idx == end_idx);

    // partially schedule the request
    if (!sequence_candiadtes.empty()) {
      priority_queue_.pop();
      request_batch.push_back(candidate);
      sequences_batch.insert(sequences_batch.end(),
                             sequence_candiadtes.begin(),
                             sequence_candiadtes.end());
    }
    break;
  }
  CHECK(begin_idx == end_idx);

  if (sequences_batch.empty() && !priority_queue_.empty()) {
    // don't have enough memory to schedule one sequence
    LOG(ERROR) << "Not enough memory to schedule one sequence";
    Request* request = priority_queue_.top();
    priority_queue_.pop();
    // TODO: optimize the logic to only release blocks for sequences one by one
    // release all blocks for the request
    block_manager_->release_slots_for_request(request);
    // take over the ownership of the request
    std::unique_ptr<Request> finished_request(request);
    response_executor_.schedule([request = std::move(finished_request)]() {
      // TODO: add finish handling logic
      request->on_finish("", Status());
    });
  }

  running_ = std::move(request_batch);
  // DCHECK(running_.sorted())
  return sequences_batch;
}

// step the scheduler forward by one step
// may get blocked if there are no requests to process
void ContinuousBatchingScheduler::step(const absl::Duration& timeout) {
  // get a new batch of requests
  const auto deadline = absl::Now() + timeout;
  std::vector<Sequence*> seqs_batch;
  while (true) {
    seqs_batch = std::move(create_sequence_batch());
    if (!seqs_batch.empty()) {
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

  CHECK(!seqs_batch.empty());
  auto output_parameters = engine_->execute_model(seqs_batch);

  const auto& next_tokens = output_parameters.next_tokens;
  const int64_t num_seqs = next_tokens.numel();
  CHECK(num_seqs == seqs_batch.size());

  const int64_t* new_token_ids = next_tokens.data_ptr<int64_t>();
  // process sequence in batch
  for (int64_t i = 0; i < num_seqs; ++i) {
    Sequence* seq = seqs_batch[i];
    // append new token id to the sequence
    seq->append_new_token_id(static_cast<int>(new_token_ids[i]));

    // check if the sequence is finished and update its status
    seq->check_stopping_creteria();

    // stream delta to client if streaming is enabled
    if (seq->is_streaming()) {
      // TODO: move this into executor
      auto detla = seq->decode_delta_text(*tokenizer_);
      const auto finish_reason = seq->finish_reason();
      response_executor_.schedule(
          [seq, detla = std::move(detla), finish_reason = finish_reason]() {
            seq->stream_delta(detla, finish_reason);
          });
    }
  }
}

}  // namespace llm
