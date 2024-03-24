#include "speculative_scheduler.h"

#include <glog/logging.h>

#include "engine/engine.h"
#include "request/request.h"
#include "request/sequence.h"

namespace llm {

constexpr uint64_t kStepSleepTimeMs = 10;

SpeculativeScheduler::SpeculativeScheduler(const SchedulerConfig& config,
                                           Engine* llm_engine,
                                           Engine* ssm_engine)
    : config_(config), llm_engine_(llm_engine), ssm_engine_(ssm_engine) {
  CHECK(llm_engine_ != nullptr);
  llm_block_manager_ = llm_engine_->block_manager();
  CHECK(ssm_engine_ != nullptr);
  ssm_block_manager_ = ssm_engine_->block_manager();

  tokenizer_ = llm_engine_->tokenizer();
  CHECK(llm_block_manager_ != nullptr);
  CHECK(ssm_block_manager_ != nullptr);
  CHECK(tokenizer_ != nullptr);

  response_handler_ =
      std::make_unique<ResponseHandler>(llm_block_manager_, tokenizer_.get());
  scheduler_policy_ = std::make_unique<FCFSSchedulerPolicy>(
      response_handler_.get(), llm_block_manager_);
}

bool SpeculativeScheduler::schedule(std::unique_ptr<Request>& request) {
  return scheduler_policy_->schedule(request);
}

void SpeculativeScheduler::step(const absl::Duration& timeout) {
  // get a new batch of requests
  const auto deadline = absl::Now() + timeout;
  std::vector<Sequence*> spec_sequences_batch;
  while (true) {
    spec_sequences_batch = scheduler_policy_->build_batch();
    if (!spec_sequences_batch.empty()) {
      break;
    }
    const auto now = absl::Now();
    if (now > deadline) {
      return;
    }
    const auto time_to_sleep =
        std::min(absl::Milliseconds(kStepSleepTimeMs), deadline - now);
    absl::SleepFor(time_to_sleep);
  }

  // run multiple steps on ssm to generate multiple tokens.
  speculate_multiple_steps(spec_sequences_batch);

  // run validation on llm.
  validate(spec_sequences_batch);

  for (auto* seq : spec_sequences_batch) {
    // seq->update_valid_token_ids(ids);
    // stream delta to client if streaming is enabled
    if (seq->is_streaming()) {
      response_handler_->on_sequence_stream(seq);
    }
  }
}

void SpeculativeScheduler::speculate_multiple_steps(
    std::vector<Sequence*>& sequences_batch) {
  CHECK(!sequences_batch.empty());

  // TODO: should not support beam search
  std::vector<Sequence*> spec_batch(sequences_batch);
  for (uint64_t i = 0; i < config_.speculative_steps_; ++i) {
    ssm_engine_->execute_model(spec_batch);

    std::vector<Sequence*> next_spec_batch;
    next_spec_batch.reserve(spec_batch.size());
    for (auto* seq : spec_batch) {
      // TODO: record speculative token ids
      // seq->append_spec_token_id(next_token_id);
      if (!seq->is_finished()) {
        next_spec_batch.emplace_back(seq);
      }
    }
    spec_batch.swap(next_spec_batch);
  }
}

void SpeculativeScheduler::validate(std::vector<Sequence*>& sequences_batch) {
  llm_engine_->validate(sequences_batch);
}

}  // namespace llm
