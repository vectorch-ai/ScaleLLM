#include "speculative_scheduler.h"

#include "common/logging.h"
#include "engine/engine.h"
#include "memory/block_manager.h"
#include "request/request.h"
#include "request/sequence.h"
#include "scheduler/response_handler.h"
#include "scheduler/scheduler_policy.h"

namespace llm {

constexpr uint64_t kStepSleepTimeMs = 10;

SpeculativeScheduler::SpeculativeScheduler(const SchedulerConfig& config,
                                           Engine* llm_engine,
                                           Engine* ssm_engine)
    : config_(config), llm_engine_(llm_engine), ssm_engine_(ssm_engine) {
  GCHECK(llm_engine_ != nullptr);
  llm_block_manager_ = llm_engine_->block_manager();
  GCHECK(ssm_engine_ != nullptr);
  ssm_block_manager_ = ssm_engine_->block_manager();

  tokenizer_ = llm_engine_->tokenizer();
  GCHECK(llm_block_manager_ != nullptr);
  GCHECK(ssm_block_manager_ != nullptr);
  GCHECK(tokenizer_ != nullptr);

  response_handler_ = new ResponseHandler(llm_block_manager_,
                                          tokenizer_.get());
  scheduler_policy_ = new FCFSSchedulerPolicy(response_handler_,
                                              llm_block_manager_);
}

SpeculativeScheduler::~SpeculativeScheduler() {
  delete scheduler_policy_;
  delete response_handler_;
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
  auto output_parameters = validate(spec_sequences_batch);

  const auto& next_tokens = output_parameters.next_tokens;
  const int64_t num_seqs = next_tokens.numel();
  GCHECK(num_seqs == spec_sequences_batch.size());

  const int64_t* new_token_ids = next_tokens.data_ptr<int64_t>();
  // process sequence in batch
  for (int64_t i = 0; i < num_seqs; ++i) {
    Sequence* seq = spec_sequences_batch[i];
    //TODO here would be multiple new tokens
    const int32_t next_token_id = static_cast<int32_t>(new_token_ids[i]);
    // add the next token to sequence and check if the sequence is finished
    seq->append_new_token_id(next_token_id);

    // stream delta to client if streaming is enabled
    if (seq->is_streaming()) {
      response_handler_->on_sequence_stream(seq);
    }
  }
}

void SpeculativeScheduler::speculate_multiple_steps(
    std::vector<Sequence*>& sequences_batch) {
  GCHECK(!sequences_batch.empty());

  //TODO should not support beam search
  std::vector<Sequence*> spec_batch(sequences_batch);
  for (uint64_t i = 0; i < config_.speculative_steps; ++i) {
    auto output_parameters = ssm_engine_->execute_model(spec_batch);

    const auto& next_tokens = output_parameters.next_tokens;
    const int64_t num_seqs = next_tokens.numel();
    GCHECK(num_seqs == spec_batch.size());

    std::vector<Sequence*> next_spec_batch;
    next_spec_batch.reserve(spec_batch.size());
    const int64_t* new_token_ids = next_tokens.data_ptr<int64_t>();
    for (int64_t i = 0; i < num_seqs; ++i) {
      auto seq = spec_batch[i];
      const auto next_token_id = static_cast<int32_t>(new_token_ids[i]);
      seq->append_new_token_id(next_token_id);
      seq->append_spec_token_id(next_token_id);

      if (!seq->is_finished()) {
        next_spec_batch.emplace_back(seq);
      }
    }
    spec_batch.swap(next_spec_batch);
  }
}

// TODO: OutputParameters should be extent to support multiple tokens output
OutputParameters SpeculativeScheduler::validate(
    std::vector<Sequence*>& sequences_batch) {
  return llm_engine_->validate(sequences_batch);
}

}  // namespace llm
