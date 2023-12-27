#include "speculative_scheduler.h"

#include "common/logging.h"
#include "engine/engine.h"
#include "memory/block_manager.h"
#include "request/request.h"
#include "request/sequence.h"
#include "scheduler/scheduler_policy.h"
#include "scheduler/response_handler.h"

namespace llm {

constexpr uint64_t kStepSleepTimeMs = 10;

SpeculativeScheduler::SpeculativeScheduler(Engine* llm_engine,
                                           Engine* ssm_engine)
    : llm_engine_(llm_engine), ssm_engine_(ssm_engine) {
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
  std::vector<Sequence*> sequences_batch;
  while (true) {
    sequences_batch = scheduler_policy_->build_batch();
    if (!sequences_batch.empty()) {
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

  GCHECK(!sequences_batch.empty());
  // TODO: execute multiple steps in ssm
  auto ssm_output_parameters = ssm_engine_->execute_model(sequences_batch);
  // TODO: execute on validation step in llm
  auto llm_output_parameters = llm_engine_->execute_model(sequences_batch);

  const auto& next_tokens = llm_output_parameters.next_tokens;
  const int64_t num_seqs = next_tokens.numel();
  GCHECK(num_seqs == sequences_batch.size());

  const int64_t* new_token_ids = next_tokens.data_ptr<int64_t>();
  // process sequence in batch
  for (int64_t i = 0; i < num_seqs; ++i) {
    Sequence* seq = sequences_batch[i];
    //TODO here wouldbe multiple new tokens
    const int32_t next_token_id = static_cast<int32_t>(new_token_ids[i]);
    // add the next token to sequence and check if the sequence is finished
    seq->append_new_token_id(next_token_id);

    // stream delta to client if streaming is enabled
    if (seq->is_streaming()) {
      response_handler_->on_sequence_stream(seq);
    }
  }
}

}  // namespace llm
