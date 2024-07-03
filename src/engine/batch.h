#pragma once

#include <torch/torch.h>

#include <limits>
#include <vector>

#include "parameters.h"
#include "request/sequence.h"

namespace llm {

// A thin wrapper for a batch of sequences
// it is used to prepare inputs for the model and manage the outputs in a
// centralized way
// TODO: refactor the code to use this class
class Batch {
 public:
  Batch() = default;

  // it is on purpose to allow implicit conversion
  Batch(Sequence* sequence);
  Batch(const std::vector<Sequence*>& sequences);

  // reset the batch with new sequences
  void reset(const std::vector<Sequence*>& sequences);

  // add sequence into the batch with token budget
  // caller should make sure the sequence is valid
  void add(Sequence* sequence,
           uint32_t token_budget = std::numeric_limits<uint32_t>::max());

  void add(const std::vector<Sequence*>& sequences);

  // get the number of sequences in the batch
  size_t size() const { return sequences_.size(); }
  bool empty() const { return sequences_.empty(); }

  // clear the batch for reuse
  void clear();
  void reset() { clear(); }

  // index operator
  // TODO: remove this operator once refactoring is done
  Sequence* operator[](size_t i) { return sequences_[i]; }

  // prepare inputs for the batch, a stateful operation
  ModelInput prepare_model_input(uint32_t num_decoding_tokens,
                                 uint32_t min_decoding_bach_size);

  // process the sample output for each sequence
  void process_sample_output(const SampleOutput& sample_output);

  // process the accepted output for each sequence
  void process_validate_output(const SampleOutput& sample_output);

  // set the engine type for the batch
  void set_engine_type(EngineType engine_type);

 private:
  static Token build_token(int64_t index,
                           torch::Tensor token_ids,
                           torch::Tensor logprobs,
                           torch::Tensor top_tokens,
                           torch::Tensor top_logprobs);
  // sequences in the batch
  std::vector<Sequence*> sequences_;

  // max number of tokens to process for each sequence
  // default to max value
  std::vector<uint32_t> token_budgets_;

  // number of used budget for each sequence
  std::vector<uint32_t> budget_used_;
};

}  // namespace llm
