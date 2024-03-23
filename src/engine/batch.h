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

  // add sequence into the batch
  // caller should make sure the sequence is valid
  void add(
      Sequence* sequence,
      uint32_t max_tokens_to_process = std::numeric_limits<uint32_t>::max());

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
  ModelInput prepare_model_inputs();

  // TODO: do we really need a sperate function for validate?
  ModelInput prepare_model_validate_inputs();

 private:
  // sequences in the batch
  std::vector<Sequence*> sequences_;

  // max number of tokens to process for each sequence
  // default to max value
  std::vector<uint32_t> max_tokens_to_process_;
};

}  // namespace llm
