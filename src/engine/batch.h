#pragma once

#include <torch/torch.h>

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
  Batch(Sequence* sequence) : sequences_{sequence} {}
  Batch(const std::vector<Sequence*>& sequences) : sequences_(sequences) {}

  // add sequence into the batch
  // caller should make sure the sequence is valid
  void add(Sequence* sequence);
  void add(const std::vector<Sequence*>& sequences);

  // get the number of sequences in the batch
  size_t size() const { return sequences_.size(); }
  bool empty() const { return sequences_.empty(); }

  // clear the batch for reuse
  void clear() { sequences_.clear(); }
  void reset() { sequences_.clear(); }

  // reset the batch with new sequences
  void reset(const std::vector<Sequence*>& sequences) {
    sequences_ = sequences;
  }

  // index operator
  // TODO: remove this operator once refactoring is done
  Sequence* operator[](size_t i) { return sequences_[i]; }

  // prepare inputs for the batch
  ModelInput prepare_model_inputs(int32_t block_size) const;

  // TODO: do we really need a sperate function for validate?
  ModelInput prepare_model_validate_inputs(int32_t block_size) const;

  // where to put outputs?
  // TODO: iterator sequence

 private:
  std::vector<Sequence*> sequences_;
};

}  // namespace llm
