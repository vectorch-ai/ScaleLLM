#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "memory/block.h"

namespace llm {

// The sequence encapsulates all the necessary
// information for a sequence, including the prompt, the token ids, and the
// current position in generating tokens, etc.
class Sequence {
 public:
  // TODO: add functions to access following private members.

  // private:
  // token ids generated from p
  std::vector<int> token_ids;

  // the length of the prompt
  int prompt_len = 0;

  // the current position in generating tokens
  int cur_pos = 0;

  // physical blocks that hold the keys and values cache.
  std::vector<Block> blocks;

  // has the sequence been finished
  bool is_finished = false;

  // TODO: Add logits results.

  // TODO: cache related

  // TODO: sampling related
};

}  // namespace llm
