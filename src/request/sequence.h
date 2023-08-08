#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "slice.h"

namespace llm {

// The sequence encapsulates all the necessary
// information for a sequence, including the prompt, the token ids, and the
// current position in generating tokens, etc.
struct Sequence {
  // whether the sequence is in prefill stage
  bool is_prefill() const { return pos == 0; }

  // // get token ids
  // const std::vector<int>& get_token_ids() const { 
  //   return token_ids;
  // }

  // add a new token id to the sequence
  void append_new_token_id(int token_id) {
    // all tokens before pos should be processed and cached. 
    pos = token_ids.size();
    token_ids.push_back(token_id); 
  }

  // add new cache blocks
  void add_blocks(const std::vector<uint32_t>& new_blocks) {
    blocks.insert(blocks.end(), new_blocks.begin(), new_blocks.end());
  }

  // release all cache blocks
  std::vector<uint32_t> release_blocks() {
    // reset the current pos to 0 so that the cache can be recomputed next time
    pos = 0;
    return std::move(blocks);
  }

  // get the number of cache blocks to allocate for the sequence
  size_t num_blocks_to_allocate(uint32_t slots_per_block) const {
    if (is_finished) {
      // no need to allocate more blocks for a finished sequence
      return 0;
    }
    const uint32_t num_slots = token_ids.size();
    // round up to the nearest block number
    const uint32_t num_blocks =
        (num_slots + slots_per_block - 1) / slots_per_block;
    if (num_blocks <= blocks.size()) {
      // enough slots, don't need to allocate more
      return 0;
    }
    return num_blocks - blocks.size();
  }

 public:
  // prompt to generate completions for
  std::string prompt;

  // private:
  // token ids generated from p
  std::vector<int> token_ids;

  // the length of the prompt tokens
  size_t prompt_len = 0;

  // the position start to process in next step.
  // all tokens before pos should be processed and cached.
  size_t pos = 0;

  // physical block ids that hold the keys and values cache.
  std::vector<uint32_t> blocks;

  // has the sequence been finished
  bool is_finished = false;

  // TODO: Add logits results.

  // TODO: sampling related
};

}  // namespace llm
