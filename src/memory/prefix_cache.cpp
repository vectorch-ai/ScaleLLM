#include "prefix_cache.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <cstdint>
#include <vector>

#include "common/slice.h"

namespace llm {
namespace {
// get the lenght of common prefix of two token ids
template <typename VectorA, typename VectorB>
size_t common_prefix_length(const VectorA& token_ids1,
                            const VectorB& token_ids2) {
  size_t i = 0;
  while (i < token_ids1.size() && i < token_ids2.size() &&
         token_ids1[i] == token_ids2[i]) {
    ++i;
  }
  return i;
}

size_t round_down(size_t n, size_t multiple) {
  return (n / multiple) * multiple;
}

}  // namespace

PrefixCache::PrefixCache(uint32_t block_size) : block_size_(block_size) {
  CHECK(block_size_ > 0) << "Block size should be greater than 0";

  // initialize the lru list
  lru_front_.next = &lru_back_;
  lru_back_.prev = &lru_front_;
}

PrefixCache::~PrefixCache() {
  // iterator the lru list to release nodes
  size_t num_nodes = 0;
  Node* node = lru_front_.next;
  while (node != &lru_back_) {
    Node* next = node->next;
    delete node;
    node = next;
    ++num_nodes;
  }
  CHECK(num_nodes_ == num_nodes) << "detected memory leak";
}

// match the token ids with the prefix tree
// return matched blocks
std::vector<Block> PrefixCache::match(const Slice<int32_t>& token_ids) {
  const int64_t now = absl::ToUnixMicros(absl::Now());
  std::vector<Block> blocks;

  // allign tokens to block boundary
  const size_t n_tokens = round_down(token_ids.size(), block_size_);
  auto tokens_slice = token_ids.slice(0, n_tokens);

  size_t matched_tokens = 0;
  // start from the root node
  Node* next_node = &root_;
  while (next_node != nullptr && !tokens_slice.empty()) {
    Node* curr = next_node;
    next_node = nullptr;

    // match with children
    for (Node* child : curr->children) {
      size_t prefix_length =
          common_prefix_length(tokens_slice, child->token_ids);
      // truncate the prefix length at block boundary
      prefix_length = round_down(prefix_length, block_size_);

      // find a match
      if (prefix_length > 0) {
        // update the last access time and move the node to the back of the LRU
        child->last_access_time = now;
        move_node_to_lru_back(child);

        matched_tokens += prefix_length;

        // append the blocks to the result
        const size_t n_blocks = prefix_length / block_size_;
        blocks.insert(blocks.end(),
                      child->blocks.begin(),
                      child->blocks.begin() + n_blocks);
        tokens_slice = tokens_slice.slice(prefix_length);

        if (prefix_length == child->token_ids.size()) {
          // full match, continue to grand children
          next_node = child;
        }
        break;
      }
    }
  }

  return blocks;
}

// insert the token ids and blocks into the prefix tree
// return the length of new inserted tokens
size_t PrefixCache::insert(const Slice<int32_t>& token_ids,
                           const Slice<Block>& blocks) {
  const int64_t now = absl::ToUnixMicros(absl::Now());
  // allign tokens to block boundary
  const size_t n_blocks =
      std::min(token_ids.size() / block_size_, blocks.size());
  const size_t n_tokens = n_blocks * block_size_;

  // truncate the token ids and blocks to boundary
  auto tokens_slice = token_ids.slice(0, n_tokens);
  auto blocks_slice = blocks.slice(0, n_blocks);

  size_t new_inserted_tokens = 0;
  // start from the root node
  Node* next_node = &root_;
  while (next_node != nullptr && !tokens_slice.empty()) {
    Node* curr = next_node;
    next_node = nullptr;

    // match with children
    for (Node* child : curr->children) {
      size_t prefix_length =
          common_prefix_length(tokens_slice, child->token_ids);
      // we only cache a whole block, truncate the prefix length
      prefix_length = round_down(prefix_length, block_size_);

      // find a match
      if (prefix_length > 0) {
        // update the last access time and move the node to the back of the LRU
        child->last_access_time = now;
        move_node_to_lru_back(child);

        CHECK(prefix_length % block_size_ == 0)
            << "The prefix length should be multiple of block size";
        const size_t n_blocks = prefix_length / block_size_;
        // advance the token and block slices
        tokens_slice = tokens_slice.slice(prefix_length);
        blocks_slice = blocks_slice.slice(n_blocks);

        if (prefix_length < child->token_ids.size()) {
          // partial match, split the child node on the common prefix
          split_node(child, prefix_length);
        }
        next_node = child;
        break;
      }
    }

    // no child match, create a new child node
    if (next_node == nullptr) {
      create_child(curr, tokens_slice, blocks_slice, now);
      new_inserted_tokens += tokens_slice.size();
    }
  }
  return new_inserted_tokens;
}

// release the blocks hold by the prefix cache
size_t PrefixCache::evict(size_t n_blocks_to_evict) {
  size_t total_evicted = 0;
  // loop until no blocks to evict
  while (total_evicted < n_blocks_to_evict) {
    // conduct multiple round scaning to avoid invalidating leaf_nodes_ iterator
    const size_t evicted = evict_helper(n_blocks_to_evict - total_evicted);
    if (evicted == 0) {
      // no more cache to evict, just return
      break;
    }
    total_evicted += evicted;
  }
  return total_evicted;
}

size_t PrefixCache::evict_helper(size_t n_blocks_to_evict) {
  size_t total_evicted = 0;
  // evict nodes at the end to avoid invaliding iterator
  std::vector<Node*> nodes_to_evict;
  int64_t pre_access_time = 0;
  for (Node* node = lru_front_.next;
       total_evicted < n_blocks_to_evict && node != &lru_back_;
       node = node->next) {
    CHECK(pre_access_time <= node->last_access_time)
        << "The last access time should be in ascending order";
    pre_access_time = node->last_access_time;

    // skip non-leaf nodes
    if (!node->children.empty()) {
      continue;
    }

    // find first non-shared block to evict
    const auto& blocks = node->blocks;
    const size_t n_blocks = blocks.size();
    size_t non_shared_start = 0;
    for (; non_shared_start < n_blocks; ++non_shared_start) {
      if (!blocks[non_shared_start].is_shared()) {
        break;
      }
    }

    // try to only evict minimal number of blocks
    const size_t n_to_evict = std::min(n_blocks_to_evict - total_evicted,
                                       n_blocks - non_shared_start);
    total_evicted += n_to_evict;
    if (n_to_evict == n_blocks) {
      // mark the node as to be evicted
      nodes_to_evict.push_back(node);
    } else if (n_to_evict > 0) {
      // partially evict non-shared blocks
      const size_t n_blocks_left = n_blocks - n_to_evict;
      DCHECK(n_blocks_left >= non_shared_start);
      node->token_ids.resize(n_blocks_left * block_size_);
      node->blocks.resize(n_blocks_left);
    }
  }

  // release leaf nodes and update leaf_nodes_ set
  for (Node* node : nodes_to_evict) {
    release_node(node);
  }

  // update the number of blocks
  num_blocks_ -= total_evicted;
  return total_evicted;
}

void PrefixCache::release_node(Node* node) {
  DCHECK(node != &root_);
  DCHECK(node->children.empty()) << "should only release leaf node";
  // remove the node from the parent's children
  auto* parent = node->parent;
  DCHECK(parent->children.count(node) > 0);
  parent->children.erase(node);

  // delete the node
  remove_node_from_lru(node);
  delete node;
  --num_nodes_;
}

void PrefixCache::split_node(Node* node, size_t common_prefix_length) {
  CHECK(common_prefix_length > 0 && common_prefix_length % block_size_ == 0)
      << "The common prefix length should be greater than 0";
  const size_t n_blocks = common_prefix_length / block_size_;
  CHECK(node->token_ids.size() > common_prefix_length &&
        node->blocks.size() > n_blocks)
      << "The common prefix length should be less than the token ids length";

  // split the node at the common prefix
  Node* child = new Node();
  add_node_to_lru_back(child);
  ++num_nodes_;

  Slice<int32_t> token_ids(node->token_ids);
  Slice<Block> blocks(node->blocks);

  child->token_ids = token_ids.slice(common_prefix_length).to_vector();
  child->blocks = blocks.slice(n_blocks).to_vector();
  child->last_access_time = node->last_access_time;
  // point to parent
  child->parent = node;
  // take over children
  child->children = std::move(node->children);
  for (Node* grand_child : child->children) {
    grand_child->parent = child;
  }

  // truncate token_ids and blocks to the common prefix length
  node->token_ids.resize(common_prefix_length);
  node->blocks.resize(n_blocks);
  // put the new child into the children set
  node->children.insert(child);
}

void PrefixCache::create_child(Node* node,
                               const Slice<int32_t>& tokens,
                               const Slice<Block>& blocks,
                               int64_t now) {
  CHECK(!tokens.empty() && tokens.size() == blocks.size() * block_size_)
      << "The number of tokens "
         "should be equal to the number of blocks times block size";

  Node* child = new Node();
  add_node_to_lru_back(child);
  ++num_nodes_;

  num_blocks_ += blocks.size();

  child->token_ids = tokens.to_vector();
  child->blocks = blocks.to_vector();
  child->last_access_time = now;
  child->parent = node;
  node->children.insert(child);
}

// add a new node to the back of the LRU list
void PrefixCache::add_node_to_lru_back(Node* node) {
  node->prev = lru_back_.prev;
  node->next = &lru_back_;
  lru_back_.prev->next = node;
  lru_back_.prev = node;
}

void PrefixCache::remove_node_from_lru(Node* node) {
  node->prev->next = node->next;
  node->next->prev = node->prev;
}

// move the node to the back of the LRU list
void PrefixCache::move_node_to_lru_back(Node* node) {
  // remove the node from the current position
  remove_node_from_lru(node);
  // add the node to the back of the LRU list
  add_node_to_lru_back(node);
}

}  // namespace llm