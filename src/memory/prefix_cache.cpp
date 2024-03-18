#include "prefix_cache.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <cstdint>
#include <set>
#include <vector>

#include "common/slice.h"

namespace llm {
namespace {
// get the lenght of common prefix of two token ids
template <typename VectorA, typename VectorB>
uint32_t common_prefix_length(const VectorA& token_ids1,
                              const VectorB& token_ids2) {
  uint32_t i = 0;
  while (i < token_ids1.size() && i < token_ids2.size() &&
         token_ids1[i] == token_ids2[i]) {
    ++i;
  }
  return i;
}
}  // namespace

PrefixCache::PrefixCache(uint32_t block_size) : block_size_(block_size) {}

// match the token ids with the prefix tree and return matched block length
uint32_t PrefixCache::match(const std::vector<int32_t>& token_ids,
                            std::vector<Block>* blocks) {
  int64_t now = absl::ToUnixMicros(absl::Now());
  auto tokens = Slice(token_ids);

  uint32_t matched_length = 0;
  Node* next_node = &root_;
  while (next_node != nullptr) {
    Node* curr = next_node;
    next_node = nullptr;

    // match with children
    for (Node* child : curr->children) {
      uint32_t prefix_length = common_prefix_length(tokens, child->token_ids);
      // we only cache a whole block, truncate the prefix length
      prefix_length = prefix_length / block_size_ * block_size_;

      if (prefix_length == 0) {
        // no common prefix, continue to the next child
        continue;
      }
      CHECK(prefix_length % block_size_ == 0)
          << "The prefix length should be multiple of block size";

      // find a match, update the last access time
      child->last_access_time = now;
      matched_length += prefix_length;

      if (prefix_length == child->token_ids.size()) {
        // full match, continue to grand children
        blocks->insert(
            blocks->end(), child->blocks.begin(), child->blocks.end());
        tokens = tokens.sub(prefix_length);
        next_node = child;
      } else {
        // partial match, add the blocks to the result
        const uint32_t n_blocks = prefix_length / block_size_;
        blocks->insert(blocks->end(),
                       child->blocks.begin(),
                       child->blocks.begin() + n_blocks);
      }
      break;
    }
  }

  return matched_length;
}

// insert the token ids and block ids into the prefix tree
// return the length of blocks inserted
void PrefixCache::insert(const std::vector<int32_t>& token_ids,
                         const std::vector<Block>& blocks) {
  int64_t now = absl::ToUnixMicros(absl::Now());
  auto tokens_slice = Slice(token_ids);
  auto blocks_slice = Slice(blocks);

  Node* next_node = &root_;
  while (next_node != nullptr) {
    Node* curr = next_node;
    next_node = nullptr;

    // match with children
    for (Node* child : curr->children) {
      uint32_t prefix_length =
          common_prefix_length(tokens_slice, child->token_ids);
      // we only cache a whole block, truncate the prefix length
      prefix_length = prefix_length / block_size_ * block_size_;

      if (prefix_length == 0) {
        // no common prefix, continue to the next child
        continue;
      }
      CHECK(prefix_length % block_size_ == 0)
          << "The prefix length should be multiple of block size";
      const uint32_t n_blocks = prefix_length / block_size_;

      // find a match, update the last access time
      child->last_access_time = now;
      // advance the token and block slices
      tokens_slice = tokens_slice.sub(prefix_length);
      blocks_slice = blocks_slice.sub(n_blocks);

      // a full match, continue to grand children
      if (prefix_length == child->token_ids.size()) {
        next_node = child;
      } else {
        // partial match, split the child node on the common prefix
        split_node(child, prefix_length);
        // create new child node with the remaining token ids
        create_child(child, tokens_slice, blocks_slice, now);
      }
      break;
    }

    // no match at all, create a new child node
    create_child(curr, tokens_slice, blocks_slice, now);
  }
}

// release the blocks hold by the prefix cache
uint32_t PrefixCache::release(uint32_t n_blocks, std::vector<Block>* blocks) {
  // evict the least recently used blocks
  uint32_t n_released_blocks = 0;
  std::vector<Node*> nodes_to_release;
  for (auto it = leaf_nodes_.begin(); it != leaf_nodes_.end(); ++it) {
    CHECK((*it)->children.empty()) << "Leaf node should not have children";

    // check if any of the blocks in the node is still in use
    for (const auto& block : (*it)->blocks) {
      // can't release node if any block is in still in use
      if (block.is_shared()) {
        continue;
      }
    }
    // TODO: release partial blocks if the node has more blocks than needed

    // release all blocks in the node
    n_released_blocks += (*it)->blocks.size();
    nodes_to_release.push_back(*it);

    if (n_released_blocks >= n_blocks) {
      break;
    }
  }

  for (auto* node : nodes_to_release) {
    blocks->insert(blocks->end(), node->blocks.begin(), node->blocks.end());
    leaf_nodes_.erase(node);
    release_node(node);
  }
  return n_released_blocks;
}

void PrefixCache::release_node(Node* node) {
  auto* parent = node->parent;
  // remove the node from the parent's children
  parent->children.erase(node);
  if (parent->children.empty()) {
    // the parent becomes a leaf node
    leaf_nodes_.insert(parent);
  }
  // release the node
  delete node;
}

void PrefixCache::split_node(Node* node, uint32_t common_prefix_length) {
  // split the node at the common prefix
  Node* child = new Node();
  const uint32_t n_blocks = common_prefix_length / block_size_;
  CHECK(node->token_ids.size() > common_prefix_length)
      << "The common prefix length should be less than the token ids length";
  CHECK(node->blocks.size() > n_blocks)
      << "The common prefix length should be less than the block ids length";

  Slice<int32_t> token_ids = node->token_ids;
  Slice<Block> blocks = node->blocks;

  child->token_ids = token_ids.sub(common_prefix_length).to_vector();
  child->blocks = blocks.sub(n_blocks).to_vector();
  child->last_access_time = node->last_access_time;
  child->parent = node;
  // take over the children of the old node
  child->children = std::move(node->children);

  // truncate token_ids and blocks to the common prefix length
  node->token_ids.resize(common_prefix_length);
  node->blocks.resize(n_blocks);
  // put the new child into the children set
  node->children.insert(child);

  // remove the old leaf and add new leaf
  if (child->children.empty()) {
    leaf_nodes_.erase(node);
    leaf_nodes_.insert(child);
  }
}

void PrefixCache::create_child(Node* node,
                               const Slice<int32_t>& tokens,
                               const Slice<Block>& blocks,
                               int64_t now) {
  Node* child = new Node();
  child->token_ids = tokens.to_vector();
  child->blocks = blocks.to_vector();
  child->last_access_time = now;
  child->parent = node;
  node->children.insert(child);
  leaf_nodes_.insert(child);
}

}  // namespace llm