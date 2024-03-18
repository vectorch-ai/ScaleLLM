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

PrefixCache::PrefixCache(uint32_t block_size) : block_size_(block_size) {
  CHECK(block_size_ > 0) << "Block size should be greater than 0";
}

// match the token ids with the prefix tree
// return the length of matched tokens
uint32_t PrefixCache::match(const std::vector<int32_t>& token_ids,
                            std::vector<Block>* blocks) {
  const int64_t now = absl::ToUnixMicros(absl::Now());

  // allign tokens to block boundary
  auto tokens_slice = Slice(token_ids).align_to(block_size_);
  uint32_t matched_tokens = 0;
  Node* next_node = &root_;
  while (next_node != nullptr && !tokens_slice.empty()) {
    Node* curr = next_node;
    next_node = nullptr;

    // match with children
    for (Node* child : curr->children) {
      uint32_t prefix_length =
          common_prefix_length(tokens_slice, child->token_ids);
      // truncate the prefix length at block boundary
      prefix_length = prefix_length / block_size_ * block_size_;

      if (prefix_length == 0) {
        // no common prefix, continue to the next child
        continue;
      }
      CHECK(prefix_length % block_size_ == 0)
          << "The prefix length should be multiple of block size";

      // find a match, update the last access time
      child->last_access_time = now;
      matched_tokens += prefix_length;

      if (prefix_length == child->token_ids.size()) {
        // full match, continue to grand children
        blocks->insert(
            blocks->end(), child->blocks.begin(), child->blocks.end());
        tokens_slice = tokens_slice.sub(prefix_length);
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

  return matched_tokens;
}

// insert the token ids and blocks into the prefix tree
// return the length of shared tokens
uint32_t PrefixCache::insert(const std::vector<int32_t>& token_ids,
                             const std::vector<Block>& blocks) {
  CHECK(blocks.size() * block_size_ >= token_ids.size())
      << "The number of blocks should be greater than or equal to the number "
         "of tokens";

  const int64_t now = absl::ToUnixMicros(absl::Now());
  // allign tokens to block boundary
  auto tokens_slice = Slice(token_ids).align_to(block_size_);
  auto blocks_slice = Slice(blocks);
  uint32_t shared_tokens = 0;
  Node* next_node = &root_;
  while (next_node != nullptr && !tokens_slice.empty()) {
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

      // find a match, update the last access time
      child->last_access_time = now;
      shared_tokens += prefix_length;

      CHECK(prefix_length % block_size_ == 0)
          << "The prefix length should be multiple of block size";
      const uint32_t n_blocks = prefix_length / block_size_;

      // advance the token and block slices
      tokens_slice = tokens_slice.sub(prefix_length);
      blocks_slice = blocks_slice.sub(n_blocks);

      if (prefix_length == child->token_ids.size()) {
        // a full match, continue to grand children
        next_node = child;
      } else if (!tokens_slice.empty()) {
        // partial match, and still have tokens left
        // split the child node on the common prefix
        split_node(child, prefix_length);
        // create new child with the remaining token ids
        create_child(child, tokens_slice, blocks_slice, now);
      }
      break;
    }

    // no match at all, create a new child node
    create_child(curr, tokens_slice, blocks_slice, now);
  }
  return shared_tokens;
}

// release the blocks hold by the prefix cache
uint32_t PrefixCache::evict(uint32_t n_blocks) {
  // evict the least recently used blocks
  uint32_t n_released_blocks = 0;
  std::vector<Node*> nodes_to_release;
  for (auto it = leaf_nodes_.begin();
       n_released_blocks < n_blocks && it != leaf_nodes_.end();
       ++it) {
    auto* leaf_node = *it;
    CHECK(leaf_node->children.empty()) << "Leaf node should not have children";

    // check if any of the blocks in the node is still in use
    const bool has_shared_block =
        std::any_of(leaf_node->blocks.begin(),
                    leaf_node->blocks.end(),
                    [](const Block& block) { return block.is_shared(); });
    // can't release node if any block is still in use
    if (has_shared_block) {
      continue;
    }

    // evict all blocks in the node
    n_released_blocks += leaf_node->blocks.size();
    nodes_to_release.push_back(leaf_node);
  }

  std::for_each(nodes_to_release.begin(),
                nodes_to_release.end(),
                [this](Node* node) { release_node(node); });

  return n_released_blocks;
}

void PrefixCache::release_node(Node* node) {
  // remove the node from the leaf nodes
  leaf_nodes_.erase(node);
  // remove the node from the parent's children
  auto* parent = node->parent;
  parent->children.erase(node);
  if (parent->children.empty()) {
    // the parent becomes a leaf node
    leaf_nodes_.insert(parent);
  }

  // delete the node
  delete node;
  CHECK(num_nodes_ > 0) << "The number of nodes should be greater than 0";
  --num_nodes_;
}

void PrefixCache::split_node(Node* node, uint32_t common_prefix_length) {
  // split the node at the common prefix
  Node* child = new Node();
  ++num_nodes_;

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
  ++num_nodes_;

  child->token_ids = tokens.to_vector();
  child->blocks = blocks.to_vector();
  child->last_access_time = now;
  child->parent = node;
  node->children.insert(child);
  leaf_nodes_.insert(child);
}

}  // namespace llm