#pragma once

#include <cstdint>
#include <set>
#include <unordered_set>
#include <vector>

#include "block.h"
#include "common/slice.h"

namespace llm {

class PrefixCache final {
 public:
  PrefixCache(uint32_t block_size);

  // match the token ids with the prefix tree and return matched block length
  uint32_t match(const std::vector<int32_t>& token_ids,
                 std::vector<Block>* blocks);

  // insert the token ids and block ids into the prefix tree
  // return the length of blocks inserted
  void insert(const std::vector<int32_t>& token_ids,
              const std::vector<Block>& blocks);

  // release the blocks hold by the prefix cache
  uint32_t release(uint32_t n_blocks, std::vector<Block>* blocks);

  // get the number of blocks in the prefix cache
  uint32_t num_blocks() const { return num_nodes() * block_size_; }

  // get the number of leaf nodes in the prefix tree
  uint32_t num_leaf_nodes() const { return leaf_nodes_.size(); }

  // get the total number of nodes in the prefix tree
  uint32_t num_nodes() const { return num_nodes_; }

 private:
  struct Node {
    // the token ids that the node represents
    // assert(token_ids.size() == blocks.size() * block_size)
    std::vector<int32_t> token_ids;
    // the block ids that the node represents
    std::vector<Block> blocks;
    // the children nodes, used to traverse down the tree
    std::unordered_set<Node*> children;
    // the parent node, used to traverse up the tree
    Node* parent = nullptr;
    // the last access time of the node, used to evict blocks
    int64_t last_access_time = 0;
  };

  // Define comparison operator for sorting in set
  struct Greater {
    bool operator()(const Node* lhs, const Node* rhs) const {
      return lhs->last_access_time > rhs->last_access_time;
    }
  };

  // release the node and update leaf_nodes_
  void release_node(Node* node);

  // split the node on the common prefix
  void split_node(Node* node, uint32_t common_prefix_length);

  // create a new child node under the node
  void create_child(Node* node,
                    const Slice<int32_t>& tokens,
                    const Slice<Block>& blocks,
                    int64_t now);

  // the leaf nodes in the prefix tree, used to evict blocks
  // sorted by last_access_time
  std::set<Node*, Greater> leaf_nodes_;

  // the root node of the prefix tree
  Node root_;

  // the block size of the memory blocks
  uint32_t block_size_;

  // the number of nodes in the prefix tree, excluding the root node
  uint32_t num_nodes_ = 0;
};

}  // namespace llm