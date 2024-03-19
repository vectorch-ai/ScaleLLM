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

  // match the token ids with the prefix tree
  // return the length of matched tokens
  size_t match(const std::vector<int32_t>& token_ids,
               std::vector<Block>* blocks);

  // insert the token ids and blocks into the prefix tree
  // return the length of new inserted tokens
  size_t insert(const std::vector<int32_t>& token_ids,
                const std::vector<Block>& blocks);

  // evict blocks hold by the prefix cache
  // return the actual number of blocks evicted
  size_t evict(size_t n_blocks);

  // get the number of blocks in the prefix cache
  size_t num_blocks() const { return num_blocks_; }

  // get the number of leaf nodes in the prefix tree
  size_t num_leaf_nodes() const { return leaf_nodes_.size(); }

  // get the total number of nodes in the prefix tree
  size_t num_nodes() const { return num_nodes_; }

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

  // release the node and update leaf_nodes_
  void release_node(Node* node);

  // split the node on the common prefix
  void split_node(Node* node, size_t common_prefix_length);

  // create a new child node under the node
  void create_child(Node* node,
                    const Slice<int32_t>& tokens,
                    const Slice<Block>& blocks,
                    int64_t now);

  size_t evict_helper(size_t n_blocks);

  bool is_leaf_node(Node* node) const;

  // the leaf nodes in the prefix tree, used to evict blocks
  // TODO: add a LRU policy to evict blocks
  std::set<Node*> leaf_nodes_;

  // the root node of the prefix tree
  Node root_;

  // the block size of the memory blocks
  uint32_t block_size_;

  // the number of nodes in the prefix tree, excluding the root node
  size_t num_nodes_ = 0;

  // the total number of blocks in the prefix cache
  size_t num_blocks_ = 0;
};

}  // namespace llm