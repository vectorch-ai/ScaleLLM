#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "block.h"
#include "common/slice.h"

namespace llm {

class PrefixCache final {
 public:
  explicit PrefixCache(uint32_t block_size);

  ~PrefixCache();

  // disable copy, move and assign
  PrefixCache(const PrefixCache&) = delete;
  PrefixCache(PrefixCache&&) = delete;
  PrefixCache& operator=(const PrefixCache&) = delete;
  PrefixCache& operator=(PrefixCache&&) = delete;

  // match the token ids with the prefix tree
  // return matched blocks
  std::vector<Block> match(const std::vector<int32_t>& token_ids) {
    return match(Slice<int32_t>(token_ids));
  }
  std::vector<Block> match(const Slice<int32_t>& token_ids);

  // insert the token ids and blocks into the prefix tree
  // return the length of new inserted tokens
  size_t insert(const std::vector<int32_t>& token_ids,
                const std::vector<Block>& blocks) {
    return insert(Slice<int32_t>(token_ids), Slice<Block>(blocks));
  }
  size_t insert(const Slice<int32_t>& token_ids, const Slice<Block>& blocks);

  // evict blocks hold by the prefix cache
  // return the actual number of evicted blocks
  size_t evict(size_t n_blocks);

  // get the number of blocks in the prefix cache
  size_t num_blocks() const { return num_blocks_; }

  // get the total number of nodes in the prefix tree
  size_t num_nodes() const { return nodes_.size(); }

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

    // the previous and next nodes, used to maintain the LRU list
    Node* prev = nullptr;
    Node* next = nullptr;
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

  // delete the node from the LRU list
  static void remove_node_from_lru(Node* node);

  // add a new node to the back of the LRU list
  void add_node_to_lru_back(Node* node);

  // move the node to the back of the LRU list
  void move_node_to_lru_back(Node* node);

  std::unordered_set<Node*> nodes_;

  // the root node of the prefix tree
  Node root_;

  // the front and back nodes of the LRU list
  // the back node is the least recently used node
  // sorted by the last access time in ascending order
  Node lru_front_;
  Node lru_back_;

  // the block size of the memory blocks
  uint32_t block_size_;

  // the total number of blocks in the prefix cache
  size_t num_blocks_ = 0;
};

}  // namespace llm