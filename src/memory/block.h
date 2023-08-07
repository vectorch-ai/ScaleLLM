#pragma once

#include <glog/logging.h>

#include <cstdint>
#include <vector>

namespace llm {

enum class MemoryType {
  kCPU,
  kGPU,
};

// forward declaration
class BlockAllocator;

// Memory block represents a contiguous memory region. It is used to track
// memory usage.
class Block final {
 public:
  ~Block();

  // copy constructor and assignment operator
  Block(const Block& other);
  Block& operator=(const Block& other);

  // move related operations
  Block(Block&& other) noexcept;
  Block& operator=(Block&& other) noexcept;

  // get the memory type
  MemoryType memory_type() const { return type_; }

  // get the block id
  uint32_t id() const { return id_; }

  // get the block size in bytes
  uint32_t size() const;

  // get the reference count, 0 if the block is invalid after move
  uint32_t ref_count() const { return ref_count_ == nullptr ? 0 : *ref_count_; }

  // check if the block is shared
  bool is_shared() const { return ref_count() > 1; }

 private:
  friend class BlockAllocator;
  Block(MemoryType type, uint32_t id, BlockAllocator* allocator);

  // memory type: CPU or GPU
  MemoryType type_;
  // block id
  uint32_t id_;
  // reference count
  uint32_t* ref_count_ = nullptr;
  // allocator that manages this block
  BlockAllocator* allocator_ = nullptr;
};

}  // namespace llm
