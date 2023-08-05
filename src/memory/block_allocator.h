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
// memory usage and to allocate memory.
class Block final {
 public:
  ~Block();

  // copy constructor and assignment operator
  Block(const Block& other);
  Block& operator=(const Block& other);

  // move related operations
  Block(Block&& other) noexcept;
  Block& operator=(Block&& other) noexcept;

  // get the block id
  uint32_t id() const { return id_; }

  // get the block size in bytes
  uint32_t size() const { return size_; }

  // get the memory type
  MemoryType memory_type() const { return type_; }

  // get the reference count, 0 if the block is invalid after move
  uint32_t ref_count() const { return ref_count_ == nullptr ? 0 : *ref_count_; }

  // check if the block is shared
  bool is_shared() const { return ref_count() > 1; }

 private:
  friend class BlockAllocator;
  Block(MemoryType type, uint32_t id, uint32_t size, BlockAllocator* allocator);

  // memory type: CPU or GPU
  MemoryType type_;
  // block id
  uint32_t id_;
  // block size in bytes
  uint32_t size_;
  // reference count
  uint32_t* ref_count_ = nullptr;
  // allocator
  BlockAllocator* allocator_ = nullptr;
};

// BlockAllocator is used to track memory blocks. It is not thread safe.
// Please note: The actual memory has been allocated outside of this class. This
// class only manages the allocation and deallocation of memory block ids.
class BlockAllocator final {
 public:
  BlockAllocator(uint32_t num_cpu_blocks,
                 uint32_t num_gpu_blocks,
                 uint32_t block_size_in_bytes);

  ~BlockAllocator() = default;

  // disable copy and move operations
  BlockAllocator(const BlockAllocator&) = delete;
  BlockAllocator& operator=(const BlockAllocator&) = delete;
  BlockAllocator(BlockAllocator&&) = delete;
  BlockAllocator& operator=(BlockAllocator&&) = delete;

  // allocate a block of memory
  Block allocate(MemoryType type);

  // get block size in bytes
  uint32_t block_size_in_bytes() const { return block_size_in_bytes_; }

  // get number of free cpu blocks
  uint32_t free_cpu_block_count() const { return cpu_block_count_; }

  // get number of free gpu blocks
  uint32_t free_gpu_block_count() const { return gpu_block_count_; }

 private:
  friend class Block;
  // free a block of memory
  // should only be called by Block destructor implicitly
  void free(const Block& block);

  // free cpu block count
  uint32_t cpu_block_count_ = 0;

  // free gpu block count
  uint32_t gpu_block_count_ = 0;

  // block size in bytes
  uint32_t block_size_in_bytes_ = 0;

  // free cpu block list
  std::vector<uint32_t> free_cpu_blocks_;

  // free gpu block list
  std::vector<uint32_t> free_gpu_blocks_;
};

}  // namespace llm
