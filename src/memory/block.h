#pragma once

#include <cstdint>

namespace llm {

// forward declaration
class BlockAllocator;

// Memory block represents a contiguous memory region.
// It is used to track memory usage. the block will be released when the
// reference count drops to zero.
class Block final {
 public:
  ~Block();

  // add reference count to allow using with std::vector
  Block() = default;

  // used for testing
  Block(int32_t id);

  Block(int32_t id, BlockAllocator* allocator);

  // copy constructor and assignment operator
  Block(const Block& other);
  Block& operator=(const Block& other);

  // move related operations
  Block(Block&& other) noexcept;
  Block& operator=(Block&& other) noexcept;

  // get the block id
  int32_t id() const { return id_; }

  // get the block size
  uint32_t size() const;

  // get the reference count, 0 if the block is invalid after move
  uint32_t ref_count() const { return ref_count_ == nullptr ? 0 : *ref_count_; }

  // check if the block is shared
  bool is_shared() const { return ref_count() > 1; }

  // check if the block is valid
  bool is_valid() const { return id_ >= 0 && ref_count_ != nullptr; }

  // equeal operator
  bool operator==(const Block& other) const {
    return id_ == other.id_ && ref_count_ == other.ref_count_ &&
           allocator_ == other.allocator_;
  }

 private:
  // increase reference count
  void inc_ref_count();

  // decrease reference count
  void dec_ref_count();

  // block id
  int32_t id_ = -1;
  // reference count
  uint32_t* ref_count_ = nullptr;
  // allocator that manages this block
  BlockAllocator* allocator_ = nullptr;
};

}  // namespace llm