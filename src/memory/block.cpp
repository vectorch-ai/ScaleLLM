#include "block_allocator.h"

#include <glog/logging.h>

#include <cstdint>
#include <vector>

namespace llm {

// Memory block represents a contiguous memory region. It is used to track
// memory usage.

Block::Block(MemoryType type,
             uint32_t id,
             BlockAllocator* allocator)
    : type_(type),
      id_(id),
      ref_count_(new uint32_t(1)),
      allocator_(allocator) {}

Block::~Block() {
  // decrease reference count and free memory if necessary
  if (ref_count_ != nullptr && --(*ref_count_) == 0) {
    delete ref_count_;
    allocator_->free(*this);
  }
}

// copy constructor
Block::Block(const Block& other)
    : type_(other.type_),
      id_(other.id_),
      ref_count_(other.ref_count_),
      allocator_(other.allocator_) {
  // increase reference count
  ++(*ref_count_);
}

// copy assignment
Block& Block::operator=(const Block& other) {
  if (this != &other) {
    if (ref_count_ != nullptr && --(*ref_count_) == 0) {
      delete ref_count_;
      allocator_->free(*this);
    }
    type_ = other.type_;
    id_ = other.id_;
    allocator_ = other.allocator_;
    ref_count_ = other.ref_count_;
    ++(*ref_count_);
  }
  return *this;
}

Block::Block(Block&& other) noexcept
    : type_(other.type_),
      id_(other.id_),
      ref_count_(other.ref_count_),
      allocator_(other.allocator_) {
  // reset other
  other.ref_count_ = nullptr;
  other.allocator_ = nullptr;
}

Block& Block::operator=(Block&& other) noexcept {
  if (this != &other) {
    // decrease reference count and free memory if necessary
    if (ref_count_ != nullptr && --(*ref_count_) == 0) {
      delete ref_count_;
      allocator_->free(*this);
    }

    type_ = other.type_;
    id_ = other.id_;
    allocator_ = other.allocator_;
    ref_count_ = other.ref_count_;

    other.ref_count_ = nullptr;
    other.allocator_ = nullptr;
  }
  return *this;
}

uint32_t Block::size() const {
  return allocator_ == nullptr ? 0 : allocator_->block_size_in_bytes();
}

}  // namespace llm
