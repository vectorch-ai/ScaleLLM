#include "block.h"

#include <glog/logging.h>

#include <cstdint>

#include "block_allocator.h"

namespace llm {

Block::Block(int32_t id) : Block(id, nullptr) {}

Block::Block(int32_t id, BlockAllocator* allocator)
    : id_(id), ref_count_(new uint32_t(1)), allocator_(allocator) {}

Block::~Block() {
  // decrease reference count
  dec_ref_count();
}

// copy constructor
Block::Block(const Block& other)
    : id_(other.id_),
      ref_count_(other.ref_count_),
      allocator_(other.allocator_) {
  // increase reference count
  inc_ref_count();
}

// copy assignment
Block& Block::operator=(const Block& other) {
  if (this != &other) {
    dec_ref_count();

    id_ = other.id_;
    allocator_ = other.allocator_;
    ref_count_ = other.ref_count_;
    inc_ref_count();
  }
  return *this;
}

Block::Block(Block&& other) noexcept
    : id_(other.id_),
      ref_count_(other.ref_count_),
      allocator_(other.allocator_) {
  // reset other without adjusting the reference count
  other.id_ = -1;
  other.ref_count_ = nullptr;
  other.allocator_ = nullptr;
}

Block& Block::operator=(Block&& other) noexcept {
  if (this != &other) {
    dec_ref_count();

    id_ = other.id_;
    allocator_ = other.allocator_;
    ref_count_ = other.ref_count_;

    other.id_ = -1;
    other.ref_count_ = nullptr;
    other.allocator_ = nullptr;
  }
  return *this;
}

uint32_t Block::size() const {
  return allocator_ == nullptr ? 0 : allocator_->block_size();
}

void Block::inc_ref_count() {
  if (ref_count_ != nullptr) {
    ++(*ref_count_);
  }
}

void Block::dec_ref_count() {
  if (ref_count_ != nullptr && --(*ref_count_) == 0) {
    // release the reference count memory
    delete ref_count_;
    // return the block id to the allocator
    if (allocator_ != nullptr) {
      allocator_->free(id_);
    }
  }
}

}  // namespace llm