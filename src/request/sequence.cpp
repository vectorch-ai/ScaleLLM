#include "sequence.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tokenizer/tokenizer.h"

namespace llm {

std::atomic<int64_t> Sequence::next_id_{1};

}  // namespace llm
