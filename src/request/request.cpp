#include "request.h"

#include <uuid.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sampling_parameter.h"
#include "sequence.h"
#include "status.h"
#include "stopping_criteria.h"

namespace llm {
namespace {
std::string generate_request_id() {
  return "cmpl-" + uuids::to_string(uuids::uuid_system_generator{}());
}
}  // namespace

Request::Request() : id(generate_request_id()) {}

void Request::add_sequence(std::string prompt,
                           std::vector<int32_t> token_ids,
                           OnStream on_stream) {
  sequences.emplace_back(std::move(prompt),
                         std::move(token_ids),
                         &sampling_param,
                         &stopping_criteria,
                         on_stream,
                         echo);
}

bool Request::is_finished() const {
  for (const auto& seq : sequences) {
    if (!seq.is_finished()) {
      return false;
    }
  }
  return true;
}
}  // namespace llm
