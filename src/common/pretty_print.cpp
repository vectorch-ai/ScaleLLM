#include "pretty_print.h"

#include <array>
#include <iomanip>
#include <sstream>

namespace llm {

std::string readable_size(size_t bytes) {
  const std::array<const char*, 5> suffixes = {"B", "KB", "MB", "GB", "TB"};
  double size = static_cast<double>(bytes);
  size_t suffix_index = 0;
  while (size >= 1024 && suffix_index < 4) {
    size /= 1024;
    ++suffix_index;
  }
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << size << " "
         << suffixes.at(suffix_index);
  return stream.str();
}

}  // namespace llm
