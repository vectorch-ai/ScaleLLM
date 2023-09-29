#include "pretty_print.h"

#include <array>
#include <iomanip>
#include <sstream>

namespace llm {

std::string readable_size(size_t bytes) {
  static const std::array<const char*, 5> suffixes = {
      "B", "KB", "MB", "GB", "TB"};
  const size_t bytes_in_kb = 1024;
  double size = static_cast<double>(bytes);
  size_t suffix_index = 0;
  while (size >= bytes_in_kb && suffix_index < suffixes.size() - 1) {
    size /= bytes_in_kb;
    ++suffix_index;
  }
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << size << " "
         << suffixes.at(suffix_index);
  return stream.str();
}

}  // namespace llm
