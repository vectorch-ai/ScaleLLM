#include "json_reader.h"

#include <glog/logging.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
namespace llm {

bool JsonReader::parse(const std::string& json_file_path) {
  if (!std::filesystem::exists(json_file_path)) {
    return false;
  }

  std::ifstream ifs(json_file_path);
  if (!ifs.is_open()) {
    return false;
  }

  data_ = nlohmann::json::parse(ifs);
  return true;
}

bool JsonReader::contains(const std::string_view& key) const {
  // slipt the key by '.' then traverse the json object
  std::vector<std::string> keys = absl::StrSplit(key, '.');
  nlohmann::json data = data_;
  for (const auto& k : keys) {
    if (!data.contains(k)) {
      return false;
    }
    data = data[k];
  }
  return true;
}

}  // namespace llm