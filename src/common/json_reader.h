#pragma once
#include <absl/strings/str_split.h>

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace llm {

// an thin wrapper around nlohmann/json to read json files.
// it supports read keys with dot notation from json.
// for exmaple: value_or("a.b.c", 0) will return 100 for following json:
// {
//   "a": {
//     "b": {
//       "c": 100
//     }
//   }
// }
//
class JsonReader {
 public:
  // parse the json file, return true if success
  bool parse(const std::string& json_file_path);

  // check if the json contains the key, key can be nested with dot notation
  bool contains(const std::string& key) const;

  template <typename T, typename T2>
  T value_or(const std::vector<std::string>& keys,
             T2 default_value) const {
    for (const auto& key : keys) {
      if (auto data = value<T>(key)) {
        return data.value();
      }
    }
    // may introduce implicit conversion from T2 to T
    return default_value;
  }

  template <typename T, typename T2>
  T value_or(const std::string& key, T2 default_value) const {
    if (auto data = value<T>(key)) {
      return data.value();
    }
    // may introduce implicit conversion from T2 to T
    return default_value;
  }

  template <typename T>
  std::optional<T> value(const std::string& key) const {
    // slipt the key by '.' then traverse the json object
    const std::vector<std::string> keys = absl::StrSplit(key, '.');
    nlohmann::json data = data_;
    for (const auto& k : keys) {
      if (data.contains(k)) {
        data = data[k];
      } else {
        return std::nullopt;
      }
    }
    if (!data.is_null()) {
      return data.get<T>();
    }
    return std::nullopt;
  }

 private:
  nlohmann::json data_;
};

}  // namespace llm