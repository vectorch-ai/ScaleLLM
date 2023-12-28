#pragma once

#include <unordered_map>
#include <vector>

#include "tokenizer.h"

namespace llm {

// a simple c++ implementation of the openai/tiktoken
// https://github.com/openai/tiktoken
class TiktokenTokenizer : public Tokenizer {
 public:
  explicit TiktokenTokenizer(const std::string& vocab_file_path);

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids) const override;

  std::string decode(const std::vector<int32_t>& ids) const override;

  size_t vocab_size() const override;

  std::unique_ptr<Tokenizer> clone() const override;

 private:
  void byte_pair_encode(const std::string_view& piece,
                        std::vector<int32_t>* ids) const;

  std::string vocab_file_path_;

  // token to ids
  std::unordered_map<std::string, int32_t> encoder_;
  // id to token
  std::unordered_map<int32_t, std::string> decoder_;

  // TODO: add special token support
};

}  // namespace llm
