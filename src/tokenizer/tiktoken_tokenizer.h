#pragma once

#include <absl/container/flat_hash_map.h>
#include <re2/re2.h>

#include <vector>

#include "tokenizer.h"

namespace llm {

// a simple c++ implementation of the openai/tiktoken
// https://github.com/openai/tiktoken
class TiktokenTokenizer : public Tokenizer {
 public:
  explicit TiktokenTokenizer(const std::string& vocab_file_path)
      : TiktokenTokenizer(vocab_file_path, {}) {}

  TiktokenTokenizer(const std::string& vocab_file_path,
                    const std::vector<std::string>& special_tokens);

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids) const override;

  std::string decode(const std::vector<int32_t>& ids) const override;

  size_t vocab_size() const override;

  std::unique_ptr<Tokenizer> clone() const override;

 private:
  void load_vocab(const std::string& vocab_file_path);
  
  void byte_pair_encode(const std::string_view& piece,
                        std::vector<int32_t>* ids) const;

  std::string vocab_file_path_;

  std::vector<std::string> special_tokens_;

  // token to ids
  absl::flat_hash_map<std::string, int32_t> encoder_;
  // id to token
  absl::flat_hash_map<int32_t, std::string> decoder_;

  // special tokens to ids
  absl::flat_hash_map<std::string, int32_t> special_token_encoder_;

  // special token ids to tokens
  absl::flat_hash_map<int32_t, std::string> special_token_decoder_;

  // special token regex (optional)
  std::unique_ptr<re2::RE2> special_token_regex_;
};

}  // namespace llm
