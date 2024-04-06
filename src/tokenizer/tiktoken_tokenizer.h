#pragma once

#include <absl/container/flat_hash_map.h>
#include <re2/re2.h>

#include <vector>

#include "tokenizer.h"
#include "tokenizer_args.h"

namespace llm {

// a simple c++ implementation of the openai/tiktoken
// https://github.com/openai/tiktoken
class TiktokenTokenizer : public Tokenizer {
 public:
  TiktokenTokenizer(const std::string_view& dir_path,
                    const TokenizerArgs& args);

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids) const override;

  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override;

  size_t vocab_size() const override;

  std::unique_ptr<Tokenizer> clone() const override;

 private:
  void load_special_tokens(const std::vector<std::string>& special_tokens,
                           int32_t start_id);

  void load_vocab(const std::string& vocab_file_path);

  void encode_internal(const std::string_view& text,
                       std::vector<int32_t>* ids) const;

  void byte_pair_encode(const std::string_view& piece,
                        std::vector<int32_t>* ids) const;

  std::optional<int32_t> token_to_id(const std::string_view& token) const;

  std::string dir_path_;

  TokenizerArgs args_;

  // token to ids
  absl::flat_hash_map<std::string, int32_t> encoder_;
  // id to token
  absl::flat_hash_map<int32_t, std::string> decoder_;

  // a regex pattern to tokenize text
  // N.B. RE2 doesn't support look-around assertions.
  // https://github.com/google/re2/wiki/Syntax
  std::unique_ptr<re2::RE2> regex_;

  // special tokens to ids
  absl::flat_hash_map<std::string, int32_t> special_token_encoder_;

  // special token ids to tokens
  absl::flat_hash_map<int32_t, std::string> special_token_decoder_;

  // special token regex (optional)
  std::unique_ptr<re2::RE2> special_token_regex_;

  // token ids to add to the beginning of the input sequence
  std::vector<int32_t> prefix_token_ids_;
};

}  // namespace llm
