#pragma once
#include <absl/container/flat_hash_map.h>
#include <re2/re2.h>

#include <cstdint>

#include "sentencepiece/sentencepiece_processor.h"
#include "tokenizer.h"

namespace llm {

// a tokenizer that uses google/SentencePiece
class SentencePieceTokenizer : public Tokenizer {
 public:
  SentencePieceTokenizer(const std::string& vocab_file_path,
                         const std::vector<std::string>& special_tokens,
                         bool prepend_bos);

  SentencePieceTokenizer(const std::string& vocab_file_path, bool prepend_bos)
      : SentencePieceTokenizer(vocab_file_path, {}, prepend_bos) {}

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids) const override;

  std::string decode(const std::vector<int32_t>& ids) const override;

  size_t vocab_size() const override;

  std::unique_ptr<Tokenizer> clone() const override;

 private:
  bool encode_internal(const std::string_view& text,
                       std::vector<int32_t>* ids) const;
  void decode_internal(const std::vector<int32_t>& ids,
                       size_t start,
                       size_t end,
                       std::stringstream* ss) const;
  std::string vocab_file_path_;

  std::vector<std::string> special_tokens_;

  sentencepiece::SentencePieceProcessor sp_processor_;

  // special tokens to ids
  absl::flat_hash_map<std::string, int32_t> special_token_encoder_;

  // special token ids to tokens
  absl::flat_hash_map<int32_t, std::string> special_token_decoder_;

  // special token regex (optional)
  std::unique_ptr<re2::RE2> special_token_regex_;

  bool prepend_bos_ = false;
};

}  // namespace llm
