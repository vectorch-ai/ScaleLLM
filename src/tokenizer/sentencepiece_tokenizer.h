#pragma once

#include <sentencepiece_processor.h>

#include "tokenizer.h"

namespace llm {

// a tokenizer that uses google/SentencePiece
class SentencePieceTokenizer : public Tokenizer {
 public:
  explicit SentencePieceTokenizer(const std::string& model_path);

  std::vector<int> Encode(const std::string_view& text) const override;

  std::string Decode(const std::vector<int>& ids) const override;

 private:
  sentencepiece::SentencePieceProcessor sp_processor_;
};

}  // namespace llm
