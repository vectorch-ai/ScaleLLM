#pragma once

#include <sentencepiece_processor.h>

#include "tokenizer.h"

namespace llm {

// a tokenizer that uses google/SentencePiece
class SentencePieceTokenizer : public Tokenizer {
 public:
  explicit SentencePieceTokenizer(const std::string& model_path);

  std::vector<int> encode(const std::string_view& text) const override;

  std::string decode(const std::vector<int>& ids) const override;

  int n_words() const { return sp_processor_.GetPieceSize(); }

 private:
  sentencepiece::SentencePieceProcessor sp_processor_;
};

}  // namespace llm
