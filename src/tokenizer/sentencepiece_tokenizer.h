#pragma once

#include "sentencepiece/sentencepiece_processor.h"
#include "tokenizer.h"

namespace llm {

// a tokenizer that uses google/SentencePiece
class SentencePieceTokenizer : public Tokenizer {
 public:
  explicit SentencePieceTokenizer(const std::string& vocab_file_path);

  bool encode(const std::string_view& text,
              std::vector<int>* ids) const override;

  std::string decode(const std::vector<int>& ids) const override;

  size_t vocab_size() const override { return sp_processor_.GetPieceSize(); }

  std::unique_ptr<Tokenizer> clone() const override;

 private:
  std::string vocab_file_path_;

  sentencepiece::SentencePieceProcessor sp_processor_;
};

}  // namespace llm
