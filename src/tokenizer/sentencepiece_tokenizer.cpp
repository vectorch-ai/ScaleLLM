#include "sentencepiece_tokenizer.h"

#include "sentencepiece/sentencepiece_processor.h"
#include <glog/logging.h>

namespace llm {

SentencePieceTokenizer::SentencePieceTokenizer(const std::string& model_path) {
  const auto status = sp_processor_.Load(model_path);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to load SentencePiece model from " << model_path
               << ": " << status.ToString() << ", error " << status.ToString();
  }
}

std::vector<int> SentencePieceTokenizer::encode(
    const std::string_view& text) const {
  std::vector<int> tokens;
  const auto status = sp_processor_.Encode(text, &tokens);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to encode text: " << status.ToString();
  }
  // prepend bos token
  tokens.insert(tokens.begin(), sp_processor_.bos_id());
  return tokens;
}

std::string SentencePieceTokenizer::decode(const std::vector<int>& ids) const {
  std::string text;
  const auto status = sp_processor_.Decode(ids, &text);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to decode ids: " << status.ToString();
  }
  return text;
}

}  // namespace llm
