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

bool SentencePieceTokenizer::encode(const std::string_view& text,
                                    std::vector<int>* ids) const {
  const auto status = sp_processor_.Encode(text, ids);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to encode text: " << text << ", error "
               << status.ToString();
    return false;
  }
  // prepend bos token
  ids->insert(ids->begin(), sp_processor_.bos_id());
  return true;
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
