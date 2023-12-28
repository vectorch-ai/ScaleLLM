#include "sentencepiece_tokenizer.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <re2/re2.h>

#include "common/logging.h"
#include "sentencepiece.pb.h"
#include "sentencepiece/sentencepiece_processor.h"

#define RETURN_FALSE_IF_ERROR(expr)  \
  do {                               \
    const auto _status = expr;       \
    if (!_status.ok()) return false; \
  } while (0)

#define RETURN_IF_ERROR(expr)  \
  do {                         \
    const auto _status = expr; \
    if (!_status.ok()) return; \
  } while (0)

namespace llm {

SentencePieceTokenizer::SentencePieceTokenizer(
    const std::string& vocab_file_path,
    const std::vector<std::string>& special_tokens,
    bool prepend_bos)
    : vocab_file_path_(vocab_file_path),
      special_tokens_(special_tokens),
      prepend_bos_(prepend_bos) {
  const auto status = sp_processor_.Load(vocab_file_path);
  if (!status.ok()) {
    GLOG(FATAL) << "Failed to load SentencePiece model from " << vocab_file_path
                << ": " << status.ToString() << ", error " << status.ToString();
  }

  if (special_tokens.empty()) {
    // no special tokens, just return
    return;
  }

  // add special tokens and construct special token regex
  // TODO: use special token start id from tokenizer args
  int32_t next_id = static_cast<int32_t>(sp_processor_.GetPieceSize());
  for (const auto& token : special_tokens) {
    if (token.empty()) {
      continue;
    }
    if (!special_token_encoder_.try_emplace(token, next_id).second) {
      GLOG(WARNING) << "Duplicate special token: " << token;
    }
    if (!special_token_decoder_.try_emplace(next_id, token).second) {
      GLOG(WARNING) << "Duplicate special token id: " << next_id;
    }
    ++next_id;
  }

  // build special token regex
  std::vector<std::string> escaped_tokens;
  escaped_tokens.reserve(special_tokens.size());
  for (const auto& token : special_tokens) {
    if (token.empty()) {
      continue;
    }
    // escape each token
    const auto escaped_token = re2::RE2::QuoteMeta(token);
    escaped_tokens.push_back(escaped_token);
  }
  if (!escaped_tokens.empty()) {
    const auto special_token_regex_str = absl::StrJoin(escaped_tokens, "|");
    // surround with () to match special tokens
    const auto regex_str = absl::StrCat("(", special_token_regex_str, ")");
    special_token_regex_ = std::make_unique<re2::RE2>(regex_str);
  }
}

bool SentencePieceTokenizer::encode_internal(const std::string_view& text,
                                             std::vector<int32_t>* ids) const {
  if (text.empty()) {
    // empty text, just return
    return true;
  }

  sentencepiece::SentencePieceText spt;
  RETURN_FALSE_IF_ERROR(sp_processor_.Encode(text, &spt));
  for (const auto& sp : spt.pieces()) {
    ids->emplace_back(sp.id());
  }
  return true;
}

bool SentencePieceTokenizer::encode(const std::string_view& text,
                                    std::vector<int32_t>* ids) const {
  // prepend bos token
  if (prepend_bos_) {
    ids->insert(ids->begin(), sp_processor_.bos_id());
  }

  if (special_token_regex_ == nullptr) {
    return encode_internal(text, ids);
  }

  std::string_view input = text;
  std::string_view special;
  while (true) {
    const auto* start = input.begin();
    if (!re2::RE2::FindAndConsume(&input, *special_token_regex_, &special)) {
      // no more special tokens
      break;
    }

    // encode text before special token if exists
    const std::string_view sub_input(start,
                                     input.begin() - start - special.size());
    if (!encode_internal(sub_input, ids)) {
      return false;
    }

    // add special token id if exists
    const auto sit = special_token_encoder_.find(special);
    if (sit != special_token_encoder_.end()) {
      // find one special token
      ids->push_back(sit->second);
    }
  }

  // encode remaining text if exists
  return encode_internal(input, ids);
}

void SentencePieceTokenizer::decode_internal(const std::vector<int32_t>& ids,
                                             size_t start,
                                             size_t end,
                                             std::stringstream* ss) const {
  if (start >= end) {
    // no text to decode
    return;
  }

  sentencepiece::SentencePieceText spt;
  std::vector<std::string> pieces;
  const int num_pieces = sp_processor_.GetPieceSize();
  pieces.reserve(end - start);
  for (size_t i = start; i < end; ++i) {
    const auto id = ids[i];
    if (id < 0 || id >= num_pieces) {
      GLOG(ERROR) << "Invalid id: " << id;
      continue;
    }
    pieces.emplace_back(sp_processor_.IdToPiece(id));
  }
  RETURN_IF_ERROR(sp_processor_.Decode(pieces, &spt));
  (*ss) << spt.text();
}

std::string SentencePieceTokenizer::decode(
    const std::vector<int32_t>& ids) const {
  std::stringstream ss;
  size_t start = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    const auto sit = special_token_decoder_.find(ids[i]);
    if (sit == special_token_decoder_.end()) {
      continue;
    }
    // decode text before special token if exists
    decode_internal(ids, start, i, &ss);
    // output special token
    ss << sit->second;
    start = i + 1;
  }

  // decode remaining text if exists
  decode_internal(ids, start, ids.size(), &ss);
  return ss.str();
}

size_t SentencePieceTokenizer::vocab_size() const {
  // vocab size = sentencepiece vocab size + special tokens
  return sp_processor_.GetPieceSize() + special_tokens_.size();
}

std::unique_ptr<Tokenizer> SentencePieceTokenizer::clone() const {
  return std::make_unique<SentencePieceTokenizer>(
      this->vocab_file_path_, this->special_tokens_, this->prepend_bos_);
}

}  // namespace llm
