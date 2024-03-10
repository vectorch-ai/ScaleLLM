#include "hf_tokenizer.h"

#include <glog/logging.h>

#include "huggingface/tokenizers.h"

namespace llm {

std::unique_ptr<HFTokenizer> HFTokenizer::from_file(
    const std::string& tokenizer_file_path) {
  TokenizerHandle handle = tokenizer_from_file(tokenizer_file_path.c_str());
  CHECK(handle != nullptr) << "Failed to load tokenizer from file: "
                           << tokenizer_file_path;
  return std::make_unique<HFTokenizer>(tokenizer_file_path, handle);
}

HFTokenizer::HFTokenizer(const std::string& tokenizer_file_path,
                         TokenizerHandle handle)
    : tokenizer_file_path_(tokenizer_file_path), handle_(handle) {
  CHECK(handle_ != nullptr);
}

std::unique_ptr<Tokenizer> HFTokenizer::clone() const {
  return from_file(tokenizer_file_path_);
}

HFTokenizer::~HFTokenizer() { tokenizer_free(handle_); }

bool HFTokenizer::encode(const std::string_view& text,
                         std::vector<int32_t>* ids) const {
  tokenizer_encode(
      handle_, text.data(), text.size(), /*add_special_tokens=*/true);
  const uint32_t* data = nullptr;
  size_t len = 0;
  tokenizer_get_encode_ids(handle_, &data, &len);
  ids->reserve(len);
  for (size_t i = 0; i < len; ++i) {
    ids->push_back(static_cast<int32_t>(data[i]));
  }
  return true;
}

std::string HFTokenizer::decode(const std::vector<int32_t>& ids) const {
  tokenizer_decode(handle_,
                   reinterpret_cast<const uint32_t*>(ids.data()),
                   ids.size(),
                   /*skip_special_tokens=*/true);
  const char* data = nullptr;
  size_t len = 0;
  tokenizer_get_decode_str(handle_, &data, &len);
  return {data, len};
}

size_t HFTokenizer::vocab_size() const {
  return tokenizer_vocab_size(handle_, /*with_added_tokens=*/true);
}

}  // namespace llm
