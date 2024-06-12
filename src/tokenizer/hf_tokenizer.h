#pragma once

#include "huggingface/tokenizers.h"
#include "tokenizer.h"

namespace llm {

// a tokenizer that uses hf/tokenizers
// not thread-safe, can't be used in multiple threads.
class HFTokenizer : public Tokenizer {
 public:
  HFTokenizer(const std::string& tokenizer_file_path, TokenizerHandle handle);

  ~HFTokenizer() override;

  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids) const override;

  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const override;

  std::optional<int32_t> token_to_id(
      const std::string_view& token) const override;

  std::string id_to_token(int32_t id) const override;

  size_t vocab_size() const override;

  std::unique_ptr<Tokenizer> clone() const override;

  static std::unique_ptr<HFTokenizer> from_file(const std::string& path);

 private:
  std::string tokenizer_file_path_;

  TokenizerHandle handle_ = nullptr;
};

}  // namespace llm
