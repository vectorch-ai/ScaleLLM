#pragma once

// The C API
#ifdef __cplusplus
extern "C" {
#endif

// The C interface to the hf-tokenizers library
// ported from https://github.com/mlc-ai/tokenizers-cpp
#include <stddef.h>
#include <stdint.h>

using TokenizerHandle = void*;

TokenizerHandle tokenizer_from_file(const char* path);
// TokenizerHandle tokenizer_from_pretrained(const char* identifier);

void tokenizer_encode(TokenizerHandle handle,
                      const char* data,
                      size_t len,
                      bool add_special_tokens);

void tokenizer_decode(TokenizerHandle handle,
                      const uint32_t* data,
                      size_t len,
                      bool skip_special_tokens);

void tokenizer_get_decode_str(TokenizerHandle handle,
                              const char** data,
                              size_t* len);

void tokenizer_get_encode_ids(TokenizerHandle handle,
                              const uint32_t** id_data,
                              size_t* len);

void tokenizer_id_to_token(TokenizerHandle handle,
                           uint32_t id,
                           const char** data,
                           size_t* len);

// -1 if token is not in vocab
int32_t tokenizer_token_to_id(TokenizerHandle handle,
                              const char* token,
                              size_t len);

void tokenizer_free(TokenizerHandle handle);

size_t tokenizer_get_vocab_size(TokenizerHandle handle, bool with_added_tokens);

#ifdef __cplusplus
}
#endif
