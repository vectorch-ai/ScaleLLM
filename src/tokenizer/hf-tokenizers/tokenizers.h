#pragma once

// The C interface to the hf-tokenizers library
// ported from https://github.com/mlc-ai/tokenizers-cpp
#include <cstddef>
#include <cstdint>

using TokenizerHandle = void*;

TokenizerHandle tokenizer_from_file(const char* path);
TokenizerHandle tokenizer_from_pretrained(const char* identifier);

void tokenizer_encode(TokenizerHandle handle,
                      const char* data,
                      size_t len,
                      int add_special_token);

void tokenizer_decode(TokenizerHandle handle,
                      const uint32_t* data,
                      size_t len,
                      int skip_special_token);

void tokenizer_get_decode_str(TokenizerHandle handle,
                              const char** data,
                              size_t* len);

void tokenizer_get_encode_ids(TokenizerHandle handle,
                              const uint32_t** id_data,
                              size_t* len);

void tokenizer_free(TokenizerHandle handle);
