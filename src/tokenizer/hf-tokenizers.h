#pragma once

// The C interface to the hf-tokenizers library
// ported from https://github.com/mlc-ai/tokenizers-cpp
#include <cstddef>
#include <cstdint>

using TokenizerHandle = void*;

TokenizerHandle tokenizers_new_from_str(const char* json, size_t len);

TokenizerHandle byte_level_bpe_tokenizers_new_from_str(const char* vocab,
                                                       size_t vocab_len,
                                                       const char* merges,
                                                       size_t merges_len,
                                                       const char* added_tokens,
                                                       size_t added_tokens_len);

void tokenizers_encode(TokenizerHandle handle,
                       const char* data,
                       size_t len,
                       int add_special_token);

void tokenizers_decode(TokenizerHandle handle,
                       const uint32_t* data,
                       size_t len,
                       int skip_special_token);

void tokenizers_get_decode_str(TokenizerHandle handle,
                               const char** data,
                               size_t* len);

void tokenizers_get_encode_ids(TokenizerHandle handle,
                               const uint32_t** id_data,
                               size_t* len);

void tokenizers_free(TokenizerHandle handle);
