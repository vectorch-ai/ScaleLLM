include(cc_library)
include(cc_test)

cc_library(
  NAME 
    tokenizer
  HDRS 
    tokenizer_args.h
    tokenizer.h
    tiktoken_tokenizer.h
    sentencepiece_tokenizer.h
    hf_tokenizer.h
  SRCS 
    tiktoken_tokenizer.cpp
    sentencepiece_tokenizer.cpp
    hf_tokenizer.cpp
  DEPS
    :sentencepiece
    absl::flat_hash_map
    absl::strings
    tokenizers
    glog::glog
    re2::re2
)

cc_test(
  NAME
    tokenizer_test
  SRCS
    sentencepiece_tokenizer_test.cpp
    tiktoken_tokenizer_test.cpp
  DEPS
    :tokenizer
    tokenizers
    GTest::gtest_main
  DATA 
    data/tokenizer.model
    data/test.tiktoken
)

# build huggingface tokenizers (rust)
add_subdirectory(tokenizers)
