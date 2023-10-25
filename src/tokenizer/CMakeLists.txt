include(cc_library)
include(cc_test)

cc_library(
  NAME 
    tokenizer
  HDRS 
    tokenizer.h
    sentencepiece_tokenizer.h
    hf_tokenizer.h
  SRCS 
    sentencepiece_tokenizer.cpp
    hf_tokenizer.cpp
  DEPS
    :sentencepiece
    tokenizers
    glog::glog
)

cc_test(
  NAME
    tokenizer_test
  SRCS
    sentencepiece_tokenizer_test.cpp
    hf_tokenizer_test.cpp
  DEPS
    :tokenizer
    tokenizers
    GTest::gtest_main
  DATA 
    data/tokenizer.model
    data/tokenizer.json.bin
)

# build huggingface tokenizers (rust)
add_subdirectory(tokenizers)
