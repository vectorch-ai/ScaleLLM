include(cc_library)
include(cc_test)

cc_library(
  NAME 
    tokenizer
  HDRS 
    hf-tokenizers.h
    tokenizer.h
    sentencepiece_tokenizer.h
  SRCS 
    sentencepiece_tokenizer.cpp
  DEPS
    :sentencepiece
    glog::glog
    hf-tokenizers
)

cc_test(
  NAME
    tokenizer_test
  SRCS
    sentencepiece_tokenizer_test.cpp
  DEPS
    :tokenizer
    GTest::gtest_main
)

# build huggingface tokenizers (rust)
add_subdirectory(hf-tokenizers)
