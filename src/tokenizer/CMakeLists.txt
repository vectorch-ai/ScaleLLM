include(cc_library)
include(cc_test)

cc_library(
  NAME 
    tokenizer
  HDRS 
    tokenizer.h
    sentencepiece_tokenizer.h
  SRCS 
    sentencepiece_tokenizer.cpp
  DEPS
    glog::glog
    SentencePiece::sentencepiece
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
