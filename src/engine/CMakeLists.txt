include(cc_library)
include(cc_test)

cc_library(
  NAME 
    engine
  HDRS
    parameters.h
    utils.h
    batch.h
    worker.h
    engine.h
    llm_engine.h
  SRCS
    utils.cpp
    batch.cpp
    worker.cpp
    llm_engine.cpp
  DEPS
    torch
    :common
    :request
    :state_dict
    :models
    :sampler
    :tokenizer
    :model_loader
    glog::glog
    Folly::folly
    absl::synchronization
  LINKOPTS
    atomic
)

cc_test(
  NAME
    engine_test
  SRCS
    batch_test.cpp
    worker_test.cpp
  DEPS
    :engine
    absl::time
    GTest::gtest_main
)
