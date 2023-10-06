include(cc_binary)

cc_binary(
  NAME
    micro_benchmark
  SRCS
    # kv_cache_benchmark.cpp
    # attention_benchmark.cpp
    activation_benchmark.cpp
  DEPS
    :layers
    benchmark::benchmark
    benchmark::benchmark_main
)
