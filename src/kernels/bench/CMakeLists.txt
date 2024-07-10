include(cc_binary)

cc_binary(
  NAME 
    kernel_bench
  SRCS 
    bench_demo.cu
  DEPS
    nvbench::nvbench
    nvbench::main
)
