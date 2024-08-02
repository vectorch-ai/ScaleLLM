extern "C" {
#include "aot/add_kernel_fp16.h"
#include "aot/add_kernel_fp32.h"
}

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace llm {

TEST(TritionTest, ExampleKernel) {
  // ref to: https://github.com/triton-lang/triton/blob/3cf365b631f4d8fbcff3ed5275b9099686f17489/python/test/unit/tools/test_aot.py#L120
  // initialize CUDA handles
  CUdevice dev;
  CUcontext ctx;
  CUstream stream;
  cuInit(0);
  cuDeviceGet(&dev, 0);
  cuCtxCreate(&ctx, 0, dev);
  cuStreamCreate(&stream, 0);

  // load kernel before using it
  load_add_kernel_fp16();

  cuStreamSynchronize(stream);
  // TODO: launch kernel and compare results

  cuStreamSynchronize(stream);

  // unload kernel after using it
  unload_add_kernel_fp16();

  // destroy CUDA handles
  cuCtxDestroy(ctx);
}

}  // namespace llm
