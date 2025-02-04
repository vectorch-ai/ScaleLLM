
#include "aot/add_kernel_fp16_sm80.cuh"
#include "aot/add_kernel_fp32_sm80.cuh"

#include <c10/cuda/CUDAStream.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

namespace llm {

TEST(TritionTest, ExampleFp16Kernel) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  const auto options = torch::dtype(torch::kHalf).device(torch::kCUDA);
  const int64_t n_elements = 98432;
  // prepare inputs
  auto x = torch::rand({n_elements}, options);
  auto y = torch::rand({n_elements}, options);

  auto output_ref = x + y;
  auto output = torch::empty_like(output_ref);

  // load kernel before using it
  load_add_kernel_fp16_sm80();

  // launch kernel and compare results
  auto stream = at::cuda::getCurrentCUDAStream();
  CUresult result = add_kernel_fp16_sm80_default(
      stream, reinterpret_cast<CUdeviceptr>(x.data_ptr()),
      reinterpret_cast<CUdeviceptr>(y.data_ptr()),
      reinterpret_cast<CUdeviceptr>(output.data_ptr()), n_elements);
  EXPECT_EQ(result, CUDA_SUCCESS);

  // wait for kernel to finish
  stream.synchronize();

  EXPECT_TRUE(torch::equal(output, output_ref));

  // unload kernel after using it
  unload_add_kernel_fp16_sm80();
}

TEST(TritionTest, ExampleFp32Kernel) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  const auto options = torch::dtype(torch::kFloat).device(torch::kCUDA);
  const int64_t n_elements = 98432;
  // prepare inputs
  auto x = torch::rand({n_elements}, options);
  auto y = torch::rand({n_elements}, options);

  auto output_ref = x + y;
  auto output = torch::empty_like(output_ref);

  // load kernel before using it
  load_add_kernel_fp32_sm80();

  // launch kernel and compare results
  auto stream = at::cuda::getCurrentCUDAStream();
  CUresult result = add_kernel_fp32_sm80_default(
      stream, reinterpret_cast<CUdeviceptr>(x.data_ptr()),
      reinterpret_cast<CUdeviceptr>(y.data_ptr()),
      reinterpret_cast<CUdeviceptr>(output.data_ptr()), n_elements);
  EXPECT_EQ(result, CUDA_SUCCESS);

  // wait for kernel to finish
  stream.synchronize();

  EXPECT_TRUE(torch::equal(output, output_ref));

  // unload kernel after using it
  unload_add_kernel_fp32_sm80();
}

} // namespace llm
