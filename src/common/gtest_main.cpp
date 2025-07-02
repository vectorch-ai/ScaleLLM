#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

namespace {
// Adapted from https://github.com/NVIDIA/cutlass/blob/main/test/unit/common/filter_architecture.cpp#L78

// generate gtest filter string based on device compute capability
std::string gen_gtest_filter() {
  static const int kMaxComputeCapability = 10000;

  int compute_capability = 0;
  cudaError_t err;
  int cudaDeviceId;
  err = cudaGetDevice(&cudaDeviceId);
  if (cudaSuccess != err) {
    std::cout << "*** Warn: Could not detect active GPU device ID"
              << " [" << cudaGetErrorString(err) << "]" << std::endl;
    // set to 0 to disable all kernel tests
    compute_capability = 0;
  } else {
    cudaDeviceProp device_properties;
    err = cudaGetDeviceProperties(&device_properties, cudaDeviceId);
    if (cudaSuccess != err) {
      std::cerr << "*** Error: Could not get device properties for GPU "
                << cudaDeviceId << " [" << cudaGetErrorString(err) << "]"
                << std::endl;
      exit(1);
    }
    compute_capability =
        (device_properties.major * 10) + device_properties.minor;
  }

  // Text filters for each kernel based on supported compute capability
  struct {
    /// Unit test filter string
    char const* filter;

    /// Minimum compute capability for the kernels in the named test
    int min_compute_capability;

    /// Maximum compute capability for which the kernels are enabled
    int max_compute_capability;
  } test_filters[] = {{"SM80*", 80, kMaxComputeCapability},
                      {"SM89*", 89, 89},
                      {"SM90*", 90, 90},
                      {"SM100*", 100, 100},
                      {"SM120*", 120, 120},
                      {nullptr, 0, 0}};
  // Set negative test filters
  std::stringstream ss;
  ss << "-";
  for (int i = 0, j = 0; test_filters[i].filter; ++i) {
    if (compute_capability < test_filters[i].min_compute_capability ||
        compute_capability > test_filters[i].max_compute_capability) {
      ss << (j++ ? ":" : "") << test_filters[i].filter;
    }
  }
  return ss.str();
}
}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // honor --gtest_filter from commandline
  if (::testing::GTEST_FLAG(filter).empty() ||
      ::testing::GTEST_FLAG(filter) == "*") {
    ::testing::GTEST_FLAG(filter) = gen_gtest_filter();
  }
  return RUN_ALL_TESTS();
}