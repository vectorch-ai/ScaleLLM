#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <vector>

namespace {
// Adapted from
// https://github.com/NVIDIA/cutlass/blob/main/test/unit/common/filter_architecture.cpp#L78

// generate gtest filter string based on device compute capability
std::string gen_gtest_filter() {
  static const int kMaxComputeCapability = 10000;
  // Text filters for each kernel based on supported compute capability
  struct Filter {
    // Test filter string
    char const* filter;

    // Minimum and Maximum compute capability for the test group
    int min_compute_capability;
    int max_compute_capability;
  };
  static const std::vector<Filter> filters = {
      {"SM80*", 80, kMaxComputeCapability},
      {"SM89*", 89, 89},
      {"SM90*", 90, 90},
      {"SM100*", 100, 100},
      {"SM120*", 120, 120}};

  int compute_capability = 0;
  cudaError_t err;
  int cudaDeviceId;
  err = cudaGetDevice(&cudaDeviceId);
  if (cudaSuccess != err) {
    std::cerr << "*** Warn: Could not detect active GPU device ID" << " ["
              << cudaGetErrorString(err) << "]" << std::endl;
  } else {
    cudaDeviceProp device_properties;
    err = cudaGetDeviceProperties(&device_properties, cudaDeviceId);
    if (cudaSuccess != err) {
      std::cerr << "*** Warn: Could not get device properties for GPU "
                << cudaDeviceId << " [" << cudaGetErrorString(err) << "]"
                << std::endl;
    } else {
      compute_capability =
          (device_properties.major * 10) + device_properties.minor;
    }
  }
  // Set negative test filters
  std::stringstream ss;
  ss << "-";
  int i = 0;
  for (const auto& filter : filters) {
    if (compute_capability >= filter.min_compute_capability &&
        compute_capability <= filter.max_compute_capability) {
      // Skip filter for supported tests
      continue;
    }
    // add separator if not the first filter
    ss << (i++ ? ":" : "") << filter.filter;
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
