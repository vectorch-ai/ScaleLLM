#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <vector>

namespace {
// Adapted from
// https://github.com/NVIDIA/cutlass/blob/main/test/unit/common/filter_architecture.cpp#L78

// generate gtest filter string based on device compute capability
std::string gen_gtest_filter(const std::string& cmd_filter) {
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
      {"SM70*", 70, 75},
      {"SM75*", 75, kMaxComputeCapability},
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
  if (cmd_filter.empty()) {
    // If no cmd filter, return the negative filters
    return ss.str();
  }
  // If cmd filter is present, append the negative filters
  // to the existing filter
  return cmd_filter + ":" + ss.str();
}
}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const auto cmd_filter = ::testing::GTEST_FLAG(filter);
  ::testing::GTEST_FLAG(filter) = gen_gtest_filter(cmd_filter);
  return RUN_ALL_TESTS();
}
