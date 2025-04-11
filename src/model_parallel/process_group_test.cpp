#include "process_group.h"

#include <c10/cuda/CUDAStream.h>
#include <gtest/gtest.h>
#include <torch/types.h>

namespace llm {
namespace {
void run_collective_test(int world_size,
                         torch::DeviceType device_type,
                         std::function<void(ProcessGroup* pg)> func) {
  // create process groups
  std::vector<torch::Device> devices;
  devices.reserve(world_size);
  for (int i = 0; i < world_size; ++i) {
    devices.emplace_back(device_type, i);
  }
  auto process_groups = ProcessGroup::create_process_groups(devices);
  EXPECT_EQ(process_groups.size(), world_size);

  // run collective test in parallel
  std::vector<std::thread> threads;
  threads.reserve(process_groups.size());
  for (int i = 0; i < world_size; ++i) {
    ProcessGroup* pg = process_groups[i].get();
    threads.emplace_back([func, pg]() { func(pg); });
  }

  // wait for all threads to finish
  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

template <typename T>
std::vector<T> to_vector(const torch::Tensor& tensor) {
  auto t = tensor.cpu();
  return {t.const_data_ptr<T>(), t.const_data_ptr<T>() + t.numel()};
}
}  // namespace

class CollectiveTest : public ::testing::TestWithParam<
                           std::tuple<torch::DeviceType, torch::ScalarType>> {};

TEST_P(CollectiveTest, AllReduce) {
  const auto& [device_type, dtype] = GetParam();

  // [1, 2, 4, 8]
  for (int world_size = 1; world_size <= torch::cuda::device_count();
       world_size *= 2) {
    // create tensors
    const int num_test_tensors = 50;
    std::vector<torch::Tensor> tensors;
    tensors.reserve(num_test_tensors);
    for (int i = 0; i < num_test_tensors; ++i) {
      tensors.push_back(torch::randn({100, 4096}, dtype));
    }

    run_collective_test(world_size, device_type, [&tensors](ProcessGroup* pg) {
      const int rank = pg->rank();
      const int size = pg->world_size();
      const auto& device = pg->device();
      torch::DeviceGuard device_guard(device);
      at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
      for (int i = 0; i <= tensors.size() - size; ++i) {
        auto tensor = tensors[i + rank].to(device);
        pg->allreduce(tensor);
        stream.synchronize();
        // check the result
        auto expected = torch::zeros_like(tensors[i]);
        for (int j = 0; j < size; ++j) {
          expected += tensors[i + j];
        }
        EXPECT_TRUE(torch::equal(tensor.cpu(), expected));
      }
    });
  }
}

TEST_P(CollectiveTest, AllGather) {
  const auto& [device_type, dtype] = GetParam();

  for (int world_size = 1; world_size <= torch::cuda::device_count();
       world_size *= 2) {
    // create tensors
    const int num_test_tensors = 50;
    std::vector<torch::Tensor> tensors;
    tensors.reserve(num_test_tensors);
    for (int i = 0; i < num_test_tensors; ++i) {
      tensors.push_back(torch::ones({100, 4096}, dtype));
    }

    run_collective_test(world_size, device_type, [&tensors](ProcessGroup* pg) {
      const int rank = pg->rank();
      const int size = pg->world_size();
      const auto& device = pg->device();
      torch::DeviceGuard device_guard(device);
      at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
      for (int i = 0; i <= tensors.size() - size; ++i) {
        auto tensor = tensors[i + rank].to(device);
        std::vector<torch::Tensor> outputs(size);
        for (int j = 0; j < size; ++j) {
          outputs[j] = torch::empty_like(tensor);
        }
        pg->allgather(tensor, outputs);
        stream.synchronize();
        for (int j = 0; j < size; ++j) {
          EXPECT_TRUE(torch::equal(tensors[i + j], outputs[j].cpu()));
        }
      }
    });
  }
}

TEST_P(CollectiveTest, AllToAll) {
  const auto& [device_type, dtype] = GetParam();

  for (int world_size = 1; world_size <= torch::cuda::device_count();
       world_size *= 2) {
    run_collective_test(world_size, device_type, [dtype](ProcessGroup* pg) {
      const int size = pg->world_size();
      const auto& device = pg->device();
      torch::DeviceGuard device_guard(device);
      at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

      // create size tensors
      auto size_options = torch::device(device).dtype(torch::kInt64);
      auto input_splits = torch::randint(1, 100, {size}, size_options);
      auto output_splits = torch::empty({size}, size_options);
      // alltoall communication for sizes
      pg->alltoall(input_splits, output_splits);

      // sizes to vector
      auto input_split_sizes = to_vector<int64_t>(input_splits);
      auto output_split_sizes = to_vector<int64_t>(output_splits);

      const auto input_size =
          std::reduce(input_split_sizes.begin(), input_split_sizes.end());
      const auto output_size =
          std::reduce(output_split_sizes.begin(), output_split_sizes.end());

      // create tensors
      auto options = torch::device(device).dtype(dtype);
      auto input = torch::randn({input_size}, options);
      auto output = torch::empty({output_size}, options);

      // alltoall communication for data
      pg->alltoall(input, output, input_split_sizes, output_split_sizes);

      auto input2 = torch::empty_like(input);
      // alltoall communication again with swapped input and output
      // NOLINTNEXTLINE(readability-suspicious-call-argument)
      pg->alltoall(output, input2, output_split_sizes, input_split_sizes);

      // we should get the same input tensor back
      EXPECT_TRUE(torch::equal(input, input2));
    });
  }
}

INSTANTIATE_TEST_SUITE_P(
    ProcessGroupTest,
    CollectiveTest,
    ::testing::Combine(::testing::Values(torch::kCUDA),  // device type
                       ::testing::Values(torch::kFloat,
                                         torch::kHalf,
                                         torch::kBFloat16)  // dtype
                       ));

}  // namespace llm