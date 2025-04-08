#include "process_group.h"

#include <c10/core/Device.h>
#include <c10/cuda/CUDAStream.h>
#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/types.h>

namespace llm {
namespace {
void run_collective_test(int world_size,
                         std::function<void(ProcessGroup* pg)> func) {
  // create process groups
  std::vector<torch::Device> devices;
  devices.reserve(world_size);
  for (int i = 0; i < world_size; ++i) {
    devices.emplace_back(torch::kCUDA, i);
  }
  auto process_groups = ProcessGroup::create_process_groups(devices);
  EXPECT_EQ(process_groups.size(), world_size);

  // run collective test in parallel
  std::vector<std::thread> threads;
  threads.reserve(process_groups.size());
  for (int i = 0; i < world_size; ++i) {
    threads.emplace_back([func, pg = process_groups[i].get()]() { func(pg); });
  }

  // wait for all threads to finish
  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}
}  // namespace

TEST(ProcessGroupTest, NCCLAllReduce) {
  // skip test if less than two gpus
  if (torch::cuda::device_count() < 2) {
    GTEST_SKIP() << "Skipping test because less than two gpus";
  }

  for (int i = 2; i <= torch::cuda::device_count(); i *= 2) {
    // create tensors
    const int num_test_tensors = 50;
    std::vector<torch::Tensor> tensors;
    tensors.reserve(num_test_tensors);
    for (int i = 0; i < num_test_tensors; ++i) {
      tensors.push_back(torch::ones({100, 4096}, torch::kHalf));
    }

    run_collective_test(i, [&tensors](ProcessGroup* pg) {
      const int rank = pg->rank();
      const int world_size = pg->world_size();
      const auto& device = pg->device();
      torch::DeviceGuard device_guard(device);
      at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
      for (int i = 0; i <= tensors.size() - world_size; ++i) {
        auto tensor = tensors[i + rank].to(device);
        pg->allreduce(tensor);
        stream.synchronize();
        // check the result
        auto expected = torch::zeros_like(tensors[i]);
        for (int j = 0; j < world_size; ++j) {
          expected += tensors[i + j];
        }
        EXPECT_TRUE(torch::equal(tensor.cpu(), expected));
      }
    });
  }
}

TEST(ProcessGroupTest, NCCLAllGather) {
  // skip test if less than two gpus
  if (torch::cuda::device_count() < 2) {
    GTEST_SKIP() << "Skipping test because less than two gpus";
  }

  for (int i = 2; i <= torch::cuda::device_count(); i *= 2) {
    // create tensors
    const int num_test_tensors = 50;
    std::vector<torch::Tensor> tensors;
    tensors.reserve(num_test_tensors);
    for (int i = 0; i < num_test_tensors; ++i) {
      tensors.push_back(torch::ones({100, 4096}, torch::kHalf));
    }

    run_collective_test(i, [&tensors](ProcessGroup* pg) {
      const int rank = pg->rank();
      const int world_size = pg->world_size();
      const auto& device = pg->device();
      torch::DeviceGuard device_guard(device);
      at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
      for (int i = 0; i <= tensors.size() - world_size; ++i) {
        auto tensor = tensors[i + rank].to(device);
        std::vector<torch::Tensor> outputs(world_size);
        for (int j = 0; j < world_size; ++j) {
          outputs[j] = torch::empty_like(tensor);
        }
        pg->allgather(tensor, outputs);
        stream.synchronize();
        for (int j = 0; j < world_size; ++j) {
          EXPECT_TRUE(torch::equal(tensors[i + j], outputs[j].cpu()));
        }
      }
    });
  }
}

TEST(ProcessGroupTest, NCCLAll2All) {
  // skip test if less than two gpus
  if (torch::cuda::device_count() < 2) {
    GTEST_SKIP() << "Skipping test because less than two gpus";
  }

  // run all to all
  // >>> input = torch.arange(4) + rank * 4
  // >>> input
  // tensor([0, 1, 2, 3])     # Rank 0
  // tensor([4, 5, 6, 7])     # Rank 1
  // tensor([8, 9, 10, 11])   # Rank 2
  // tensor([12, 13, 14, 15]) # Rank 3
  // >>> output = torch.empty([4], dtype=torch.int64)
  // >>> dist.all_to_all_single(output, input)
  // >>> output
  // tensor([0, 4, 8, 12])    # Rank 0
  // tensor([1, 5, 9, 13])    # Rank 1
  // tensor([2, 6, 10, 14])   # Rank 2
  // tensor([3, 7, 11, 15])   # Rank 3

  for (int i = 2; i <= torch::cuda::device_count(); i *= 2) {
    run_collective_test(i, [](ProcessGroup* pg) {
      const int rank = pg->rank();
      const int world_size = pg->world_size();
      const auto& device = pg->device();
      torch::DeviceGuard device_guard(device);

      // create size tensors
      auto options = torch::device(device).dtype(torch::kInt32);
      auto input_sizes = torch::ones({world_size}, options);
      auto output_sizes = torch::empty({world_size}, options);
      at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
      pg->alltoall(input_sizes, output_sizes);
      stream.synchronize();

      LOG(INFO) << "rank: " << rank << ", input_sizes: " << input_sizes
                << ", output_sizes: " << output_sizes;
      // EXPECT_TRUE(false);
    });
  }
}

}  // namespace llm