#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <thread>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

#include "linear.h"
#include "torch_utils/state_dict.h"

namespace llm {

// TODO: Remove this test
TEST(LayersTest, TestGroupNCCL) {
  // test group nccl in multiple threads environment
  // each thread has its own group instance
  // on device 0
  torch::Device device(torch::kCUDA, 0);
  torch::DeviceGuard device_guard(device);
  const int64_t world_size = 2;
  const int64_t rank = 0;
  const std::string path_ = "/tmp/test_group_nccl1";
  auto store = c10::make_intrusive<::c10d::FileStore>(path_, world_size);
  c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> opts =
      c10::make_intrusive<c10d::ProcessGroupNCCL::Options>();
  auto pg = std::make_unique<::c10d::ProcessGroupNCCL>(
      store, rank, world_size, std::move(opts));

  auto tensor0 = torch::rand({10, 20});
  auto tensor1 = torch::rand({10, 20});

  std::thread rank1_thread([tensor0, tensor1]() {
    // on device 1
    torch::Device device(torch::kCUDA, 1);
    torch::DeviceGuard device_guard(device);

    const int64_t world_size = 2;
    const int64_t rank = 1;
    const std::string path_ = "/tmp/test_group_nccl1";
    auto store = c10::make_intrusive<::c10d::FileStore>(path_, world_size);
    c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> opts =
        c10::make_intrusive<c10d::ProcessGroupNCCL::Options>();
    auto pg = std::make_unique<::c10d::ProcessGroupNCCL>(
        store, rank, world_size, std::move(opts));

    // for (int i = 0; i < 10; ++i) {
    std::vector<at::Tensor> tensors = {tensor1.to(device)};
    auto work = pg->allreduce(tensors);
    work->wait();

    auto result = tensors[0].cpu();
    EXPECT_TRUE(torch::equal(result, tensor0 + tensor1));
    // }
  });

  // for (int i = 0; i < 10; ++i) {
  std::vector<at::Tensor> tensors = {tensor0.to(device)};
  auto work = pg->allreduce(tensors);
  work->wait();

  auto result = tensors[0].cpu();
  EXPECT_TRUE(torch::equal(result, tensor0 + tensor1));
  // }

  if (rank1_thread.joinable()) {
    rank1_thread.join();
  }
}

TEST(LayersTest, TestLoadStateDict) {
  // test load state dict for linear
  const int64_t in_features = 10;
  const int64_t out_features = 20;

  torch::Device device(torch::kCPU);
  torch::ScalarType dtype(torch::kFloat);
  ParallelArgs parallel_args(0, 1, nullptr);
  ColumnParallelLinear linear(in_features,
                              out_features,
                              /*gather_output=*/false,
                              parallel_args,
                              dtype,
                              device);
  std::unordered_map<std::string, torch::Tensor> state_dict_data;
  // Allocate transposed weight matrix
  state_dict_data["weight"] = torch::randn({out_features, in_features});

  StateDict state_dict(state_dict_data);
  // test load state dict for transformer
  linear->load_state_dict(state_dict);

  EXPECT_EQ(state_dict_data["weight"].data_ptr(),
            state_dict.get_tensor("weight").data_ptr());

  auto named_parameters = linear->named_parameters(/*recurse=*/false);
  EXPECT_TRUE(
      torch::equal(state_dict_data["weight"], named_parameters["weight"]));
}

}  // namespace llm
