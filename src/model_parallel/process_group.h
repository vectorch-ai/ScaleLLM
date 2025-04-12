#pragma once
#include <nccl.h>
#include <torch/torch.h>

#include <memory>
#include <vector>

namespace llm {
// A interface for process group.
class ProcessGroup {
 public:
  ProcessGroup(int rank, int world_size, const torch::Device& device)
      : rank_(rank), world_size_(world_size), device_(device) {}

  virtual ~ProcessGroup() = default;

  int rank() const { return rank_; }

  int world_size() const { return world_size_; }

  const torch::Device& device() const { return device_; }

  // allreduce: reduce the input tensor across all processes, and all processes
  // get the result.
  // blocking operation.
  virtual void allreduce(torch::Tensor& input) const = 0;

  // allgather: gather tensors from all processes and concatenate them.
  virtual void allgather(const torch::Tensor& input,
                         std::vector<torch::Tensor>& outputs) const = 0;

  // allgather: gather tensors from all processes and concatenate them.
  virtual void allgather(const torch::Tensor& input,
                         torch::Tensor& outputs) const = 0;

  // alltoall: scatter input tensor to all processes and gather the result.
  virtual void alltoall(const torch::Tensor& input,
                        torch::Tensor& output) const = 0;

  virtual void alltoall(
      const torch::Tensor& input,
      torch::Tensor& output,
      const std::vector<int64_t>& input_split_sizes,
      const std::vector<int64_t>& output_split_sizes) const = 0;

  // Create a process group where each process has a single GPU
  // devices: list of devices to create process groups on.
  static std::vector<std::unique_ptr<ProcessGroup>> create_process_groups(
      const std::vector<torch::Device>& devices);

 private:
  // rank of current process.
  int rank_ = 0;

  // number of processes.
  int world_size_ = 0;

  // device of current process.
  torch::Device device_;
};

// A wrapper for nccl designed to function in a multi-threaded environment,
// where each thread should possess its own nccl communicator and stream.
// Unfortunately, the ProcessGroupNCCL in PyTorch is designed exclusively for
// multi-process environments. Using it in a multi-threading environment can
// lead to deadlocks.
// TODO: This class can be removed once PyTorch's ProcessGroupNCCL supports
// multi-threading environments.
class ProcessGroupNCCL : public ProcessGroup {
 public:
  // Constructor.
  ProcessGroupNCCL(int rank,
                   int world_size,
                   const torch::Device& device,
                   ncclComm_t comm);

  // Destructor.
  ~ProcessGroupNCCL() override;

  void allreduce(torch::Tensor& input) const override;

  void allgather(const torch::Tensor& input,
                 std::vector<torch::Tensor>& outputs) const override;

  void allgather(const torch::Tensor& input,
                 torch::Tensor& outputs) const override;

  void alltoall(const torch::Tensor& input,
                torch::Tensor& output) const override;

  void alltoall(const torch::Tensor& input,
                torch::Tensor& output,
                const std::vector<int64_t>& input_split_sizes,
                const std::vector<int64_t>& output_split_sizes) const override;

 private:
  // nccl communicator.
  ncclComm_t comm_ = nullptr;
};
}  // namespace llm
