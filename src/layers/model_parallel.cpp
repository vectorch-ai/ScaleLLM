#include "model_parallel.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <torch/csrc/distributed/c10d/Types.hpp>
#include <vector>

#include "models/parallel_args.h"

namespace llm {
namespace distributed {
// gathers tensors from the whole group in a list.
// tensors: list of tensors to gather. Output list.
// input: tensor to be broadcasted from current thread.
// process_group: process group to work on.
void all_gather(std::vector<torch::Tensor> outputs,
                torch::Tensor input,
                c10d::Backend* process_group) {
  CHECK_NOTNULL(process_group);
  std::vector<std::vector<torch::Tensor>> output_tensors{outputs};
  std::vector<torch::Tensor> input_tensors{input};
  auto work = process_group->allgather(output_tensors, input_tensors);
  CHECK(work->wait()) << "allgather failed";
}

void all_reduce(torch::Tensor input, c10d::Backend* process_group) {
  CHECK_NOTNULL(process_group);
  std::vector<torch::Tensor> tensors{input};
  auto work = process_group->allreduce(tensors);
  CHECK(work->wait()) << "allreduce failed";
}
}  // namespace distributed

torch::Tensor gather_from_model_parallel_region(
    torch::Tensor input,
    const ParallelArgs& parallel_args) {
  const auto world_size = parallel_args.world_size();
  if (world_size == 1) {
    // bypass if only have one gpu
    return input;
  }

  const auto rank = parallel_args.rank();
  auto* process_group = parallel_args.process_group();
  std::vector<torch::Tensor> tensors(world_size);
  for (int64_t i = 0; i < world_size; ++i) {
    tensors[i] = (i == rank) ? input : torch::empty_like(input);
  }
  distributed::all_gather(tensors, input, process_group);
  return torch::cat(tensors, /*dim=*/-1).contiguous();
}

torch::Tensor reduce_from_model_parallel_region(
    torch::Tensor input,
    const ParallelArgs& parallel_args) {
  const auto world_size = parallel_args.world_size();
  if (world_size == 1) {
    // bypass if only have one gpu
    return input;
  }

  distributed::all_reduce(input, parallel_args.process_group());
  return input;
}

torch::Tensor scatter_to_model_parallel_region(
    torch::Tensor input,
    const ParallelArgs& parallel_args) {
  const auto world_size = parallel_args.world_size();
  if (world_size == 1) {
    // bypass if only have one gpu
    return input;
  }

  // get the size for last dimension
  const auto last_dim_size = input.size(-1);
  CHECK(last_dim_size % world_size == 0)
      << "last_dim_size " << last_dim_size << " not divisible by world_size "
      << world_size;

  // torch::split does not create contiguous tensors by default.
  const auto tensor_list = input.split(last_dim_size / world_size, /*dim=*/-1);
  const auto rank = parallel_args.rank();
  return tensor_list[rank];
}

}  // namespace llm
