#include "worker.h"

#include <c10/cuda/CUDAGuard.h>
#include <folly/futures/Future.h>
#include <torch/torch.h>

#include <string>
#include <utility>

#include "executor.h"
// #include "torch_utils/state_dict.h"

namespace llm {

void Worker::load_state_dict(const StateDict& state_dict) {

}

void Worker::execute_model_async(
    torch::Tensor tokens,     // [num_tokens]
    torch::Tensor positions,  // [num_tokens]
    torch::Tensor slots,      // [num_tokens] key value cache slots
    InputParameters parameters) {
  executor_.schedule([this, tokens, positions, slots, parameters] {
    // run the model on the given input in working thread
    this->execute_model(tokens, positions, slots, parameters);
  });
}

// initialize model, cache manager. async call
folly::Future<bool> Worker::init_async() {
  folly::Promise<bool> promise;
  auto future = promise.getFuture();
  executor_.schedule([this, promise = std::move(promise)]() mutable {
    // initialize model and cache manager within the working thread
    const bool success = this->init();
    promise.setValue(success);
  });
  return future;
}

void Worker::execute_model(
    torch::Tensor tokens,     // [num_tokens]
    torch::Tensor positions,  // [num_tokens]
    torch::Tensor slots,      // [num_tokens] key value cache slots
    InputParameters parameters) const {
  // all tensors should be on the same device as model
  auto d_tokens = tokens.to(device_);
  auto d_positions = positions.to(device_);
  auto d_slots = slots.to(device_);

  // call model forward and return the result
}

bool Worker::init() {
  if (device_.is_cuda()) {
    // set device for the working thread. the cuda runtime API is thread safe.
    // it maintains a thread local state for each thread.
    // just need to set once for each working thread.
    at::cuda::set_device(device_.index());
  }

  return true;
}

}  // namespace llm
