#pragma once

#include <torch/torch.h>

#include <flashinfer/attention/layout.cuh>

#include "handler.h"

class BatchPrefillWithPagedKVCachePyTorchWrapper {
 public:
  void Plan(torch::Tensor float_workspace_buffer,
            torch::Tensor int_workspace_buffer,
            torch::Tensor qo_indptr,
            torch::Tensor page_kv_indptr,
            unsigned int batch_size,
            unsigned int num_qo_heads,
            unsigned int num_kv_heads,
            unsigned int head_dim,
            unsigned page_size,
            torch::Tensor empty_q_data);
  bool IsCUDAGraphEnabled() const { return handler_->IsCUDAGraphEnabled(); }
  void UpdatePageLockedBufferSize(uint32_t int_workspace_size_in_bytes);
  std::vector<torch::Tensor> Run(torch::Tensor q,
                                 torch::Tensor qo_indptr,
                                 std::optional<torch::Tensor> paged_kv_cache,
                                 std::optional<torch::Tensor> paged_k_cache,
                                 std::optional<torch::Tensor> paged_v_cache,
                                 torch::Tensor paged_kv_indptr,
                                 torch::Tensor paged_kv_indices,
                                 torch::Tensor paged_kv_last_page_len,
                                 bool causal,
                                 unsigned int pos_encoding_mode,
                                 bool allow_fp16_qk_reduction,
                                 int window_left,
                                 float logits_soft_cap,
                                 float sm_scale,
                                 float rope_scale,
                                 float rope_theta,
                                 bool return_lse);
  std::vector<torch::Tensor> RunCustomMask(
      torch::Tensor q,
      torch::Tensor qo_indptr,
      std::optional<torch::Tensor> paged_kv_cache,
      std::optional<torch::Tensor> paged_k_cache,
      std::optional<torch::Tensor> paged_v_cache,
      torch::Tensor paged_kv_indptr,
      torch::Tensor paged_kv_indices,
      torch::Tensor paged_kv_last_page_len,
      torch::Tensor packed_custom_mask,
      torch::Tensor qk_indptr,
      unsigned int pos_encoding_mode,
      bool allow_fp16_qk_reduction,
      int window_left,
      float logits_soft_cap,
      float sm_scale,
      float rope_scale,
      float rope_theta,
      bool return_lse);
  BatchPrefillWithPagedKVCachePyTorchWrapper(unsigned int layout,
                                             bool enable_cuda_graph)
      : kv_layout_(flashinfer::QKVLayout(layout)),
        handler_(std::make_shared<flashinfer::BatchPrefillHandler>(
            enable_cuda_graph)) {}

 private:
  std::shared_ptr<flashinfer::BatchPrefillHandler> handler_;
};