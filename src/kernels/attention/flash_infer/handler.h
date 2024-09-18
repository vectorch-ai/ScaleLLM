//  Adapted from https://github.com/flashinfer-ai/flashinfer/
#pragma once
// clang-format off
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <functional>


#include <flashinfer/allocator.h>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/attention/logits_post_hook.cuh>
#include <flashinfer/attention/warp_layout.cuh>

// #include "kv_cache.h"

// clang-format on

namespace flashinfer {

// binary search to find the smallest kv_chunk_size that can fit into the grid
// returns kv_chunk_size
inline uint32_t search_kv_chunk_size(
    const uint32_t max_grid_size,
    const uint32_t num_kv_heads,
    const uint32_t min_kv_chunk_size,
    const uint32_t max_kv_chunk_size,
    const std::function<int64_t(int64_t)>& cal_batch_size) {
  int64_t low = min_kv_chunk_size;
  int64_t high = max_kv_chunk_size;
  while (low < high) {
    const int64_t mid = (low + high) / 2;
    const int64_t batch_size = cal_batch_size(mid);
    if (batch_size * num_kv_heads > max_grid_size) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  // low holds the smallest kv_chunk_size that can fit into the grid
  return low;
}

template <typename IdType>
struct SplitParams {
  // whether to split kv
  bool split_kv;
  // the max batch size?
  uint32_t split_max_batch_size;
  // total number of tiles in qo
  uint32_t total_num_tiles_q;
  // total number of partitions
  uint32_t new_batch_size;
  // warp layout
  WarpLayout warp_layout;
  // kv_chunk_size that can fit into the grid
  uint32_t kv_chunk_size;
  // total number of rows in qo
  uint32_t total_num_rows;
  // request idx for each cta
  std::vector<IdType> request_indices;
  // qo_tile_idx for each cta
  std::vector<IdType> qo_tile_indices;
  // kv_tile_idx for each cta
  std::vector<IdType> kv_tile_indices;
  // kv_tile indptr for each row in qo?
  std::vector<IdType> merge_indptr;
  // kv_tile indptr for each request
  std::vector<IdType> o_indptr;
};

template <typename IdType>
SplitParams<IdType> split_input(IdType* qo_indptr_h,
                                IdType* paged_kv_indptr_h,
                                uint32_t batch_size,
                                uint32_t num_qo_heads,
                                uint32_t num_kv_heads,
                                uint32_t head_dim,
                                uint32_t page_size,
                                int32_t num_sm) {
  SplitParams<IdType> split_params;

  const uint32_t gqa_group_size = num_qo_heads / num_kv_heads;
  split_params.total_num_rows = qo_indptr_h[batch_size];

  int num_blocks_per_sm = 2;
  int max_grid_size = num_blocks_per_sm * num_sm;
  split_params.split_max_batch_size = max_grid_size / num_kv_heads;

  // step 1: compute qo_chunk_size
  std::vector<int64_t> packed_qo_len_arr(batch_size);
  std::vector<int64_t> kv_chunk_len_arr(batch_size);
  int64_t max_kv_chunk_len = 0;
  int64_t sum_packed_qo_len = 0;
  for (uint32_t i = 0; i < batch_size; ++i) {
    packed_qo_len_arr[i] =
        int64_t(qo_indptr_h[i + 1] - qo_indptr_h[i]) * int64_t(gqa_group_size);
    auto kv_chunk_len =
        int64_t(paged_kv_indptr_h[i + 1] - paged_kv_indptr_h[i]);
    kv_chunk_len_arr[i] = std::max<int64_t>(kv_chunk_len, 1);
    max_kv_chunk_len = std::max(max_kv_chunk_len, kv_chunk_len);
    sum_packed_qo_len += packed_qo_len_arr[i];
  }
  int64_t avg_packed_qo_len = sum_packed_qo_len / batch_size;
  // WarpLayout: (num_warps_x, num_warps_z, num_frags_x)
  if (avg_packed_qo_len > 64 && head_dim < 256) {
    split_params.warp_layout = WarpLayout::k4x1x2;
  } else {
    auto compute_capacity = GetCudaComputeCapability();
    if (compute_capacity.first >= 8) {
      // Ampere or newer
      if (avg_packed_qo_len > 16) {
        split_params.warp_layout = WarpLayout::k4x1x1;
      } else {
        // avg_packed_qo_len <= 16
        split_params.warp_layout = WarpLayout::k1x4x1;
      }
    } else {
      // NOTE(Zihao): not enough shared memory on Turing for 1x4x1 layout
      split_params.warp_layout = WarpLayout::k4x1x1;
    }
  }
  const uint32_t qo_chunk_size = get_num_rows_per_cta(split_params.warp_layout);

  // lambda to calculate batch_size given kv_chunk_size
  auto cal_batch_size = [&](int64_t kv_chunk_size) -> int64_t {
    int64_t batch_size = 0;
    for (size_t i = 0; i < packed_qo_len_arr.size(); ++i) {
      batch_size += ceil_div(packed_qo_len_arr[i], qo_chunk_size) *
                    ceil_div(kv_chunk_len_arr[i], kv_chunk_size);
    }
    return batch_size;
  };

  const uint32_t min_kv_chunk_size = std::max((128 / page_size), 1U);
  // step 2: determine kv_chunk_size
  auto kv_chunk_size = search_kv_chunk_size(max_grid_size,
                                            num_kv_heads,
                                            min_kv_chunk_size,
                                            max_kv_chunk_len,
                                            cal_batch_size);

  split_params.split_kv = kv_chunk_size < max_kv_chunk_len;
  split_params.new_batch_size = cal_batch_size(kv_chunk_size);

  // step 3: split qo_indptr and kv_indptr
  split_params.merge_indptr.push_back(0);
  split_params.o_indptr.push_back(0);
  split_params.total_num_tiles_q = 0;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    int64_t packed_qo_len = packed_qo_len_arr[request_idx];
    int64_t kv_len = std::max(int(kv_chunk_len_arr[request_idx]), 1);
    int64_t num_tiles_q = ceil_div(packed_qo_len, qo_chunk_size);
    int64_t num_tiles_kv = ceil_div(kv_len, kv_chunk_size);
    split_params.total_num_tiles_q += num_tiles_q;

    for (uint32_t q_tile_idx = 0; q_tile_idx < num_tiles_q; ++q_tile_idx) {
      for (uint32_t kv_tile_idx = 0; kv_tile_idx < num_tiles_kv;
           ++kv_tile_idx) {
        split_params.request_indices.push_back(request_idx);
        split_params.qo_tile_indices.push_back(q_tile_idx);
        split_params.kv_tile_indices.push_back(kv_tile_idx);
      }
    }

    int64_t qo_len = packed_qo_len / gqa_group_size;
    for (uint32_t row = 0; row < qo_len; ++row) {
      split_params.merge_indptr.push_back(split_params.merge_indptr.back() +
                                          num_tiles_kv);
    }
    split_params.o_indptr.push_back(split_params.o_indptr.back() +
                                    qo_len * num_tiles_kv);
  }

  // step 4: multiply kv_chunk_size by page_size to get kv length per chunk
  split_params.kv_chunk_size = kv_chunk_size * page_size;

  return split_params;
}

class BatchPrefillHandler {
 public:
  template <typename IdType>
  IdType* GetRequestIndices() const {
    return (IdType*)request_indices_;
  }

  template <typename IdType>
  IdType* GetQOTileIndices() const {
    return (IdType*)qo_tile_indices_;
  }

  template <typename IdType>
  IdType* GetKVTileIndices() const {
    return (IdType*)kv_tile_indices_;
  }

  template <typename IdType>
  IdType* GetMergeIndptr() const {
    return (IdType*)merge_indptr_;
  }

  template <typename IdType>
  IdType* GetOIndptr() const {
    return (IdType*)o_indptr_;
  }

  template <typename IdType>
  IdType* GetKVChunkSizePtr() const {
    return (IdType*)kv_chunk_size_ptr_;
  }

  template <typename DType>
  DType* GetTempV() const {
    return (DType*)tmp_v_;
  }

  bool* GetBlockValidMask() const { return block_valid_mask_; }

  float* GetTempS() const { return tmp_s_; }

  uint32_t GetPaddedBatchSize() const { return padded_batch_size_; }

  WarpLayout GetWarpLayout() const { return warp_layout_; }

  uint32_t GetTotalNumRows() const { return total_num_rows_; }

  void UpdatePageLockedBufferSize(size_t int_workspace_size_in_bytes) {
    cudaFreeHost(page_locked_buffer_);
    cudaMallocHost(&page_locked_buffer_, int_workspace_size_in_bytes);
  }

  template <typename DTypeOut, typename IdType>
  cudaError_t Plan(void* float_buffer,
                   size_t float_workspace_size_in_bytes,
                   void* int_buffer,
                   size_t int_workspace_size_in_bytes,
                   IdType* qo_indptr_h,
                   IdType* paged_kv_indptr_h,
                   uint32_t batch_size,
                   uint32_t num_qo_heads,
                   uint32_t num_kv_heads,
                   uint32_t head_dim,
                   uint32_t page_size) {
    Clear();
    if (num_qo_heads % num_kv_heads != 0) {
      std::ostringstream err_msg;
      err_msg << "num_qo_heads " << num_qo_heads
              << " should be divisible by num_kv_heads " << num_kv_heads;
      throw std::invalid_argument(err_msg.str());
    }

    int num_sm = 0;
    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
        &num_sm, cudaDevAttrMultiProcessorCount, dev_id));

    auto split_params = split_input(qo_indptr_h,
                                    paged_kv_indptr_h,
                                    batch_size,
                                    num_qo_heads,
                                    num_kv_heads,
                                    head_dim,
                                    page_size,
                                    num_sm);
    const uint32_t qo_tile_size = get_num_rows_per_cta(warp_layout_);

    if (IsCUDAGraphEnabled()) {
      padded_batch_size_ = std::max(split_params.split_max_batch_size,
                                    split_params.total_num_tiles_q);
      AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
      request_indices_ =
          int_allocator.aligned_alloc<void>(sizeof(IdType) * padded_batch_size_,
                                            16,
                                            "batch_prefill_request_indices");
      void* request_indices_h_ = page_locked_buffer_;
      qo_tile_indices_ =
          int_allocator.aligned_alloc<void>(sizeof(IdType) * padded_batch_size_,
                                            16,
                                            "batch_prefill_qo_tile_indices");
      void* qo_tile_indices_h_ =
          (char*)page_locked_buffer_ +
          ((char*)qo_tile_indices_ - (char*)request_indices_);
      kv_tile_indices_ =
          int_allocator.aligned_alloc<void>(sizeof(IdType) * padded_batch_size_,
                                            16,
                                            "batch_prefill_kv_tile_indices");
      void* kv_tile_indices_h_ =
          (char*)page_locked_buffer_ +
          ((char*)kv_tile_indices_ - (char*)request_indices_);
      o_indptr_ = int_allocator.aligned_alloc<void>(
          sizeof(IdType) * (batch_size + 1), 16, "batch_prefill_o_indptr");
      void* o_indptr_h_ = (char*)page_locked_buffer_ +
                          ((char*)o_indptr_ - (char*)request_indices_);
      kv_chunk_size_ptr_ = int_allocator.aligned_alloc<void>(
          sizeof(IdType), 1, "batch_prefill_kv_chunk_size_ptr");
      void* kv_chunk_size_ptr_h_ =
          (char*)page_locked_buffer_ +
          ((char*)kv_chunk_size_ptr_ - (char*)request_indices_);
      *(IdType*)kv_chunk_size_ptr_h_ = split_params.kv_chunk_size;
      if (split_params.total_num_tiles_q < split_params.split_max_batch_size) {
        // need merge_indptr
        merge_indptr_ = int_allocator.aligned_alloc<void>(
            sizeof(IdType) * (total_num_rows_ + 1),
            16,
            "batch_prefill_merge_indptr");
        void* merge_indptr_h_ =
            (char*)page_locked_buffer_ +
            ((char*)merge_indptr_ - (char*)request_indices_);
        std::copy(split_params.merge_indptr.begin(),
                  split_params.merge_indptr.end(),
                  (IdType*)merge_indptr_h_);
        block_valid_mask_ =
            int_allocator.aligned_alloc<bool>(sizeof(bool) * padded_batch_size_,
                                              16,
                                              "batch_prefill_block_valid_mask");
        bool* block_valid_mask_h_ =
            (bool*)page_locked_buffer_ +
            ((bool*)block_valid_mask_ - (bool*)request_indices_);
        for (uint32_t i = 0; i < padded_batch_size_; ++i) {
          block_valid_mask_h_[i] = i < split_params.new_batch_size;
        }
      } else {
        // total_num_tiles_q >= split_max_batch_size, we don't need to perform
        // the second round at all.
        merge_indptr_ = nullptr;
        block_valid_mask_ = nullptr;
      }
      std::copy(split_params.request_indices.begin(),
                split_params.request_indices.end(),
                (IdType*)request_indices_h_);
      std::copy(split_params.qo_tile_indices.begin(),
                split_params.qo_tile_indices.end(),
                (IdType*)qo_tile_indices_h_);
      std::copy(split_params.kv_tile_indices.begin(),
                split_params.kv_tile_indices.end(),
                (IdType*)kv_tile_indices_h_);
      std::copy(split_params.o_indptr.begin(),
                split_params.o_indptr.end(),
                (IdType*)o_indptr_h_);

      size_t num_bytes_to_copy =
          (char*)int_allocator.ptr - (char*)request_indices_;
      FLASHINFER_CUDA_CALL(cudaMemcpyAsync(request_indices_,
                                           page_locked_buffer_,
                                           num_bytes_to_copy,
                                           cudaMemcpyHostToDevice,
                                           stream_))

      if (split_params.total_num_tiles_q < split_params.split_max_batch_size) {
        AlignedAllocator float_allocator(float_buffer,
                                         float_workspace_size_in_bytes);
        tmp_v_ = float_allocator.aligned_alloc<void>(
            num_qo_heads * split_params.split_max_batch_size * qo_tile_size *
                head_dim * sizeof(DTypeOut),
            16,
            "batch_prefill_tmp_v");
        tmp_s_ = float_allocator.aligned_alloc<float>(
            num_qo_heads * split_params.split_max_batch_size * qo_tile_size *
                sizeof(float),
            16,
            "batch_prefill_tmp_s");
      } else {
        tmp_v_ = nullptr;
        tmp_s_ = nullptr;
      }
    } else {
      padded_batch_size_ = split_params.new_batch_size;
      AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
      request_indices_ = int_allocator.aligned_alloc<void>(
          sizeof(IdType) * split_params.request_indices.size(),
          16,
          "batch_prefill_request_indices");
      void* request_indices_h_ = page_locked_buffer_;
      qo_tile_indices_ = int_allocator.aligned_alloc<void>(
          sizeof(IdType) * split_params.qo_tile_indices.size(),
          16,
          "batch_prefill_qo_tile_indices");
      void* qo_tile_indices_h_ =
          (char*)page_locked_buffer_ +
          ((char*)qo_tile_indices_ - (char*)request_indices_);
      kv_tile_indices_ = int_allocator.aligned_alloc<void>(
          sizeof(IdType) * split_params.kv_tile_indices.size(),
          16,
          "batch_prefill_kv_tile_indices");
      void* kv_tile_indices_h_ =
          (char*)page_locked_buffer_ +
          ((char*)kv_tile_indices_ - (char*)request_indices_);
      if (split_params.split_kv) {
        // need merge_indptr when split_kv is true
        merge_indptr_ = int_allocator.aligned_alloc<void>(
            sizeof(IdType) * split_params.merge_indptr.size(),
            16,
            "batch_prefill_merge_indptr");
        void* merge_indptr_h_ =
            (char*)page_locked_buffer_ +
            ((char*)merge_indptr_ - (char*)request_indices_);
        std::copy(split_params.merge_indptr.begin(),
                  split_params.merge_indptr.end(),
                  (IdType*)merge_indptr_h_);
      }
      o_indptr_ = int_allocator.aligned_alloc<void>(
          sizeof(IdType) * split_params.o_indptr.size(),
          16,
          "batch_prefill_o_indptr");
      void* o_indptr_h_ = (char*)page_locked_buffer_ +
                          ((char*)o_indptr_ - (char*)request_indices_);
      kv_chunk_size_ptr_ = int_allocator.aligned_alloc<void>(
          sizeof(IdType), 1, "batch_prefill_kv_chunk_size_ptr");
      void* kv_chunk_size_ptr_h_ =
          (char*)page_locked_buffer_ +
          ((char*)kv_chunk_size_ptr_ - (char*)request_indices_);
      *(IdType*)kv_chunk_size_ptr_h_ = split_params.kv_chunk_size;
      std::copy(split_params.request_indices.begin(),
                split_params.request_indices.end(),
                (IdType*)request_indices_h_);
      std::copy(split_params.qo_tile_indices.begin(),
                split_params.qo_tile_indices.end(),
                (IdType*)qo_tile_indices_h_);
      std::copy(split_params.kv_tile_indices.begin(),
                split_params.kv_tile_indices.end(),
                (IdType*)kv_tile_indices_h_);
      std::copy(split_params.o_indptr.begin(),
                split_params.o_indptr.end(),
                (IdType*)o_indptr_h_);
      size_t num_bytes_to_copy =
          (char*)int_allocator.ptr - (char*)request_indices_;

      FLASHINFER_CUDA_CALL(cudaMemcpyAsync(request_indices_,
                                           page_locked_buffer_,
                                           num_bytes_to_copy,
                                           cudaMemcpyHostToDevice,
                                           stream_))

      if (split_params.split_kv) {
        AlignedAllocator float_allocator(float_buffer,
                                         float_workspace_size_in_bytes);
        // [n_kv_tiles, n_tokens, n_heads, head_dim]
        // new_batch_size = n_kv_tiles * n_q_tiles
        // n_tokens = n_q_tiles * qo_tile_size
        // n_kv_tiles * n_tokens = new_batch_size * qo_tile_size
        tmp_v_ = float_allocator.aligned_alloc<void>(
            split_params.new_batch_size * qo_tile_size * num_qo_heads *
                head_dim * sizeof(DTypeOut),
            16,
            "batch_prefill_tmp_v");

        // [n_kv_tiles, n_tokens, n_heads]
        tmp_s_ = float_allocator.aligned_alloc<float>(
            split_params.new_batch_size * qo_tile_size * num_qo_heads *
                sizeof(float),
            16,
            "batch_prefill_tmp_s");
      } else {
        tmp_v_ = nullptr;
        tmp_s_ = nullptr;
      }

      block_valid_mask_ = nullptr;
    }
    return cudaSuccess;
  }

  void Clear() {
    request_indices_ = nullptr;
    qo_tile_indices_ = nullptr;
    kv_tile_indices_ = nullptr;
    merge_indptr_ = nullptr;
    o_indptr_ = nullptr;
    kv_chunk_size_ptr_ = nullptr;
    tmp_v_ = nullptr;
    tmp_s_ = nullptr;
    block_valid_mask_ = nullptr;
    total_num_rows_ = 0U;
    padded_batch_size_ = 0U;
    warp_layout_ = WarpLayout::k4x1x2;
  }

  cudaStream_t GetCUDAStream() const { return stream_; }

  void SetCUDAStream(cudaStream_t stream) { stream_ = stream; }

  bool IsCUDAGraphEnabled() const { return enable_cuda_graph_; }

  BatchPrefillHandler(bool enable_cuda_graph = false)
      : request_indices_(nullptr),
        qo_tile_indices_(nullptr),
        kv_tile_indices_(nullptr),
        merge_indptr_(nullptr),
        o_indptr_(nullptr),
        kv_chunk_size_ptr_(nullptr),
        tmp_v_(nullptr),
        tmp_s_(nullptr),
        block_valid_mask_(nullptr),
        total_num_rows_(0U),
        padded_batch_size_(0U),
        warp_layout_(WarpLayout::k4x1x2),
        enable_cuda_graph_(enable_cuda_graph),
        stream_(nullptr) {
    cudaMallocHost(&page_locked_buffer_, 8 * 1024 * 1024);
  }
  ~BatchPrefillHandler() { cudaFreeHost(page_locked_buffer_); }

 protected:
  void* page_locked_buffer_;
  void* request_indices_;
  void* qo_tile_indices_;
  void* kv_tile_indices_;
  void* merge_indptr_;
  void* o_indptr_;
  void* kv_chunk_size_ptr_;
  void* tmp_v_;
  float* tmp_s_;
  bool* block_valid_mask_;
  uint32_t total_num_rows_;
  uint32_t padded_batch_size_;
  WarpLayout warp_layout_;
  bool enable_cuda_graph_;
  cudaStream_t stream_;
};

}  // namespace flashinfer
