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


#include <flashinfer/allocator.h>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/attention/logits_post_hook.cuh>
#include <flashinfer/attention/warp_layout.cuh>

// #include "kv_cache.h"

// clang-format on

namespace flashinfer {

inline std::tuple<bool, uint32_t, uint32_t> PrefillBinarySearchKVChunkSize(
    const uint32_t max_grid_size,
    const uint32_t num_kv_heads,
    const std::vector<int64_t>& packed_qo_len_arr,
    const std::vector<int64_t>& kv_len_arr,
    const uint32_t qo_chunk_size,
    const uint32_t min_kv_chunk_size = 1) {
  int64_t low = min_kv_chunk_size, high = 0;
  int64_t batch_size = packed_qo_len_arr.size();
  int64_t max_kv_len = 0;
  for (const int64_t& kv_len : kv_len_arr) {
    max_kv_len = std::max(max_kv_len, kv_len);
  }
  high = max_kv_len;
  int64_t new_batch_size;
  while (low < high) {
    int64_t mid = (low + high) / 2;
    new_batch_size = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      new_batch_size += ceil_div(packed_qo_len_arr[i], qo_chunk_size) *
                        ceil_div(kv_len_arr[i], mid);
    }
    if (new_batch_size * num_kv_heads > max_grid_size) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  new_batch_size = 0;
  for (uint32_t i = 0; i < batch_size; ++i) {
    new_batch_size += ceil_div(packed_qo_len_arr[i], qo_chunk_size) *
                      ceil_div(std::max(int(kv_len_arr[i]), 1), low);
  }
  return {low < max_kv_len, low, new_batch_size};
}

template <typename IdType>
cudaError_t PrefillSplitQOKVIndptr(bool& split_kv,
                                   uint32_t& split_max_batch_size,
                                   uint32_t& total_num_tiles_q,
                                   uint32_t& new_batch_size,
                                   WarpLayout& warp_layout,
                                   uint32_t& kv_chunk_size,
                                   uint32_t& total_num_rows,
                                   std::vector<IdType>& request_indices,
                                   std::vector<IdType>& qo_tile_indices,
                                   std::vector<IdType>& kv_tile_indices,
                                   std::vector<IdType>& merge_indptr,
                                   std::vector<IdType>& o_indptr,
                                   IdType* qo_indptr_h,
                                   IdType* paged_kv_indptr_h,
                                   uint32_t batch_size,
                                   uint32_t num_qo_heads,
                                   uint32_t num_kv_heads,
                                   uint32_t head_dim,
                                   uint32_t page_size) {
  request_indices.clear();
  qo_tile_indices.clear();
  kv_tile_indices.clear();
  merge_indptr.clear();
  o_indptr.clear();
  merge_indptr.push_back(0);
  o_indptr.push_back(0);

  const uint32_t gqa_group_size = num_qo_heads / num_kv_heads;
  total_num_rows = qo_indptr_h[batch_size];

  // step 0: get the number of SMs
  int num_sm = 0;
  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
  int num_blocks_per_sm = 2;
  int max_grid_size = num_blocks_per_sm * num_sm;
  split_max_batch_size = max_grid_size / num_kv_heads;

  // step 1: compute qo_chunk_size
  std::vector<int64_t> packed_qo_len_arr(batch_size), kv_len_arr(batch_size);
  int64_t sum_packed_qo_len = 0;
  for (uint32_t i = 0; i < batch_size; ++i) {
    packed_qo_len_arr[i] =
        int64_t(qo_indptr_h[i + 1] - qo_indptr_h[i]) * int64_t(gqa_group_size);
    kv_len_arr[i] = int64_t(paged_kv_indptr_h[i + 1] - paged_kv_indptr_h[i]);
    sum_packed_qo_len += packed_qo_len_arr[i];
  }
  int64_t avg_packed_qo_len = sum_packed_qo_len / batch_size;
  if (avg_packed_qo_len > 64 && head_dim < 256) {
    warp_layout = WarpLayout::k4x1x2;  // (num_warps_x = 4, num_warps_z = 1,
                                       // num_frags_x = 2)
  } else {
    auto compute_capacity = GetCudaComputeCapability();
    if (compute_capacity.first >= 8) {
      // Ampere or newer
      if (avg_packed_qo_len > 16) {
        warp_layout = WarpLayout::k4x1x1;  // (num_warps_x = 4, num_warps_z = 1,
                                           // num_frags_x = 1)
      } else {
        // avg_packed_qo_len <= 16
        warp_layout = WarpLayout::k1x4x1;  // (num_warps_x = 1, num_warps_z = 4,
                                           // num_frags_x = 1)
      }
    } else {
      // NOTE(Zihao): not enough shared memory on Turing for 1x4x1 layout
      warp_layout = WarpLayout::k4x1x1;
    }
  }
  const uint32_t qo_chunk_size = get_num_rows_per_cta(warp_layout);

  // step 2: determine kv_chunk_size
  std::tie(split_kv, kv_chunk_size, new_batch_size) =
      PrefillBinarySearchKVChunkSize(
          max_grid_size,
          num_kv_heads,
          packed_qo_len_arr,
          kv_len_arr,
          qo_chunk_size,
          /*min_kv_chunk_size=*/std::max((128 / page_size), 1U));

  // step 3: split qo_indptr and kv_indptr
  total_num_tiles_q = 0;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    int64_t packed_qo_len = packed_qo_len_arr[request_idx],
            kv_len = std::max(int(kv_len_arr[request_idx]), 1);
    int64_t num_tiles_q = ceil_div(packed_qo_len, qo_chunk_size),
            num_tiles_kv = ceil_div(kv_len, kv_chunk_size);
    total_num_tiles_q += num_tiles_q;
    for (uint32_t q_tile_idx = 0; q_tile_idx < num_tiles_q; ++q_tile_idx) {
      for (uint32_t kv_tile_idx = 0; kv_tile_idx < num_tiles_kv;
           ++kv_tile_idx) {
        request_indices.push_back(request_idx);
        qo_tile_indices.push_back(q_tile_idx);
        kv_tile_indices.push_back(kv_tile_idx);
      }
    }

    int64_t qo_len = packed_qo_len / gqa_group_size;
    for (uint32_t row = 0; row < qo_len; ++row) {
      merge_indptr.push_back(merge_indptr.back() + num_tiles_kv);
    }
    o_indptr.push_back(o_indptr.back() + qo_len * num_tiles_kv);
  }

  // step 4: multiply kv_chunk_size by page_size
  kv_chunk_size *= page_size;

  return cudaSuccess;
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
    bool split_kv;
    uint32_t split_max_batch_size, new_batch_size, total_num_tiles_q,
        kv_chunk_size;
    std::vector<IdType> request_indices_vec, qo_tile_indices_vec,
        kv_tile_indices_vec, merge_indptr_vec, o_indptr_vec;
    FLASHINFER_CUDA_CALL(PrefillSplitQOKVIndptr(split_kv,
                                                split_max_batch_size,
                                                total_num_tiles_q,
                                                new_batch_size,
                                                warp_layout_,
                                                kv_chunk_size,
                                                total_num_rows_,
                                                request_indices_vec,
                                                qo_tile_indices_vec,
                                                kv_tile_indices_vec,
                                                merge_indptr_vec,
                                                o_indptr_vec,
                                                qo_indptr_h,
                                                paged_kv_indptr_h,
                                                batch_size,
                                                num_qo_heads,
                                                num_kv_heads,
                                                head_dim,
                                                page_size));
    const uint32_t qo_tile_size = get_num_rows_per_cta(warp_layout_);

    if (IsCUDAGraphEnabled()) {
      padded_batch_size_ = std::max(split_max_batch_size, total_num_tiles_q);
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
      *(IdType*)kv_chunk_size_ptr_h_ = kv_chunk_size;
      if (total_num_tiles_q < split_max_batch_size) {
        // need merge_indptr
        merge_indptr_ = int_allocator.aligned_alloc<void>(
            sizeof(IdType) * (total_num_rows_ + 1),
            16,
            "batch_prefill_merge_indptr");
        void* merge_indptr_h_ =
            (char*)page_locked_buffer_ +
            ((char*)merge_indptr_ - (char*)request_indices_);
        std::copy(merge_indptr_vec.begin(),
                  merge_indptr_vec.end(),
                  (IdType*)merge_indptr_h_);
        block_valid_mask_ =
            int_allocator.aligned_alloc<bool>(sizeof(bool) * padded_batch_size_,
                                              16,
                                              "batch_prefill_block_valid_mask");
        bool* block_valid_mask_h_ =
            (bool*)page_locked_buffer_ +
            ((bool*)block_valid_mask_ - (bool*)request_indices_);
        for (uint32_t i = 0; i < padded_batch_size_; ++i) {
          block_valid_mask_h_[i] = i < new_batch_size;
        }
      } else {
        // total_num_tiles_q >= split_max_batch_size, we don't need to perform
        // the second round at all.
        merge_indptr_ = nullptr;
        block_valid_mask_ = nullptr;
      }
      std::copy(request_indices_vec.begin(),
                request_indices_vec.end(),
                (IdType*)request_indices_h_);
      std::copy(qo_tile_indices_vec.begin(),
                qo_tile_indices_vec.end(),
                (IdType*)qo_tile_indices_h_);
      std::copy(kv_tile_indices_vec.begin(),
                kv_tile_indices_vec.end(),
                (IdType*)kv_tile_indices_h_);
      std::copy(o_indptr_vec.begin(), o_indptr_vec.end(), (IdType*)o_indptr_h_);

      size_t num_bytes_to_copy =
          (char*)int_allocator.ptr - (char*)request_indices_;
      FLASHINFER_CUDA_CALL(cudaMemcpyAsync(request_indices_,
                                           page_locked_buffer_,
                                           num_bytes_to_copy,
                                           cudaMemcpyHostToDevice,
                                           stream_))

      if (total_num_tiles_q < split_max_batch_size) {
        AlignedAllocator float_allocator(float_buffer,
                                         float_workspace_size_in_bytes);
        tmp_v_ = float_allocator.aligned_alloc<void>(
            num_qo_heads * split_max_batch_size * qo_tile_size * head_dim *
                sizeof(DTypeOut),
            16,
            "batch_prefill_tmp_v");
        tmp_s_ = float_allocator.aligned_alloc<float>(
            num_qo_heads * split_max_batch_size * qo_tile_size * sizeof(float),
            16,
            "batch_prefill_tmp_s");
      } else {
        tmp_v_ = nullptr;
        tmp_s_ = nullptr;
      }
    } else {
      padded_batch_size_ = new_batch_size;
      AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
      request_indices_ = int_allocator.aligned_alloc<void>(
          sizeof(IdType) * request_indices_vec.size(),
          16,
          "batch_prefill_request_indices");
      void* request_indices_h_ = page_locked_buffer_;
      qo_tile_indices_ = int_allocator.aligned_alloc<void>(
          sizeof(IdType) * qo_tile_indices_vec.size(),
          16,
          "batch_prefill_qo_tile_indices");
      void* qo_tile_indices_h_ =
          (char*)page_locked_buffer_ +
          ((char*)qo_tile_indices_ - (char*)request_indices_);
      kv_tile_indices_ = int_allocator.aligned_alloc<void>(
          sizeof(IdType) * kv_tile_indices_vec.size(),
          16,
          "batch_prefill_kv_tile_indices");
      void* kv_tile_indices_h_ =
          (char*)page_locked_buffer_ +
          ((char*)kv_tile_indices_ - (char*)request_indices_);
      if (split_kv) {
        // need merge_indptr when split_kv is true
        merge_indptr_ = int_allocator.aligned_alloc<void>(
            sizeof(IdType) * merge_indptr_vec.size(),
            16,
            "batch_prefill_merge_indptr");
        void* merge_indptr_h_ =
            (char*)page_locked_buffer_ +
            ((char*)merge_indptr_ - (char*)request_indices_);
        std::copy(merge_indptr_vec.begin(),
                  merge_indptr_vec.end(),
                  (IdType*)merge_indptr_h_);
      }
      o_indptr_ = int_allocator.aligned_alloc<void>(
          sizeof(IdType) * o_indptr_vec.size(), 16, "batch_prefill_o_indptr");
      void* o_indptr_h_ = (char*)page_locked_buffer_ +
                          ((char*)o_indptr_ - (char*)request_indices_);
      kv_chunk_size_ptr_ = int_allocator.aligned_alloc<void>(
          sizeof(IdType), 1, "batch_prefill_kv_chunk_size_ptr");
      void* kv_chunk_size_ptr_h_ =
          (char*)page_locked_buffer_ +
          ((char*)kv_chunk_size_ptr_ - (char*)request_indices_);
      *(IdType*)kv_chunk_size_ptr_h_ = kv_chunk_size;
      std::copy(request_indices_vec.begin(),
                request_indices_vec.end(),
                (IdType*)request_indices_h_);
      std::copy(qo_tile_indices_vec.begin(),
                qo_tile_indices_vec.end(),
                (IdType*)qo_tile_indices_h_);
      std::copy(kv_tile_indices_vec.begin(),
                kv_tile_indices_vec.end(),
                (IdType*)kv_tile_indices_h_);
      std::copy(o_indptr_vec.begin(), o_indptr_vec.end(), (IdType*)o_indptr_h_);
      size_t num_bytes_to_copy =
          (char*)int_allocator.ptr - (char*)request_indices_;

      FLASHINFER_CUDA_CALL(cudaMemcpyAsync(request_indices_,
                                           page_locked_buffer_,
                                           num_bytes_to_copy,
                                           cudaMemcpyHostToDevice,
                                           stream_))

      if (split_kv) {
        AlignedAllocator float_allocator(float_buffer,
                                         float_workspace_size_in_bytes);
        tmp_v_ = float_allocator.aligned_alloc<void>(
            num_qo_heads * new_batch_size * qo_tile_size * head_dim *
                sizeof(DTypeOut),
            16,
            "batch_prefill_tmp_v");
        tmp_s_ = float_allocator.aligned_alloc<float>(
            num_qo_heads * new_batch_size * qo_tile_size * sizeof(float),
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
