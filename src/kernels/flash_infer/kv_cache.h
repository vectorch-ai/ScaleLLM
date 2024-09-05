#pragma once

#include <flashinfer/fastdiv.cuh>

namespace flashinfer {

/*!
 * \brief Paged key-value cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 */
template <typename DType, typename IdType>
struct paged_kv_t {
  // number of samples in the batch
  uint32_t batch_size;

  uint_fastdiv page_size;
  uint32_t num_heads;
  uint32_t head_dim;

  // strides for page, entry and head
  uint32_t stride_page;
  uint32_t stride_n;
  uint32_t stride_h;

  // The flattened key-value cache
  // [n_pages, page_size, n_heads, head_dim]
  DType* k_data;
  DType* v_data;

  // [nnz_pages] The page indices array
  IdType* indices;

  // [batch_size + 1] The page indptr array, with the first element 0, the last
  // element nnz_pages
  IdType* indptr;

  // TODO: replace with cu_kv_seq_lens
  // [batch_size] The offset of the last page for each request in the batch
  IdType* last_page_len;

  /*!
   * \brief Construct an empty paged key-value cache
   */
  __host__ __device__ __forceinline__ paged_kv_t()
      : num_heads(0),
        page_size(0),
        head_dim(0),
        batch_size(0),
        stride_page(0),
        stride_n(0),
        stride_h(0),
        k_data(nullptr),
        v_data(nullptr),
        indices(nullptr),
        indptr(nullptr),
        last_page_len(nullptr) {}

  /*!
   * \brief Construct a paged key-value cache
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param layout The layout of last 3 dimensions in KV-Cache.
   * \param kv_data The flattened key-value cache
   * \param k_data The flattened key cache
   * \param v_data The flattened value cache
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each sequence
   * \note This constructor should only be used when page_storage ==
   * kIndices
   */
  __host__ __forceinline__ paged_kv_t(uint32_t num_heads,
                                      uint32_t page_size,
                                      uint32_t head_dim,
                                      uint32_t batch_size,
                                      DType* k_data,
                                      DType* v_data,
                                      IdType* indices,
                                      IdType* indptr,
                                      IdType* last_page_len)
      : num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        batch_size(batch_size),
        k_data(k_data),
        v_data(v_data),
        indices(indices),
        indptr(indptr),
        last_page_len(last_page_len) {
    stride_page = page_size * num_heads * head_dim;
    stride_n = num_heads * head_dim;
    stride_h = head_dim;
  }

  __host__ __device__ __forceinline__ int64_t kv_ptr_delta() const {
    return num_heads * page_size * head_dim;
  }

  /*!
   * \brief Compute the offset of element in the allocated buffer.
   * \param page_idx The page index
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   */
  __host__ __device__ __forceinline__ size_t
  get_elem_offset(size_t page_idx,
                  size_t head_idx,
                  size_t entry_idx,
                  size_t feat_idx) const {
    return page_idx * stride_page + head_idx * stride_h + entry_idx * stride_n +
           feat_idx;
  }

  __device__ __forceinline__ DType* get_k_ptr(IdType page_iter,
                                              uint32_t head_idx,
                                              uint32_t entry_idx,
                                              uint32_t feat_idx) const {
    return k_data +
           get_elem_offset(
               __ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
  }

  __device__ __forceinline__ DType* get_v_ptr(IdType page_iter,
                                              uint32_t head_idx,
                                              uint32_t entry_idx,
                                              uint32_t feat_idx) const {
    return v_data +
           get_elem_offset(
               __ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
  }
};

}  // namespace flashinfer
