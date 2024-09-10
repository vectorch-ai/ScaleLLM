#pragma once

#include <cstddef>
#include <flashinfer/fastdiv.cuh>

namespace flashinfer {

/*!
 * \brief Paged key-value cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 */
template <typename DType, typename IdType>
struct paged_kv_t {
  // number of sequences in the batch
  uint32_t batch_size_;

  uint_fastdiv page_size_;

  // The flattened key-value cache
  // [n_pages, page_size, n_heads, head_dim]
  DType* k_data_;
  DType* v_data_;

  // [nnz_pages] The page indices array
  IdType* indices_;

  // [batch_size + 1] The page indptr array, with the first element 0, the last
  // element nnz_pages
  IdType* indptr_;

  // strides for page, entry and head
  size_t stride_page_;
  size_t stride_n_;
  size_t stride_h_;

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
   */
  __host__ __forceinline__ paged_kv_t(uint32_t num_heads,
                                      uint32_t page_size,
                                      uint32_t head_dim,
                                      uint32_t batch_size,
                                      DType* k_data,
                                      DType* v_data,
                                      IdType* indices,
                                      IdType* indptr)
      : page_size_(page_size),
        batch_size_(batch_size),
        k_data_(k_data),
        v_data_(v_data),
        indices_(indices),
        indptr_(indptr) {
    stride_h_ = head_dim;
    stride_n_ = stride_h_ * num_heads;
    stride_page_ = stride_n_ * page_size;
  }

  // get the kv offset for given request, kv_idx, kv_head_idx and feat_idx
  __device__ __forceinline__ size_t get_kv_offset(uint32_t request_idx,
                                                  uint32_t kv_idx,
                                                  uint32_t kv_head_idx,
                                                  uint32_t feat_idx) const {
    uint32_t page_idx, page_offset;
    page_size_.divmod(kv_idx, page_idx, page_offset);

    // get the page id from block table
    const auto page_idx_base = __ldg(indptr_ + request_idx);
    const auto page_id = __ldg(indices_ + page_idx_base + page_idx);

    return page_id * stride_page_ + page_offset * stride_n_ +
           kv_head_idx * stride_h_ + feat_idx;
  }

  __device__ __forceinline__ DType* k_data(size_t offset) const {
    return k_data_ + offset;
  }

  __device__ __forceinline__ DType* v_data(size_t offset) const {
    return v_data_ + offset;
  }
};

}  // namespace flashinfer
