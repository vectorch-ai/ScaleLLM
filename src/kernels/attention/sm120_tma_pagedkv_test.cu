#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "cute/swizzle_layout.hpp"
#include "gather_tma_tensor.hpp"

namespace llm {
using namespace cute;

template <class ElementType, class SmemLayout>
struct SharedStorage {
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem;
  alignas(16) cute::uint64_t tma_load_mbar[1];
};

template <class T,
          class TiledCopy,
          class CTA_Tiler,
          class GmemLayout,
          class SmemLayout>
__global__ void tma_test_device_cute(T* g_out,
                                     int const* block_ids,
                                     int block_size,
                                     CUTE_GRID_CONSTANT TiledCopy const tma,
                                     CTA_Tiler cta_tiler,
                                     GmemLayout gmem_layout,
                                     SmemLayout smem_layout) {
  using namespace cute;

  CUTE_STATIC_ASSERT_V(product_each(shape(cta_tiler)) ==
                       product_each(shape(smem_layout)));

  // Use Shared Storage structure to allocate and distribute aligned SMEM
  // addresses
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<T, SmemLayout>;
  SharedStorage& shared_storage =
      *reinterpret_cast<SharedStorage*>(shared_memory);

  // Construct SMEM tensor
  // (CTA_TILE_M,CTA_TILE_N,...)
  Tensor sA =
      make_tensor(make_smem_ptr(shared_storage.smem.begin()), smem_layout);
  Tensor sA_no_siwzzle = make_tensor(make_smem_ptr(shared_storage.smem.begin()),
                                     get_nonswizzle_portion(smem_layout));

  // Shared memory barriers use 64bits in SMEM for synchronization
  uint64_t* tma_load_mbar = shared_storage.tma_load_mbar;

  // TMA requires special handling of strides to deal with coord codomain
  // mapping Represent the full tensors -- get these from TMA
  // Tensor mA = tma.get_tma_tensor(shape(gmem_layout));

  auto coord_transform = [block_ids, block_size](auto coord) {
    constexpr int I = 1;
    const int idx = get<I>(coord);
    const int blk_idx = idx / block_size;
    const int blk_offset = idx % block_size;
    const int g_idx = (block_ids[blk_idx] * block_size) + blk_offset;
    // print("mapping: %d => %d\n", idx, g_idx);
    // return replace<I>(coord, g_idx);
    return replace<I>(coord, g_idx);
  };

  // (m, n) => (1@0, 1@1)
  Tensor mA = make_gather_tma_tensor(tma, shape(gmem_layout), coord_transform);
  Tensor mB = make_tensor(make_gmem_ptr<T>(g_out), gmem_layout);

  constexpr int R = rank_v<CTA_Tiler>;
  // (CTA_TILE_M,CTA_TILE_N, REST_M,REST_N,...)
  Tensor gA = flat_divide(mA, cta_tiler);
  // (CTA_TILE_M,CTA_TILE_N, REST_M,REST_N,...)
  Tensor gB = flat_divide(mB, cta_tiler);

  //
  // Prepare the TMA_LOAD
  //

  auto cta_tma = tma.get_slice(Int<0>{});  // CTA slice
  // (TMA,TMA_M,TMA_N,REST_M,REST_N)
  Tensor tAgA_x = cta_tma.partition_S(gA);
  // (TMA,TMA_M,TMA_N)
  Tensor tAsA_x = cta_tma.partition_D(sA);

  //
  // Perform the TMA_LOAD
  //

  // INPUT: Group the REST_X modes and the TMA_X modes to easily iterate through
  // the tiles
  // (TMA,REST)
  Tensor tAgA = group_modes<1, rank(tAgA_x)>(tAgA_x);
  Tensor tAsA = group_modes<1, rank(tAsA_x)>(tAsA_x);
  static_assert(size<1>(tAsA) == 1);

  // OUTPUT: Group the CTA_TILE_X modes and REST_X modes for output
  // (CTA_TILE, REST)
  Tensor tBgB = group_modes<0, R>(group_modes<R, rank(gB)>(gB));

  // Loop over the TMA stages, using smem as our buffer
  // (TMA,REST)
  for (int stage = 0; stage < size<1>(tAgA); ++stage) {
    // Set the bytes transferred in this TMA transaction (may involve multiple
    // issues)
    constexpr int kTmaTransactionBytes =
        sizeof(make_tensor_like(tensor<0>(tAsA)));

    if (threadIdx.x == 0) {
      // print("\n ########### %d ########### \n", stage);
      // print("sA: ");
      // print(sA);
      // print("\n");

      /// Initialize shared memory barrier
      tma_load_mbar[0] = 0;
      cute::initialize_barrier(tma_load_mbar[0], 1 /*numThreads*/);
      cute::set_barrier_transaction_bytes(tma_load_mbar[0],
                                          kTmaTransactionBytes);

      copy(tma.with(tma_load_mbar[0]), tAgA(_, stage), tAsA(_, 0));
    }
    __syncthreads();

    /// Wait on the shared memory barrier until the phase bit flips from
    /// kPhaseBit value
    constexpr int kPhaseBit = 0;
    cute::wait_barrier(tma_load_mbar[0], kPhaseBit);

    // Subbyte elements could cause race conditions, so be even more
    // conservative
    if (thread0()) {
      copy(sA, tBgB(_, stage));
    }

    __syncthreads();
  }
}

template <class T,
          class TmaType = T,
          class CopyOp,
          class GMEM_Layout,
          class G_GMEM_Layout,
          class SMEM_Layout,
          class CTA_Tile>
auto test_tma_block_load(CopyOp const& copy_op,
                         GMEM_Layout const& gmem_layout,
                         G_GMEM_Layout const& gather_gmem_layout,
                         SMEM_Layout const& smem_layout,
                         CTA_Tile const& cta_tile,
                         int32_t block_size) {
  assert(block_size % 8 == 0);
  const int m_gather = size<0>(gather_gmem_layout);
  assert(m_gather % block_size == 0);

  const int32_t n_blocks = m_gather / block_size;
  const int32_t n_slots = n_blocks * block_size;
  // generate blocks
  std::vector<int32_t> block_ids;
  std::vector<int32_t> slot_ids;
  block_ids.reserve(n_blocks);
  slot_ids.reserve(n_slots);
  for (int i = 0; i < n_blocks; ++i) {
    const int blk_id = i ^ 1;
    block_ids.push_back(blk_id);
    const int32_t slot_base = blk_id * block_size;
    for (int32_t j = 0; j < block_size; ++j) {
      slot_ids.push_back(slot_base + j);
    }
  }

  // Allocate and initialize host test data
  size_t N = ceil_div(cosize(gmem_layout) * sizeof_bits<T>::value, 8);
  thrust::host_vector<uint8_t> h_in(N);
  for (size_t i = 0; i < h_in.size(); ++i) {
    h_in[i] = uint8_t(i % 13);
  }
  Tensor hA_in =
      make_tensor(make_gmem_ptr<T>(raw_pointer_cast(h_in.data())), gmem_layout);

  // Allocate and initialize device test data
  size_t GN = ceil_div(cosize(gather_gmem_layout) * sizeof_bits<T>::value, 8);
  thrust::device_vector<uint8_t> d_in = h_in;
  thrust::device_vector<uint8_t> d_out(GN, uint8_t(-1));  // overflow uint
  thrust::device_vector<int32_t> d_block_ids = block_ids;

  // Create TMA for this device Tensor
  Tensor gA =
      make_tensor(make_gmem_ptr<T>(raw_pointer_cast(d_in.data())), gmem_layout);
  auto tma =
      make_tma_copy<TmaType>(copy_op, gA, smem_layout, cta_tile, Int<1>{});

  // Launch
  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  tma_test_device_cute<<<1, 128, smem_size>>>(
      reinterpret_cast<T*>(raw_pointer_cast(d_out.data())),
      reinterpret_cast<int32_t const*>(raw_pointer_cast(d_block_ids.data())),
      block_size,
      tma,
      cta_tile,
      gather_gmem_layout,
      smem_layout);

  // Copy results back to host
  thrust::host_vector<uint8_t> h_out = d_out;
  Tensor hA_out = make_tensor(make_gmem_ptr<T>(raw_pointer_cast(h_out.data())),
                              gather_gmem_layout);

  thrust::host_vector<uint8_t> h_out_ref(GN, uint8_t(-1));
  Tensor hA_out_ref = make_tensor(
      make_gmem_ptr<T>(raw_pointer_cast(h_out_ref.data())), gather_gmem_layout);
  for (int i = 0; i < slot_ids.size(); ++i) {
    cute::copy(hA_in(slot_ids[i], _), hA_out_ref(i, _));
  }

  // Validate the results. Print only the first 3 errors.
  int count = 3;
  for (int i = 0; i < int(size(hA_out)) && count > 0; ++i) {
    EXPECT_EQ(hA_out_ref(i), hA_out(i));
    if (hA_out_ref(i) != hA_out(i)) {
      --count;
    }
  }

  return tma;
}

template <class T,
          class TmaType = T,
          class GMEM_Layout,
          class G_GMEM_Layout,
          class SMEM_Layout,
          class CTA_Tile>
auto test_tma_block_load(GMEM_Layout const& gmem_layout,
                         G_GMEM_Layout const& gather_gmem_layout,
                         SMEM_Layout const& smem_layout,
                         CTA_Tile const& cta_tile,
                         int32_t block_size) {
  return test_tma_block_load<T, TmaType>(SM90_TMA_LOAD{},
                                         gmem_layout,
                                         gather_gmem_layout,
                                         smem_layout,
                                         cta_tile,
                                         block_size);
}

template <class T,
          class TmaType = T,
          class GMEM_Layout,
          class G_GMEM_Layout,
          class SMEM_Layout>
auto test_tma_block_load(GMEM_Layout const& gmem_layout,
                         G_GMEM_Layout const& gather_gmem_layout,
                         SMEM_Layout const& smem_layout,
                         int32_t block_size) {
  return test_tma_block_load<T, TmaType>(gmem_layout,
                                         gather_gmem_layout,
                                         smem_layout,
                                         product_each(shape(smem_layout)),
                                         block_size);
}

template <class T, template <typename> typename SWIZZLE_ATOM>
auto test_tma_block_load_swizzle_tile_k(int32_t block_size) {
  auto gmem_layout = make_layout(make_shape(256, 256), GenRowMajor{});

  auto gather_shape = Shape<_64, _256>{};
  auto gather_gmem_layout = make_layout(gather_shape, GenRowMajor{});
  auto smem_layout =
      tile_to_shape(SWIZZLE_ATOM<T>{}, gather_shape, Step<_1, _0>{});

  // TODO: fix the test failures related to tma box size
  // assert (size<1>(SWIZZLE_ATOM<T>{}) != size<1>(smem_layout));
  return test_tma_block_load<T>(
      gmem_layout, gather_gmem_layout, smem_layout, block_size);
}

auto test_tma_block_load(int32_t block_size) {
  test_tma_block_load_swizzle_tile_k<half_t, GMMA::Layout_K_INTER_Atom>(
      block_size);
  test_tma_block_load_swizzle_tile_k<half_t, GMMA::Layout_K_SW32_Atom>(
      block_size);
  test_tma_block_load_swizzle_tile_k<half_t, GMMA::Layout_K_SW64_Atom>(
      block_size);
  test_tma_block_load_swizzle_tile_k<half_t, GMMA::Layout_K_SW128_Atom>(
      block_size);

  test_tma_block_load_swizzle_tile_k<uint8_t, GMMA::Layout_K_INTER_Atom>(
      block_size);
  test_tma_block_load_swizzle_tile_k<uint8_t, GMMA::Layout_K_SW32_Atom>(
      block_size);
  test_tma_block_load_swizzle_tile_k<uint8_t, GMMA::Layout_K_SW64_Atom>(
      block_size);
  test_tma_block_load_swizzle_tile_k<uint8_t, GMMA::Layout_K_SW128_Atom>(
      block_size);
}

TEST(SM120_Tma, Test_Tma_Block_Load) { test_tma_block_load(/*block_size=*/8); }

}  // namespace llm
