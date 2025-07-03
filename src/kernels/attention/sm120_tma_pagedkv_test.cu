#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/swizzle_layout.hpp"

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
__global__ void tma_test_device_cute(T const* g_in,
                                     T* g_out,
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
  Tensor mA = tma.get_tma_tensor(shape(gmem_layout));
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

#if 1
  if (thread0()) {
    print(tma);
    print("TILE  :  ");
    print(cta_tiler);
    print("\n");
    print("  mA  :  ");
    print(mA);
    print("\n");
    print("  mB  :  ");
    print(mB);
    print("\n");
    print("  gA  :  ");
    print(gA);
    print("\n");
    print("  gB  :  ");
    print(gB);
    print("\n");
    print("  sA  :  ");
    print(sA);
    print("\n");
    print("tAgA_x:  ");
    print(tAgA_x);
    print("\n");
    print("tAsA_x:  ");
    print(tAsA_x);
    print("\n");
  }
#endif

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

#if 1
  if (thread0()) {
    print("tAgA  :  ");
    print(tAgA);
    print("\n");
    print("tAsA  :  ");
    print(tAsA);
    print("\n");
    print("tBgB  :  ");
    print(tBgB);
    print("\n");
  }
#endif

  // Test L2 prefetch
  if (threadIdx.x == 0) {
    // when to use prefetch ???
    prefetch(tma, tAgA);
  }

  // Loop over the TMA stages, using smem as our buffer
  // (TMA,REST)
  for (int stage = 0; stage < size<1>(tAgA); ++stage) {
    // Set the bytes transferred in this TMA transaction (may involve multiple
    // issues)
    constexpr int kTmaTransactionBytes =
        sizeof(make_tensor_like(tensor<0>(tAsA)));

    if (threadIdx.x == 0) {
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
    // why no phasebit flip???

    //
    // Write out trivially smem -> gmem
    //

    // Subbyte elements could cause race conditions, so be even more
    // conservative
    if (thread0()) {
      print("\n ########### %d ########### \n", stage);
      print("sA: \n");
      print_tensor(sA);
      print("\nsA_no_swizzle: \n");
      print_tensor(sA_no_siwzzle);
      print("\n");

      copy(sA, tBgB(_, stage));
    }

    __syncthreads();
  }
}

template <class T,
          class TmaType = T,
          class CopyOp,
          class GMEM_Layout,
          class SMEM_Layout,
          class CTA_Tile>
auto test_tma_load(CopyOp const& copy_op,
                   GMEM_Layout const& gmem_layout,
                   SMEM_Layout const& smem_layout,
                   CTA_Tile const& cta_tile) {
  using namespace cute;
  print("test_tma_load ------------------------------ \n");
  print("gmem_layout: ");
  print(gmem_layout);
  print("\n");
  print("smem_layout: ");
  print(smem_layout);
  print("\n");
  print("cta_tile: ");
  print(cta_tile);
  print("\n");

  // Allocate and initialize host test data
  size_t N =
      ceil_div(cosize(gmem_layout) * sizeof_bits<T>::value, 8) / sizeof(T);
  thrust::host_vector<T> h_in(N);
  for (size_t i = 0; i < h_in.size(); ++i) {
    h_in[i] = T(i);
  }
  Tensor hA_in = make_tensor(recast_ptr<T>(h_in.data()), gmem_layout);

  print("\n ###################### \n");
  print("hA_in: \n");
  print_tensor(hA_in);
  print("\n");

  // Allocate and initialize device test data
  thrust::device_vector<T> d_in = h_in;
  thrust::device_vector<T> d_out(h_in.size(),
                                 T(-1));  // overflow uint

  // Create TMA for this device Tensor
  Tensor gA =
      make_tensor(make_gmem_ptr<T>(raw_pointer_cast(d_in.data())), gmem_layout);
  auto tma =
      make_tma_copy<TmaType>(copy_op, gA, smem_layout, cta_tile, Int<1>{});
  // print(tma);

  // Launch
  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  tma_test_device_cute<<<1, 128, smem_size>>>(
      reinterpret_cast<T const*>(raw_pointer_cast(d_in.data())),
      reinterpret_cast<T*>(raw_pointer_cast(d_out.data())),
      tma,
      cta_tile,
      gmem_layout,
      smem_layout);

  // Copy results back to host
  thrust::host_vector<T> h_out = d_out;
  Tensor hA_out = make_tensor(recast_ptr<T>(h_out.data()), gmem_layout);

  // Validate the results. Print only the first 3 errors.
  int count = 3;
  for (int i = 0; i < int(size(hA_out)) && count > 0; ++i) {
    EXPECT_EQ(hA_in(i), hA_out(i));
    if (hA_in(i) != hA_out(i)) {
      --count;
    }
  }

  return tma;
}

template <class T,
          class TmaType = T,
          class GMEM_Layout,
          class SMEM_Layout,
          class CTA_Tile>
auto test_tma_load(GMEM_Layout const& gmem_layout,
                   SMEM_Layout const& smem_layout,
                   CTA_Tile const& cta_tile) {
  return test_tma_load<T, TmaType>(
      SM90_TMA_LOAD{}, gmem_layout, smem_layout, cta_tile);
}

template <class T, class TmaType = T, class GMEM_Layout, class SMEM_Layout>
auto test_tma_load(GMEM_Layout const& gmem_layout,
                   SMEM_Layout const& smem_layout) {
  return test_tma_load<T, TmaType>(
      gmem_layout, smem_layout, product_each(shape(smem_layout)));
}

template <class T, template <typename> typename SWIZZLE_ATOM>
void test_tma_load_swizzle_atom_mn() {
  auto smem_layout = SWIZZLE_ATOM<T>{};
  {  // Static gmem
     // Layout gmem_layout = make_layout(shape(smem_layout), GenColMajor{});
     // test_tma_load<T>(gmem_layout, smem_layout);
  }
  {  // Dynamic gmem
    Layout gmem_layout =
        make_layout(make_shape(2 * uint32_t(size<0>(smem_layout)),
                               2 * uint32_t(size<1>(smem_layout))),
                    GenColMajor{});
    test_tma_load<T>(gmem_layout, smem_layout);
  }
}

template <class T, template <typename> typename SWIZZLE_ATOM>
void test_tma_load_swizzle_atom_k() {
  auto smem_layout = SWIZZLE_ATOM<T>{};
  {  // Static gmem
     // Layout gmem_layout = make_layout(shape(smem_layout), GenRowMajor{});
     // test_tma_load<T>(gmem_layout, smem_layout);
  }
  {  // Dynamic gmem
    Layout gmem_layout =
        make_layout(make_shape(2 * uint32_t(size<0>(smem_layout)),
                               2 * uint32_t(size<1>(smem_layout))),
                    GenRowMajor{});
    test_tma_load<T>(gmem_layout, smem_layout);
  }
}

template <class T, template <typename> typename SWIZZLE_ATOM>
auto test_tma_load_swizzle_tile_mn() {
  auto smem_layout = tile_to_shape(SWIZZLE_ATOM<T>{}, Shape<_128, _128>{});
  Layout gmem_layout = make_layout(
      make_shape(int(size<0>(smem_layout)), int(size<1>(smem_layout))),
      GenColMajor{});
  return test_tma_load<T>(gmem_layout, smem_layout);
}

template <class T, template <typename> typename SWIZZLE_ATOM>
auto test_tma_load_swizzle_tile_k() {
  auto smem_layout = tile_to_shape(SWIZZLE_ATOM<T>{}, Shape<_128, _128>{});
  Layout gmem_layout = make_layout(
      make_shape(int(size<0>(smem_layout)), int(size<1>(smem_layout))),
      GenRowMajor{});
  return test_tma_load<T>(gmem_layout, smem_layout);
}

TEST(SM120_Tma, Load_Tiny) {
  // two requirements:
  // 1: a contiguous direction (stride 1)
  // 2: other strides as multiples of 16 bytes

  // 1D
  // {
  //   Layout smem_layout = Layout<_16, _1>{};
  //   Layout gmem_layout = smem_layout;
  //   test_tma_load<int8_t>(gmem_layout, smem_layout);
  //   // test_tma_load<half_t>(gmem_layout, smem_layout);
  //   // test_tma_load<float>(gmem_layout, smem_layout);
  //   // test_tma_load<double>(gmem_layout, smem_layout);
  // }

  // 2D row-major
  // {
  //   Layout gmem_layout = Layout<Shape<_4, _32>, Stride<_32, _1>>{};
  //   Layout smem_layout = Layout<Shape<_2, _16>, Stride<_16, _1>>{};
  //   test_tma_load<int8_t>(gmem_layout, smem_layout);
  // }

  // 2D col-major
  // {
  //   Layout smem_layout = Layout<Shape<_16, _2>, Stride<_1, _16>>{};
  //   Layout gmem_layout = smem_layout;
  //   test_tma_load<int8_t>(gmem_layout, smem_layout);
  // }
}

TEST(SM120_Tma, Load_Swizzle_Atoms) {
  // test_tma_load_swizzle_atom_k<uint16_t, GMMA::Layout_K_INTER_Atom>();
  // test_tma_load_swizzle_atom_k<uint16_t, GMMA::Layout_K_SW32_Atom>();
  // test_tma_load_swizzle_atom_k<uint16_t, GMMA::Layout_K_SW64_Atom>();
  // test_tma_load_swizzle_atom_k<uint16_t, GMMA::Layout_K_SW128_Atom>();
}

TEST(SM120_Tma, Tma_Load_Swizzle_Tiles) {
  // Other T-types use too much smem
  test_tma_load_swizzle_tile_k<uint16_t, GMMA::Layout_K_INTER_Atom>();
  test_tma_load_swizzle_tile_k<uint16_t, GMMA::Layout_K_SW32_Atom>();
  test_tma_load_swizzle_tile_k<uint16_t, GMMA::Layout_K_SW64_Atom>();
  test_tma_load_swizzle_tile_k<uint16_t, GMMA::Layout_K_SW128_Atom>();
}

}  // namespace llm
