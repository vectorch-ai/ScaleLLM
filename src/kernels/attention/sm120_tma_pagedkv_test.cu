#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

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
  Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem.begin()),
                          smem_layout);  // (CTA_TILE_M,CTA_TILE_N,...)
  // Shared memory barriers use 64bits in SMEM for synchronization
  uint64_t* tma_load_mbar = shared_storage.tma_load_mbar;

  // TMA requires special handling of strides to deal with coord codomain
  // mapping Represent the full tensors -- get these from TMA
  Tensor mA = tma.get_tma_tensor(shape(gmem_layout));
  Tensor mB = make_tensor(make_gmem_ptr<T>(g_out), gmem_layout);

  constexpr int R = rank_v<CTA_Tiler>;
  Tensor gA = flat_divide(
      mA, cta_tiler);  // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)
  Tensor gB = flat_divide(
      mB, cta_tiler);  // (CTA_TILE_M,CTA_TILE_N,...REST_M,REST_N,...)

  //
  // Prepare the TMA_LOAD
  //

  auto cta_tma = tma.get_slice(Int<0>{});   // CTA slice
  Tensor tAgA_x = cta_tma.partition_S(gA);  // (TMA,TMA_M,TMA_N,REST_M,REST_N)
  Tensor tAsA_x = cta_tma.partition_D(sA);  // (TMA,TMA_M,TMA_N)

#if 0
  if (thread0()) {
    print(tma);
    print("TILE  :  "); print(cta_tiler); print("\n");
    print("  mA  :  "); print(  mA);   print("\n");
    print("  mB  :  "); print(  mB);   print("\n");
    print("  gA  :  "); print(  gA);   print("\n");
    print("  gB  :  "); print(  gB);   print("\n");
    print("  sA  :  "); print(  sA);   print("\n");
    print("tAgA_x:  "); print(tAgA_x); print("\n");
    print("tAsA_x:  "); print(tAsA_x); print("\n");
  }
#endif

  //
  // Perform the TMA_LOAD
  //

  // INPUT: Group the REST_X modes and the TMA_X modes to easily iterate through
  // the tiles
  Tensor tAgA = group_modes<1, rank(tAgA_x)>(tAgA_x);  // (TMA,REST)
  Tensor tAsA = group_modes<1, rank(tAsA_x)>(tAsA_x);  // (TMA,REST)
  static_assert(size<1>(tAsA) == 1);

  // OUTPUT: Group the CTA_TILE_X modes and REST_X modes for output
  Tensor tBgB =
      group_modes<0, R>(group_modes<R, rank(gB)>(gB));  // (CTA_TILE, REST)

#if 0
  if (thread0()) {
    print("tAgA  :  "); print(tAgA); print("\n");
    print("tAsA  :  "); print(tAsA); print("\n");
    print("tBgB  :  "); print(tBgB); print("\n");
  }
#endif

  // Test L2 prefetch
  if (threadIdx.x == 0) {
    prefetch(tma, tAgA);
  }

  // Loop over the TMA stages, using smem as our buffer
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

    //
    // Write out trivially smem -> gmem
    //

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
          class SMEM_Layout,
          class CTA_Tile>
auto test_tma_load(CopyOp const& copy_op,
                   GMEM_Layout const& gmem_layout,
                   SMEM_Layout const& smem_layout,
                   CTA_Tile const& cta_tile) {
  using namespace cute;

  // Allocate and initialize host test data
  size_t N = ceil_div(cosize(gmem_layout) * sizeof_bits<T>::value, 8);
  thrust::host_vector<uint8_t> h_in(N);
  for (size_t i = 0; i < h_in.size(); ++i) {
    h_in[i] = uint8_t(i % 13);
  }
  Tensor hA_in = make_tensor(recast_ptr<T>(h_in.data()), gmem_layout);

  // Allocate and initialize device test data
  thrust::device_vector<uint8_t> d_in = h_in;
  thrust::device_vector<uint8_t> d_out(h_in.size(),
                                       uint8_t(-1));  // overflow uint

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
  thrust::host_vector<uint8_t> h_out = d_out;
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

TEST(SM120_Tma, Load_1D) {
  {
    Layout smem_layout = Layout<_256, _1>{};
    {
      Layout gmem_layout = smem_layout;
      test_tma_load<int8_t>(gmem_layout, smem_layout);
      test_tma_load<half_t>(gmem_layout, smem_layout);
      test_tma_load<float>(gmem_layout, smem_layout);
      test_tma_load<double>(gmem_layout, smem_layout);
    }

    {
      Layout gmem_layout = make_layout(128, GenColMajor{});
      test_tma_load<int8_t>(gmem_layout, smem_layout);
      test_tma_load<half_t>(gmem_layout, smem_layout);
      test_tma_load<float>(gmem_layout, smem_layout);
      test_tma_load<double>(gmem_layout, smem_layout);
    }

    {
      Layout gmem_layout = make_layout(384, GenColMajor{});
      test_tma_load<int8_t>(gmem_layout, smem_layout);
      test_tma_load<half_t>(gmem_layout, smem_layout);
      test_tma_load<float>(gmem_layout, smem_layout);
      test_tma_load<double>(gmem_layout, smem_layout);
    }
  }

  {
    Layout smem_layout = Layout<Shape<_8, _8>, Stride<_1, _8>>{};
    {
      Layout gmem_layout = smem_layout;
      test_tma_load<int8_t>(gmem_layout, smem_layout);
      test_tma_load<half_t>(gmem_layout, smem_layout);
      test_tma_load<float>(gmem_layout, smem_layout);
      test_tma_load<double>(gmem_layout, smem_layout);
    }

    // This doesn't result in a 1D TMA, even though it could/should...
    {
      Layout gmem_layout = tile_to_shape(smem_layout, Shape<_16, _16>{});
      test_tma_load<int8_t>(gmem_layout, smem_layout);
      test_tma_load<half_t>(gmem_layout, smem_layout);
      test_tma_load<float>(gmem_layout, smem_layout);
      test_tma_load<double>(gmem_layout, smem_layout);
    }
  }
}

TEST(SM120_Tma, Tma_Load_32x32_Col) {
  Layout smem_layout = Layout<Shape<_32, _32>, Stride<_1, _32>>{};
  {
    Layout gmem_layout = smem_layout;
    test_tma_load<int8_t>(gmem_layout, smem_layout);
    test_tma_load<half_t>(gmem_layout, smem_layout);
    test_tma_load<float>(gmem_layout, smem_layout);
    test_tma_load<double>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = make_layout(make_shape(32, 32), GenColMajor{});
    test_tma_load<int8_t>(gmem_layout, smem_layout);
    test_tma_load<half_t>(gmem_layout, smem_layout);
    test_tma_load<float>(gmem_layout, smem_layout);
    test_tma_load<double>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout =
        make_layout(make_shape(32, 32), make_stride(Int<1>{}, 1024));
    test_tma_load<int8_t>(gmem_layout, smem_layout);
    test_tma_load<half_t>(gmem_layout, smem_layout);
    test_tma_load<float>(gmem_layout, smem_layout);
    test_tma_load<double>(gmem_layout, smem_layout);
  }
}

TEST(SM120_Tma, Tma_Load_32x32_Row) {
  Layout smem_layout = Layout<Shape<_32, _32>, Stride<_32, _1>>{};
  {
    Layout gmem_layout = smem_layout;
    test_tma_load<int8_t>(gmem_layout, smem_layout);
    test_tma_load<half_t>(gmem_layout, smem_layout);
    test_tma_load<float>(gmem_layout, smem_layout);
    test_tma_load<double>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout = make_layout(make_shape(32, 32), GenRowMajor{});
    test_tma_load<int8_t>(gmem_layout, smem_layout);
    test_tma_load<half_t>(gmem_layout, smem_layout);
    test_tma_load<float>(gmem_layout, smem_layout);
    test_tma_load<double>(gmem_layout, smem_layout);
  }

  {
    Layout gmem_layout =
        make_layout(make_shape(32, 32), make_stride(1024, Int<1>{}));
    test_tma_load<int8_t>(gmem_layout, smem_layout);
    test_tma_load<half_t>(gmem_layout, smem_layout);
    test_tma_load<float>(gmem_layout, smem_layout);
    test_tma_load<double>(gmem_layout, smem_layout);
  }
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

TEST(SM120_Tma, Tma_Load_Swizzle_Atoms) {
  test_tma_load_swizzle_atom_mn<int8_t, GMMA::Layout_MN_SW128_Atom>();
  test_tma_load_swizzle_atom_mn<half_t, GMMA::Layout_MN_SW128_Atom>();
  test_tma_load_swizzle_atom_mn<float, GMMA::Layout_MN_SW128_Atom>();
  test_tma_load_swizzle_atom_mn<double, GMMA::Layout_MN_SW128_Atom>();

  test_tma_load_swizzle_atom_mn<int8_t, GMMA::Layout_MN_SW64_Atom>();
  test_tma_load_swizzle_atom_mn<half_t, GMMA::Layout_MN_SW64_Atom>();
  test_tma_load_swizzle_atom_mn<float, GMMA::Layout_MN_SW64_Atom>();
  test_tma_load_swizzle_atom_mn<double, GMMA::Layout_MN_SW64_Atom>();

  test_tma_load_swizzle_atom_mn<int8_t, GMMA::Layout_MN_SW32_Atom>();
  test_tma_load_swizzle_atom_mn<half_t, GMMA::Layout_MN_SW32_Atom>();
  test_tma_load_swizzle_atom_mn<float, GMMA::Layout_MN_SW32_Atom>();
  test_tma_load_swizzle_atom_mn<double, GMMA::Layout_MN_SW32_Atom>();

  test_tma_load_swizzle_atom_mn<int8_t, GMMA::Layout_MN_INTER_Atom>();
  test_tma_load_swizzle_atom_mn<half_t, GMMA::Layout_MN_INTER_Atom>();
  test_tma_load_swizzle_atom_mn<float, GMMA::Layout_MN_INTER_Atom>();
  test_tma_load_swizzle_atom_mn<double, GMMA::Layout_MN_INTER_Atom>();

  test_tma_load_swizzle_atom_k<int8_t, GMMA::Layout_K_SW128_Atom>();
  test_tma_load_swizzle_atom_k<half_t, GMMA::Layout_K_SW128_Atom>();
  test_tma_load_swizzle_atom_k<float, GMMA::Layout_K_SW128_Atom>();
  test_tma_load_swizzle_atom_k<double, GMMA::Layout_K_SW128_Atom>();

  test_tma_load_swizzle_atom_k<int8_t, GMMA::Layout_K_SW64_Atom>();
  test_tma_load_swizzle_atom_k<half_t, GMMA::Layout_K_SW64_Atom>();
  test_tma_load_swizzle_atom_k<float, GMMA::Layout_K_SW64_Atom>();
  test_tma_load_swizzle_atom_k<double, GMMA::Layout_K_SW64_Atom>();

  test_tma_load_swizzle_atom_k<int8_t, GMMA::Layout_K_SW32_Atom>();
  test_tma_load_swizzle_atom_k<half_t, GMMA::Layout_K_SW32_Atom>();
  test_tma_load_swizzle_atom_k<float, GMMA::Layout_K_SW32_Atom>();
  test_tma_load_swizzle_atom_k<double, GMMA::Layout_K_SW32_Atom>();

  test_tma_load_swizzle_atom_k<int8_t, GMMA::Layout_K_INTER_Atom>();
  test_tma_load_swizzle_atom_k<half_t, GMMA::Layout_K_INTER_Atom>();
  test_tma_load_swizzle_atom_k<float, GMMA::Layout_K_INTER_Atom>();
  test_tma_load_swizzle_atom_k<double, GMMA::Layout_K_INTER_Atom>();
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

TEST(SM120_Tma, Tma_Load_Swizzle_Tiles) {
  // Other T-types use too much smem
  test_tma_load_swizzle_tile_mn<int8_t, GMMA::Layout_MN_SW128_Atom>();
  test_tma_load_swizzle_tile_mn<half_t, GMMA::Layout_MN_SW128_Atom>();
  test_tma_load_swizzle_tile_mn<int8_t, GMMA::Layout_MN_SW64_Atom>();
  test_tma_load_swizzle_tile_mn<half_t, GMMA::Layout_MN_SW64_Atom>();
  test_tma_load_swizzle_tile_mn<int8_t, GMMA::Layout_MN_SW32_Atom>();
  test_tma_load_swizzle_tile_mn<half_t, GMMA::Layout_MN_SW32_Atom>();
  test_tma_load_swizzle_tile_mn<int8_t, GMMA::Layout_MN_INTER_Atom>();
  test_tma_load_swizzle_tile_mn<half_t, GMMA::Layout_MN_INTER_Atom>();
  test_tma_load_swizzle_tile_k<int8_t, GMMA::Layout_K_SW128_Atom>();
  test_tma_load_swizzle_tile_k<half_t, GMMA::Layout_K_SW128_Atom>();
  test_tma_load_swizzle_tile_k<int8_t, GMMA::Layout_K_SW64_Atom>();
  test_tma_load_swizzle_tile_k<half_t, GMMA::Layout_K_SW64_Atom>();
  test_tma_load_swizzle_tile_k<int8_t, GMMA::Layout_K_SW32_Atom>();
  test_tma_load_swizzle_tile_k<half_t, GMMA::Layout_K_SW32_Atom>();
  test_tma_load_swizzle_tile_k<int8_t, GMMA::Layout_K_INTER_Atom>();
  test_tma_load_swizzle_tile_k<half_t, GMMA::Layout_K_INTER_Atom>();
}

// Tensor by-mode
TEST(SM120_Tma, Tma_Load_Tensor) {
  // 3-mode TMA
  {
    Layout gmem_layout = make_layout(make_shape(128, 64, 5));
    auto cta_tile = Shape<_64, _32>{};  // GMEM Tiling:
                                        //   Take 64-elem from m
                                        //   Take 32-elem from k
    auto smem_layout = make_layout(Shape<_64, _32>{});
    test_tma_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  // 4-mode TMA
  {
    Layout gmem_layout =
        make_layout(make_shape(make_shape(80, 40), make_shape(32, 12)));
    auto cta_tile =
        Shape<Shape<_16, _8>,
              Shape<_32, _2>>{};  // GMEM Tiling:
                                  //   Take 16-elem from m0, 8-elem from m1,
                                  //   Take 32-elem from k0, 2-elem from k1
    auto smem_layout = make_layout(Shape<_128, _64>{});
    test_tma_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  // 5-mode TMA
  {
    Layout gmem_layout =
        make_layout(make_shape(make_shape(32, 32, 32), make_shape(32, 12)));
    auto cta_tile =
        Shape<Shape<_16, _4, _2>,
              Shape<_16, _2>>{};  // GMEM Tiling:
                                  //   Take 4-elem from m0, 4-elem from m1,
                                  //   5-elem from m2 Take 32-elem from k0,
                                  //   2-elem from k1
    auto smem_layout = make_layout(Shape<_128, _32>{});
    test_tma_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }
}

// Tensor Multimode -- TMA with more than 5 modes in GMEM (packs residual modes
// into last TMA mode)
TEST(SM120_Tma, Tma_Load_Tensor_Multimode) {
  {
    Layout gmem_layout =
        make_layout(make_shape(make_shape(32, 3, 2, 2), make_shape(32, 4, 2)));
    auto cta_tile =
        Shape<Shape<_32>,
              Shape<_32, _2>>{};  // GMEM Tiling:
                                  //  Take 32-elem from m0
                                  //  Take 32-elem from k0, 2-elem from k1
    auto smem_layout = make_layout(Shape<_32, _64>{});
    test_tma_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  {
    Layout gmem_layout =
        make_layout(make_shape(make_shape(64, 3, 2, 2), make_shape(32, 4, 2)));
    auto cta_tile =
        Shape<Shape<_32, _3>,
              Shape<_32, _2>>{};  // GMEM Tiling:
                                  //  Take 32-elem from m0, 3-elem from m1
                                  //  Take 32-elem from k0, 2-elem from k1
    auto smem_layout = make_layout(Shape<_96, _64>{});
    test_tma_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }

  {
    Layout gmem_layout = make_layout(
        make_shape(make_shape(64, 3, 2, 3, 2), make_shape(32, 4, 2, 2)));
    auto cta_tile =
        Shape<Shape<_32>,
              Shape<_16, _2>>{};  // GMEM Tiling:
                                  //  Take 32-elem from m0
                                  //  Take 16-elem from k0, 2-elem from k1
    auto smem_layout = make_layout(Shape<_32, _32>{});
    test_tma_load<half_t>(gmem_layout, smem_layout, cta_tile);
  }
}

TEST(SM120_Tma, Tma_Load_Coalesce) {
  // Interleaved ColMajor
  {
    Layout gmem_layout = make_layout(make_shape(128, make_shape(_4{}, 128)),
                                     make_stride(_4{}, make_stride(_1{}, 512)));
    auto smem_layout =
        make_layout(make_shape(_32{}, make_shape(_4{}, _32{})),
                    make_stride(_4{}, make_stride(_1{}, _128{})));

    // By default, uses cta_tile = Shape<_32,_128>
    auto tma = test_tma_load<int8_t>(gmem_layout, smem_layout);
    // Check the TMA rank
    EXPECT_EQ(rank(tma.get_tma_tensor(shape(gmem_layout))(0)), 2);
  }

  // Interleaved RowMajor
  {
    Layout gmem_layout = make_layout(make_shape(make_shape(_4{}, 128), 128),
                                     make_stride(make_stride(_1{}, 512), _4{}));
    auto smem_layout =
        make_layout(make_shape(make_shape(_4{}, _32{}), _32{}),
                    make_stride(make_stride(_1{}, _128{}), _4{}));

    // By default, uses cta_tile = Shape<_128,_32>
    auto tma = test_tma_load<int8_t>(gmem_layout, smem_layout);
    // Check the TMA rank
    EXPECT_EQ(rank(tma.get_tma_tensor(shape(gmem_layout))(0)), 2);
  }

  // Account for stride-0 modes within the TMA tile
  {
    Layout gmem_layout = make_layout(make_shape(128, make_shape(_32{}, 4)),
                                     make_stride(_1{}, make_stride(_0{}, 128)));
    auto smem_layout = make_layout(make_shape(_64{}, make_shape(_32{})),
                                   make_stride(_1{}, make_stride(_0{})));

    // By default, uses cta_tile = Shape<_64,_32>
    auto tma = test_tma_load<uint16_t>(gmem_layout, smem_layout);
    // Check the TMA rank
    EXPECT_EQ(rank(tma.get_tma_tensor(shape(gmem_layout))(0)), 2);
  }

  // Coalesce many modes and account for stride-0 modes within the TMA tile
  {
    Layout gmem_layout = make_layout(
        make_shape(make_shape(_32{}, _4{}, 4), _32{}, make_shape(_4{}, 4)),
        make_stride(
            make_stride(_16{}, _4{}, 2048), _0{}, make_stride(_1{}, _512{})));
    auto smem_layout = make_layout(
        make_shape(make_shape(_32{}, _4{}), _32{}, make_shape(_4{})),
        make_stride(make_stride(_16{}, _4{}), _0{}, make_stride(_1{})));

    // By default, uses cta_tile = Shape<_128,_32,_4>
    auto tma = test_tma_load<int8_t>(gmem_layout, smem_layout);
    // Check the TMA rank (Could be 3 instead of 4 with even better
    // coalescing...?)
    EXPECT_EQ(rank(tma.get_tma_tensor(shape(gmem_layout))(0)), 4);
  }
}

TEST(SM120_Tma, Tma_Load_InternalType) {
  Layout smem_layout = Layout<Shape<_32, _32>, Stride<_1, _32>>{};
  Layout gmem_layout = make_layout(make_shape(64, 64));

  // Downcasted tensors to smaller TmaTypes
  {
    test_tma_load<int8_t, uint8_t>(gmem_layout, smem_layout);
    test_tma_load<half_t, uint8_t>(gmem_layout, smem_layout);
    test_tma_load<float, uint8_t>(gmem_layout, smem_layout);
    test_tma_load<double, uint8_t>(gmem_layout, smem_layout);
  }

  // Upcasted tensors to larger TmaTypes
  {
    test_tma_load<int8_t, uint64_t>(gmem_layout, smem_layout);
    test_tma_load<half_t, uint64_t>(gmem_layout, smem_layout);
    test_tma_load<float, uint64_t>(gmem_layout, smem_layout);
    test_tma_load<double, uint64_t>(gmem_layout, smem_layout);
  }

  // Complex<double> is 128bit, which the TMA has no concept of
  {
    test_tma_load<complex<double>, uint64_t>(gmem_layout, smem_layout);
    test_tma_load<complex<double>, uint32_t>(gmem_layout, smem_layout);
  }
}

}  // namespace llm
