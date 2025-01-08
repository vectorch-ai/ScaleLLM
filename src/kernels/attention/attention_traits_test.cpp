#include <gtest/gtest.h>

#include <cute/tensor.hpp>

#include "attention_traits_sm80.h"

namespace llm {

using namespace cute;

template <typename Traits>
void test_attention_traits() {
  // type alias
  using Element = typename Traits::Element;
  using BLK_M = typename Traits::BLK_M;
  using BLK_N = typename Traits::BLK_N;
  using BLK_K = typename Traits::BLK_K;
  using HEAD_DIM = typename Traits::HEAD_DIM;

  using TiledMma = typename Traits::TiledMma;
  using Layout = typename Traits::LayoutConvertor;

  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using GmemTiledCopyQKV = typename Traits::GmemTiledCopyQKV;
  using GmemTiledCopyO = typename Traits::GmemTiledCopyO;

  using SmemTiledCopyQ = typename Traits::SmemTiledCopyQ;
  using SmemTiledCopyK = typename Traits::SmemTiledCopyK;
  using SmemTiledCopyVt = typename Traits::SmemTiledCopyVt;
  using SmemTiledCopyO = typename Traits::SmemTiledCopyO;

  // test layout conversation
  Tensor sQ = make_tensor(counting_iterator<int>(0), SmemLayoutQ{});
  Tensor sK = make_tensor(counting_iterator<int>(0), SmemLayoutK{});
  Tensor sV = make_tensor(counting_iterator<int>(0), SmemLayoutV{});
  Tensor sVt = make_tensor(sV.data(), SmemLayoutVt{});

  // print("sQ:"); print(sQ);print("\n");
  // print("sK:"); print(sK);print("\n");
  // print("sV:"); print(sV);print("\n");
  // print("sVt:"); print(sVt);print("\n");

  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(0);
  auto tOrVt = thr_mma.partition_fragment_B(sVt);
  // print("tOrVt:"); print(tOrVt);print("\n");

  // Tensor sVtNoSwizzle = make_tensor(sVt.data(),
  //                                   get_nonswizzle_portion(SmemLayoutVt{}));
  // auto tOrVtNoSwizzle = thr_mma.partition_fragment_B(sVtNoSwizzle);
  // print("tOrVtNoSwizzle:"); print(tOrVtNoSwizzle);print("\n");


  auto tOrK = thr_mma.partition_fragment_B(sK);
  // print("tOrK:"); print(tOrK);print("\n");

  // auto sKNoSwizzle = make_tensor(sK.data(),
  //                                get_nonswizzle_portion(SmemLayoutK{}));
  // auto tOrKNoSwizzle = thr_mma.partition_fragment_B(sKNoSwizzle);
  // print("tOrKNoSwizzle:"); print(tOrKNoSwizzle);print("\n");



  // test smem layout

  // test tiled copy partition
  EXPECT_TRUE(false);
}

TEST(AttentionTraitsTest, TraitsSM80) {
  test_attention_traits<AttentionTraitsSM80<cute::half_t,
                                            /*kHeadDim_=*/64,
                                            /*kBlockM_=*/64,
                                            /*kBlockN_=*/64>>();
}

}  // namespace llm