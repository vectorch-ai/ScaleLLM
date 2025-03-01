#include <gtest/gtest.h>

#include <cute/tensor.hpp>

#include "cute/layout_composed.hpp"
#include "gather_tensor.hpp"
#include "mha_traits_sm80.h"

namespace llm {

using namespace cute;

template <typename Traits>
void test_mha_traits() {
  // type alias
  using TiledMma = typename Traits::TiledMma;

  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutVt = typename Traits::SmemLayoutVt;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using GmemTiledCopyQ = typename Traits::GmemTiledCopyQ;
  using GmemTiledCopyKV = typename Traits::GmemTiledCopyKV;
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
  // TODO: add tests for layout conformance
}

TEST(MHATraitsTest, TraitsSM80) {
  test_mha_traits<MHATraitsSM80<cute::half_t,
                                /*HEAD_DIM=*/64,
                                /*BLK_M=*/64,
                                /*BLK_N=*/64,
                                /*BLK_K=*/64>>();
}

}  // namespace llm