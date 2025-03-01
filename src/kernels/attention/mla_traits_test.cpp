#include <gtest/gtest.h>

#include <cute/tensor.hpp>

#include "cute_extensions.cuh"
#include "gather_tensor.hpp"
#include "mla_traits_sm80.h"

namespace llm {

using namespace cute;

template <typename Traits>
void test_mla_traits() {
  // type alias
  using TiledMma_QK = typename Traits::TiledMma_QK;

  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutKV = typename Traits::SmemLayoutKV;
  using SmemLayoutQRope = typename Traits::SmemLayoutQRope;
  using SmemLayoutKRope = typename Traits::SmemLayoutKRope;
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
  Tensor sKV = make_tensor(counting_iterator<int>(0), SmemLayoutKV{});
  Tensor sVt = make_tensor(sKV.data(), SmemLayoutVt{});

  Tensor sQ_rope = make_tensor(counting_iterator<int>(0), SmemLayoutQRope{});
  Tensor sKV_rope = make_tensor(counting_iterator<int>(0), SmemLayoutKRope{});

  // print("sQ:"); print(sQ);print("\n");
  // print("sKV:"); print(sKV);print("\n");
  // print("sVt:"); print(sVt);print("\n");

  // print("sQ_rope:"); print(sQ_rope);print("\n");
  // print("sKV_rope:"); print(sKV_rope);print("\n");

  TiledMma_QK tiled_mma_qk;
  auto thr_mma_qk = tiled_mma_qk.get_slice(0);
  // auto tOrVt = thr_mma_qk.partition_fragment_B(sVt);
  // TODO: add tests for layout conformance
}

TEST(MLATraitsTest, TraitsSM80) {
  test_mla_traits<MLATraitsSM80<cute::half_t,
                                /*HEAD_DIM=*/256,
                                /*ROPE_HEAD_DIM=*/64,
                                /*BLK_M=*/64,
                                /*BLK_N=*/64,
                                /*BLK_K=*/64,
                                /*STAGES=*/1>>();
}

}  // namespace llm