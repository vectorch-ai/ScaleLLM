#pragma once

#include <cute/util/type_traits.hpp>

namespace llm {

template <class ArchTag,
          class ProblemShape,
          class TileShape,
          class Element,
          class StrideQ,
          class StrideK,
          class StrideV,
          class StrideO,
          bool EVEN_K,
          bool ALIBI,
          bool SOFT_CAP,
          bool LOCAL,
          bool KV_USE_TMA,
          class Enable = void>
struct KernelBuilder {
  static_assert(cute::dependent_false<Element>,
                "Could not build a kernel for given parameters.");
};

}  // namespace llm
