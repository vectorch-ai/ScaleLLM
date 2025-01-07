
#pragma once
#include <absl/strings/str_format.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/config.hpp>
#include <ostream>

#include "svg_builder.hpp"

namespace llm {
using namespace cute;

namespace detail {

static const char* kColorMap[8] = {"rgb(175,175,255)",
                                   "rgb(175,255,175)",
                                   "rgb(255,255,175)",
                                   "rgb(255,175,175)",
                                   "rgb(210,210,255)",
                                   "rgb(210,255,210)",
                                   "rgb(255,255,210)",
                                   "rgb(255,210,210)"};

// MNK MMA Layout to SVG -- 8-value color coded by thread
template <class... Args>
CUTE_HOST void print_svg(std::ostream& os, const TiledMMA<Args...>& mma) {
  auto [C, TC] = mma.get_layoutC_MN();  // (m,n)->(tid,vid), tid->thr_idx
  auto [A, TA] = mma.get_layoutA_MK();  // (m,k)->(tid,vid), tid->thr_idx
  auto [B, TB] = mma.get_layoutB_NK();  // (n,k)->(tid,vid), tid->thr_idx

  const int cell_width = 20;
  const int cell_height = 20;
  const int num_rows = size<0>(A) + size<1>(B) + 2;  // M + K + 2
  const int num_cols = size<0>(B) + size<1>(A) + 2;  // N + K + 2

  SVGBuilder builder(num_rows, num_cols, cell_width, cell_height);

  // header
  builder.print_header(os);

  // skipping (K + 2) rows/columns
  const auto base = (size<1>(B) + 2);

  // C starting at (K + 2, K + 2)
  builder.print_comment(os, "C starting at (K + 2, K + 2)");
  for (int m = 0; m < cute::size<0>(C); ++m) {
    for (int n = 0; n < cute::size<1>(C); ++n) {
      int thrid = C(m, n) % size(TC);
      int val_idx = C(m, n) / size(TC);
      int thr_idx = TC(thrid);

      builder.print_cell(
          os, m + base, n + base, thr_idx, val_idx, kColorMap[thr_idx % 8]);
    }
  }

  // A starting at (K+1, 0)
  builder.print_comment(os, "A starting at (K+1, 0)");
  for (int m = 0; m < size<0>(A); ++m) {
    builder.print_label(os, m + base, 0, m);
  }
  for (int k = 0; k < size<1>(A); ++k) {
    builder.print_label(os, base - 1, k + 1, k);
  }
  for (int m = 0; m < size<0>(A); ++m) {
    for (int k = 0; k < size<1>(A); ++k) {
      int thrid = A(m, k) % size(TA);
      int val_idx = A(m, k) / size(TA);
      int thr_idx = TA(thrid);
      builder.print_cell(
          os, m + base, k + 1, thr_idx, val_idx, kColorMap[thr_idx % 8]);
    }
  }

  // B starting at (0, K+1)
  builder.print_comment(os, "B starting at (0, K+1)");
  for (int n = 0; n < size<0>(B); ++n) {
    builder.print_label(os, 0, n + base, n);
  }
  for (int k = 0; k < size<1>(B); ++k) {
    builder.print_label(os, k + 1, base - 1, k);
  }
  for (int n = 0; n < size<0>(B); ++n) {
    for (int k = 0; k < size<1>(B); ++k) {
      int thrid = B(n, k) % size(TB);
      int val_idx = B(n, k) / size(TB);
      int thr_idx = TB(thrid);

      builder.print_cell(
          os, k + 1, n + base, thr_idx, val_idx, kColorMap[thr_idx % 8]);
    }
  }

  // footer
  builder.print_footer(os);
}

template <class... Args>
CUTE_HOST void print_svg(std::ostream& os, const TiledCopy<Args...>& copy) {
  auto [S, TS] = copy.get_layoutS_MN();  // (m,n)->(tid,vid), tid->thr_idx
  auto [D, TD] = copy.get_layoutD_MN();  // (m,n)->(tid,vid), tid->thr_idx

  const int cell_width = 20;
  const int cell_height = 20;
  const int num_rows = size<0>(S) + size<0>(D) + 3;  // M + M + 3
  const int num_cols = size<1>(S) + size<1>(D) + 4;  // N + N + 3

  SVGBuilder builder(num_rows, num_cols, cell_width, cell_height);

  // header
  builder.print_header(os);
  // Commented prints
  // printf("%% LayoutS: "); print(S);  printf("\n");
  // printf("%% ThrIDS : "); print(TS); printf("\n");
  // printf("%% LayoutD: "); print(D);  printf("\n");
  // printf("%% ThrIDD : "); print(TD); printf("\n\n");

  // S starting at (1, 1)
  builder.print_comment(os, "S starting at (0, 0)");
  for (int m = 0; m < cute::size<0>(S); ++m) {
    for (int n = 0; n < cute::size<1>(S); ++n) {
      int thrid = S(m, n) % size(TS);
      int val_idx = S(m, n) / size(TS);
      int thr_idx = TS(thrid);

      builder.print_cell(
          os, m + 1, n + 1, thr_idx, val_idx, kColorMap[thr_idx % 8]);
    }
  }
  for (int m = 0; m < cute::size<0>(S); ++m) {
    builder.print_label(os, m + 1, 0, m);
  }
  for (int n = 0; n < cute::size<1>(S); ++n) {
    builder.print_label(os, 0, n + 1, n);
  }

  // D starting at (0, size<1>(S) + 3)
  builder.print_comment(os, "D starting at (0, size<1>(S) + 3)");
  for (int m = 0; m < cute::size<0>(D); ++m) {
    for (int n = 0; n < cute::size<1>(D); ++n) {
      int thrid = D(m, n) % size(TD);
      int val_idx = D(m, n) / size(TD);
      int thr_idx = TD(thrid);

      builder.print_cell(os,
                         m + 1,
                         n + size<1>(S) + 3,
                         thr_idx,
                         val_idx,
                         kColorMap[thr_idx % 8]);
    }
  }
  for (int m = 0; m < cute::size<0>(D); ++m) {
    builder.print_label(os, m + 1, size<1>(S) + 2, m);
  }
  for (int n = 0; n < cute::size<1>(D); ++n) {
    builder.print_label(os, 0, n + size<1>(S) + 3, n);
  }

  // footer
  builder.print_footer(os);
}

template <class Shape, class Stride>
CUTE_HOST void print_svg(std::ostream& os,
                         Layout<Shape, Stride> const& layout) {
  // TODO: add layout to SVG
}

}  // namespace detail

template <class... Args>
CUTE_HOST std::ostream& operator<<(std::ostream& os,
                                   const MMA_Atom<Args...>& mma_atom) {
  return os << make_tiled_mma(mma_atom);
}

template <class... Args>
CUTE_HOST std::ostream& operator<<(std::ostream& os,
                                   const TiledMMA<Args...>& mma) {
  detail::print_svg(os, mma);
  return os;
}

template <class... Args>
CUTE_HOST std::ostream& operator<<(std::ostream& os,
                                   const TiledCopy<Args...>& copy) {
  detail::print_svg(os, copy);
  return os;
}

}  // namespace llm