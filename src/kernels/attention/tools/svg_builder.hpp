
#pragma once
#include <absl/strings/str_format.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/config.hpp>
#include <ostream>

namespace cute {

class SVGBuilder {
  int num_rows_;
  int num_cols_;
  int cell_width_;
  int cell_height_;

 public:
  SVGBuilder(int num_rows,
             int num_cols,
             int cell_width = 20,
             int cell_height = 20)
      : num_rows_(num_rows),
        num_cols_(num_cols),
        cell_width_(cell_width),
        cell_height_(cell_height) {}

  void print_header(std::ostream& os) const {
    os << absl::StreamFormat(
        "<svg width=\"100%%\" height=\"100%%\" viewBox=\"0 0 %d %d\" "
        "preserveAspectRatio=\"xMidYMid meet\" "
        "xmlns=\"http://www.w3.org/2000/svg\">\n",
        num_cols_ * cell_width_,
        num_rows_ * cell_height_);
  }

  void print_footer(std::ostream& os) const { os << "</svg>\n"; }

  void print_label(std::ostream& os,
                   int m,
                   int n,
                   int val,
                   const char* text_color = "blue",
                   int font_size = 12) const {
    assert(m < num_rows_);
    assert(n < num_cols_);
    os << absl::StreamFormat(
        "<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" "
        "alignment-baseline=\"central\" stroke=\"%s\" "
        "font-size=\"%d\">%d</text>\n",
        n * cell_width_ + cell_width_ / 2,
        m * cell_height_ + cell_height_ / 2,
        text_color,
        font_size,
        val);
  }

  void print_cell(std::ostream& os,
                  int m,
                  int n,
                  int trd_idx,
                  const std::string& val,
                  const char* color,
                  int font_size = 8) const {
    assert(m < num_rows_);
    assert(n < num_cols_);
    // draw a cell with background color
    os << absl::StreamFormat(
        "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" "
        "fill=\"%s\" stroke=\"black\"/>\n",
        n * cell_width_,
        m * cell_height_,
        cell_width_,
        cell_height_,
        color);

    // draw thread index
    os << absl::StreamFormat(
        "<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" "
        "alignment-baseline=\"central\" font-size=\"%d\">T%d</text>\n",
        n * cell_width_ + cell_width_ / 2,
        m * cell_height_ + cell_height_ / 4,
        font_size,
        trd_idx);

    // draw value index
    os << absl::StreamFormat(
        "<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" "
        "alignment-baseline=\"central\" font-size=\"%d\">%s</text>\n",
        n * cell_width_ + cell_width_ / 2,
        m * cell_height_ + cell_height_ * 3 / 4,
        font_size,
        val);
  }

  void print_comment(std::ostream& os, const std::string& comment) const {
    os << absl::StreamFormat("<!-- %s -->\n", comment);
  }
};

}  // namespace cute