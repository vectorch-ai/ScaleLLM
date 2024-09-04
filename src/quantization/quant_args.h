#pragma once

#include <ostream>
#include <string>

#include "common/macros.h"

namespace llm {

struct QuantArgs {
  DEFINE_ARG(std::string, quant_method);

  // quantization bits
  DEFINE_ARG(int64_t, bits) = 0;

  // quantization group size
  DEFINE_ARG(int64_t, group_size) = 0;

  // aka act_order, true results in better quantisation accuracy.
  DEFINE_ARG(bool, desc_act) = false;

  // whether the input is symmetric
  DEFINE_ARG(bool, is_sym) = false;

  // whether has zero point
  DEFINE_ARG(bool, zero_point) = false;

  // check if weights can be fused
  bool can_be_fused() const {
    // can't fuse quantized weights if desc_act is true
    return quant_method().empty() || !desc_act();
  }
};

inline std::ostream& operator<<(std::ostream& os, const QuantArgs& args) {
  os << "QuantArgs: [";
  os << "quant_method: " << args.quant_method();
  os << ", bits: " << args.bits();
  os << ", group_size: " << args.group_size();
  os << ", desc_act: " << args.desc_act();
  os << ", is_sym: " << args.is_sym();
  os << ", zero_point: " << args.zero_point();
  os << "]";
  return os;
}

}  // namespace llm
