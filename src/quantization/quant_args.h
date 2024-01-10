#pragma once

#include <ostream>
#include <string>

#include "common/macros.h"

namespace llm {

struct QuantArgs {
  DEFINE_ARG(std::string, quant_method) = "";

  // quantization bits
  DEFINE_ARG(int64_t, bits) = 0;

  // quantization group size
  DEFINE_ARG(int64_t, group_size) = 0;

  // aka act_order, true results in better quantisation accuracy.
  DEFINE_ARG(bool, desc_act) = false;

  DEFINE_ARG(bool, true_sequential) = false;
};

inline std::ostream& operator<<(std::ostream& os, const QuantArgs& args) {
  os << "QuantArgs: [";
  os << "quant_method: " << args.quant_method();
  os << ", bits: " << args.bits();
  os << ", group_size: " << args.group_size();
  os << ", desc_act: " << args.desc_act();
  os << ", true_sequential: " << args.true_sequential();
  os << "]";
  return os;
}

}  // namespace llm
