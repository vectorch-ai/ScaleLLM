#include <cute/tensor.hpp>
using namespace cute;

int main() {
  MMA_Atom mma = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{};
  print_latex(mma);
  return 0;
}