#include "array.h"

#include <gtest/gtest.h>
#include <sys/types.h>
#include <torch/torch.h>

#include "range.h"

namespace llm {
TEST(ArrayTest, Empty) {
  uint32_t* data = nullptr;
  Array array(data, {});
  EXPECT_EQ(array.size(), 0);
  EXPECT_EQ(array.data(), nullptr);
}

TEST(ArrayTest, Basic) {
  // 3x4 matrix
  std::vector<uint32_t> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  Array array(data.data(), {3, 4});
  EXPECT_EQ(array.size(), 12);
  EXPECT_EQ(array.shape(), std::vector<size_t>({3, 4}));
  EXPECT_EQ(array.stride(), std::vector<size_t>({4, 1}));

  // test offset indexing
  for (size_t i : range<size_t>(12)) {
    EXPECT_EQ(array[i], data[i]);
    EXPECT_EQ(array(i), data[i]);
  }

  // test coordinate indexing
  for (size_t i : range<size_t>(3)) {
    for (size_t j : range<size_t>(4)) {
      const size_t offset = i * 4 + j;
      // test multi-dimensional indexing
      EXPECT_EQ(array(i, j), data[offset]);
      // test Coord indexing
      Coord coord = {i, j};
      EXPECT_EQ(array(coord), data[offset]);
      EXPECT_EQ(array[coord], data[offset]);
    }
  }

  // test update
  std::vector<uint32_t> orig_data = data;
  for (size_t i : range<size_t>(12)) {
    array[i] *= i;
    EXPECT_EQ(array[i], orig_data[i] * i);
  }
}

TEST(ArrayTest, Tensor) {
  // test with torch tensor
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  auto options = torch::dtype(dtype).device(device);
  auto tensor = torch::randn({3, 4, 5}, options);

  Array array(tensor.data_ptr<float>(), {3, 4, 5});
  EXPECT_EQ(array.size(), 60);
  EXPECT_EQ(array.shape(), std::vector<size_t>({3, 4, 5}));
  EXPECT_EQ(array.stride(), std::vector<size_t>({20, 5, 1}));

  // test offset indexing
  auto tensor_1d = tensor.view(-1);
  for (size_t i : range<size_t>(60)) {
    EXPECT_EQ(array[i], tensor_1d[i].item<float>());
    EXPECT_EQ(array(i), tensor_1d[i].item<float>());
  }

  // test coordinate indexing
  for (size_t i : range<size_t>(3)) {
    for (size_t j : range<size_t>(4)) {
      for (size_t k : range<size_t>(5)) {
        const size_t offset = i * 4 * 5 + j * 5 + k;
        // test multi-dimensional indexing
        EXPECT_EQ(array(i, j, k), tensor[i][j][k].item<float>());
        // test Coord indexing
        Coord coord = {i, j, k};
        EXPECT_EQ(array(coord), tensor[i][j][k].item<float>());
        EXPECT_EQ(array[coord], tensor[i][j][k].item<float>());
      }
    }
  }
}

}  // namespace llm