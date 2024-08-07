import sys

import pytest
import torch

from scalellm._C.kernels import add_test

def test_add():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    c = add_test(a, b)
    assert torch.all(c == a + b)


if __name__ == "__main__":
    pytest.main(sys.argv)
