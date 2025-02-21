'Test for error functions.'
import pytest
import torch
from neuralop.models.errors import boundary_L2error


@pytest.mark.parametrize('m', [2, 20, 43])
def test_L2norm(m):
    torch.manual_seed(m*2359)

    y = torch.randn((1, 1, m, m))
    out = y
    error = boundary_L2error(y, out)
    assert(error == 0)