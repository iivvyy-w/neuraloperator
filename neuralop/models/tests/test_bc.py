'Test for forward and backward function.'
import pytest
import torch
from neuralop.models.boundary_cond import ConstraintFunction, ConstraintLayer, generate_constraint0


@pytest.mark.parametrize('m', [20, 43, 76])
def test_forward(m):
    torch.manual_seed(m*2359)
    batch_size = 5

    A = torch.randn(m, m)
    y = torch.randn(batch_size, m)
    b = torch.randn(batch_size, m)

    cl = ConstraintLayer(A, b)
    y_star = cl.forward(y)

    assert(torch.linalg.norm((A@y_star.T).T-b) < 0.1)
    # assert(torch.linalg.norm(y-y_star) < 1.0e-3)


@pytest.mark.parametrize('m', [20, 43, 76])
def test_backward(m):
    torch.manual_seed(m*3434)
    batch_size = 5

    A = torch.randn(m, m, requires_grad = True)
    y = torch.randn(batch_size, m, requires_grad = True)
    b = torch.randn(batch_size, m)
    cl = ConstraintLayer(A, b)
    y_star = cl.forward(y)

    # Generate random gradient output
    grad_output = torch.randn_like(y_star)

    # Compute backward gradients using PyTorch autograd
    grad_y_manual, _, _ = ConstraintFunction.backward(None, grad_output)

    # Compute numerical gradients with gradcheck
    assert torch.autograd.gradcheck(ConstraintFunction.apply, (y, A, b), eps=1e-6, atol=1e-4)

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)