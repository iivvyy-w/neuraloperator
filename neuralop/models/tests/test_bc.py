'Test for forward and backward function.'
import pytest
import torch
from neuralop.models.boundary_cond import ConstraintFunction, ConstraintLayer


@pytest.mark.parametrize('m', [20, 43, 76])
def test_forward(m):
    torch.manual_seed(m*2359)
    batch_size = 5
    p = 3
    A = torch.randn(p, m)
    y = torch.randn(batch_size, m)
    b = torch.randn(batch_size, p)

    cl = ConstraintLayer(A, b)
    y_star = cl.forward(y)
    assert(torch.linalg.norm((A@y_star.T).T-b) < 1e-5)
    assert(torch.linalg.norm(y_star-y) > 1e-3)
    # assert(torch.linalg.norm(y-y_star) < 1.0e-3)



@pytest.mark.parametrize('m', [20, 43, 76])
def test_backward(m):
    torch.manual_seed(m*3434)
    batch_size = 5

    A = torch.randn(m, m, dtype=torch.double)
    y = torch.randn(batch_size, m, dtype=torch.double, requires_grad=True)
    b = torch.randn(batch_size, m, dtype=torch.double)
    assert torch.autograd.gradcheck(ConstraintFunction.apply, (y, A, b), eps=1e-6, atol=1e-4)

    y_star = ConstraintFunction.apply(y, A, b)
    grad_output = torch.randn_like(y_star)
    grad_y = torch.autograd.grad(y_star, y, grad_outputs=grad_output)[0]
    forward_grad_output = ConstraintFunction.apply(grad_output, A, torch.zeros_like(b))
    assert torch.allclose(grad_y, forward_grad_output, atol=1e-4), "Backward is not equivalent to Forward"


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)