'Test for forward and backward function.'
import pytest
import torch
from neuralop.models.boundary_cond import ConstraintFunction, ConstraintLayer
from neuralop.models.boundary_cond import InequalityConstraintFunction, ConstraintWithIneq

device_name = "cpu"
#"""
if torch.cuda.is_available():
    device_name = "cuda:0"
#"""
device = torch.device(device_name)

"""
@pytest.mark.parametrize('m', [20, 43, 76])
def test_forward(m):

    torch.manual_seed(m*2359)
    batch_size = 5
    p = 3
    A = torch.randn(p, m, device=device)
    y = torch.randn(batch_size, m, device=device)
    b = torch.randn(batch_size, p, device=device)

    cl = ConstraintLayer(A, b)
    y_star = cl.forward(y)
    assert(torch.linalg.norm((A@y_star.T).T-b) < 1e-5)
    assert(torch.linalg.norm(y_star-y) > 1e-3)
    # assert(torch.linalg.norm(y-y_star) < 1.0e-3)


@pytest.mark.parametrize('m', [20, 43, 76])
def test_backward(m):
    torch.manual_seed(m*3434)
    batch_size = 5

    A = torch.randn(m, m, dtype=torch.double, device=device)
    y = torch.randn(batch_size, m, dtype=torch.double, requires_grad=True, device=device)
    b = torch.randn(batch_size, m, dtype=torch.double, device=device)
    #assert torch.autograd.gradcheck(ConstraintFunction.apply, (y, A, b), eps=1e-6, atol=1e-4)
    
    y_star = ConstraintFunction.apply(y, A, b)
    grad_output = torch.randn_like(y_star, device=device)
    grad_y = torch.autograd.grad(y_star, y, grad_outputs=grad_output)[0]
    grad_zero = torch.zeros_like(b, device=device)
    forward_grad_output = ConstraintFunction.apply(grad_output, A, grad_zero)
    assert torch.allclose(grad_y, forward_grad_output, atol=1e-4), "Backward is not equivalent to Forward"
"""

@pytest.mark.parametrize("dtype, eps", [(torch.double, 1e-8), (torch.float, 1e-6)])
def test_forward_ineq(dtype, eps):
    #   min ||x - y||^2/2   s.t.  x1+x2=1, x1>=0, x2>=0
    #   y = [2,0]  ->  x* = [1,0]
    A_eq = torch.tensor([[1.0, 1.0]], dtype=dtype, device=device)  # (1×2)
    b_eq = torch.tensor([1.0], dtype=dtype, device=device)  # (1,)
    A_ineq = torch.tensor([[-1.0, 0.0], # x1 >= 0
                           [0.0, -1.0]], dtype=dtype, device=device)   # x2 >= 0
    b_ineq = torch.tensor([0.0, 0.0], dtype=dtype, device=device)   # (2,)
    y = torch.tensor([[2.0, 0.0]], dtype=dtype, device=device)   # (1×2)

    y_proj = ConstraintWithIneq(A_eq, b_eq, A_ineq, b_ineq, eps=eps).forward(y)
    y_exp  = torch.tensor([[1.0, 0.0]], dtype=dtype, device=device)

    assert torch.allclose(y_proj, y_exp, atol=1e-6), f"got {y_proj}, expected {y_exp}"

@pytest.mark.parametrize("dtype", [torch.double, torch.float])
def test_backward_ineq(dtype):
    A_eq = torch.tensor([[1.0, 1.0]], dtype=dtype, device=device)
    b_eq = torch.tensor([1.0], dtype=dtype, device=device)
    A_ineq = torch.tensor([[-1.0, 0.0],
                           [0.0, -1.0]], dtype=dtype, device=device)
    b_ineq = torch.tensor([0.0, 0.0],    dtype=dtype, device=device)

    layer = ConstraintWithIneq(A_eq, b_eq, A_ineq, b_ineq)
    y = torch.randn(3, 2, dtype=dtype, requires_grad=True, device=device)
    y_star = layer(y)

    grad_seed = torch.randn_like(y_star)
    # compute backward
    y_star.backward(grad_seed)
    grad_y = y.grad
    grad_proj = layer(grad_seed)

    assert torch.allclose(grad_y, grad_proj, atol=1e-6), \
        f"backward mismatch:\n grad_y={grad_y}\n proj_grad={grad_proj}"

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)