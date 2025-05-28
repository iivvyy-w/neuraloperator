import torch
import torch.nn as nn
import numpy as np
import scipy
import cupy as cp
import cupyx as cpx
from cupyx.scipy.sparse.linalg import gmres as cpx_gmres
from cupyx.scipy.sparse.linalg import minres as cpx_minres


class ConstraintFunction(torch.autograd.Function):
    @staticmethod
    def solve(batch_size, block, rhs, device):
        y_star_v_star = []
        for i in range(batch_size):
            rhs_i = cp.asarray(rhs[i])
            solution, info = cpx_minres(block, rhs_i, tol=1e-12)
            y_star_v_star.append(cp.asarray(solution))
        y_star_v_star = cp.stack(y_star_v_star).get()
        y_star_v_star = torch.tensor(y_star_v_star).to(device)
        return y_star_v_star

    @staticmethod
    def forward(ctx, y, A, b):
        """
        Solve the optimization problem.
        ------
        Parameters:
            y (torch.Tensor): Input tensor `y` of shape (batch_size, input_dim).
            A (torch.Tensor): Input tensor `A` of shape (output_dim, input_dim).
            b (torch.Tensor): Input tensor `b` of shape (batch_size, output_dim).

        Output:
            y_star (torch.Tensor): Solution `y^*` of shape (batch_size, input_dim).
            v_star (torch.Tensor): Solution `v^*` of shape (batch_size, output_dim).
        """
        batch_size = y.shape[0]
        input_dim = y.shape[1]
        output_dim = A.shape[0]

        if A.device.type != y.device.type:
            A = A.to(y.device.type)
            b = b.to(y.device.type)

        # Create the block matrix
        Id = torch.eye(input_dim, device=y.device, dtype=y.dtype)  # 2I
        if A.numel() == 0:
            block_matrix = Id
        else:
            zero_block = torch.zeros((output_dim, output_dim), device=y.device, dtype=y.dtype)
            top_block = torch.cat([Id, 1/2*A.T], dim=1)
            bottom_block = torch.cat([1/2*A, zero_block], dim=1)
            block_matrix = torch.cat([top_block, bottom_block], dim=0)

        # Create the right-hand side vector
        rhs = torch.cat([y, 1/2*b], dim=1)  # Shape: (batch_size, input_dim + output_dim)
        """
        # Solve for each batch
        y_star_v_star = []
        for i in range(batch_size):
            solution = torch.linalg.solve(block_matrix, rhs[i])
            y_star_v_star.append(solution)
        """
        """
        block_matrix_scipy = scipy.sparse.coo_matrix(block_matrix.cpu().numpy())
        y_star_v_star = []
        for i in range(batch_size):
            rhs_i = rhs[i].cpu().numpy()
            solution, info = gmres(block_matrix_scipy, rhs_i)
            y_star_v_star.append(torch.tensor(solution).to(block_matrix.device))
        y_star_v_star = torch.stack(y_star_v_star, dim=0)
        """
        if y.device.type == 'cpu':
            y_star_v_star = []
            for i in range(batch_size):
                solution = torch.linalg.solve(block_matrix, rhs[i])
                y_star_v_star.append(solution)
            y_star_v_star = torch.stack(y_star_v_star, dim=0)
        elif y.device.type == 'cuda:0' or y.device.type == 'cuda':
            block_matrix_gpu = cpx.scipy.sparse.coo_matrix(cp.asarray(block_matrix))  # Convert to COO format for CuPy
            y_star_v_star = ConstraintFunction.solve(batch_size, block_matrix_gpu, rhs, device=y.device)
        else:
            raise TypeError(f"The device is {y.device.type}")

        # Extract y_star and v_star
        y_star = y_star_v_star[:, :input_dim]
        v_star = y_star_v_star[:, input_dim:]
        ctx.save_for_backward(A, y_star, v_star, block_matrix)
        return y_star

    @staticmethod
    def backward(ctx, grad_output):
        A, y_star, v_star, block_matrix = ctx.saved_tensors
        batch_size, input_dim = y_star.shape
        output_dim = A.shape[0]

        if grad_output.device.type != y_star.device.type:
            grad_output = grad_output.to(y_star.device.type)

        if A.numel() == 0:
            return grad_output, None, None
        
        zero_block = torch.zeros((batch_size, output_dim), device=y_star.device, dtype=y_star.dtype)
        rhs_grad = torch.cat([grad_output, zero_block], dim=1)  # Shape: (batch_size, input_dim + output_dim)
        if y_star.device.type == 'cpu':
            grad_solution = torch.linalg.solve(block_matrix.unsqueeze(0).expand(batch_size, -1, -1), rhs_grad.unsqueeze(-1))
            grad_y = grad_solution[:, :input_dim, 0]
        elif y_star.device.type == 'cuda:0' or y_star.device.type == 'cuda':
            block_matrix_gpu = cpx.scipy.sparse.coo_matrix(cp.asarray(block_matrix))  # Convert to COO format for CuPy
            grad_solution = ConstraintFunction.solve(batch_size, block_matrix_gpu, rhs_grad, device=block_matrix.device)
            grad_y = grad_solution[:, :input_dim]
        else:
            raise TypeError(f"The device is {y_star.device.type}")

        return grad_y, None, None


class ConstraintLayer(nn.Module):
    def __init__(self, A, b):
        """
        Initialize the ConstraintLayer.

        ------
        parameters:
            A (torch.Tensor): The matrix `A` in the optimization problem of Ay=b.
            b (torch.Tensor): The matrix 'b' in the optimization problem.
        """
        super().__init__()
        self.A = A
        self.b = b
        self.input_dim = A.shape[1]
        self.output_dim = b.shape[1]

    def forward(self, y):
        return ConstraintFunction.apply(y, self.A, self.b)


class InequalityConstraintFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, A_eq, b_eq, A_ineq, b_ineq, tol=1e-8, max_iters=50, eps=1e-6):
        """
        y: (B * n) from the previous training
        A_eq: (m_eq * n)
        b_eq: (m_eq,)
        A_ineq: (m_i * n)
        b_ineq: (m_i,)
        output y_proj (B * n)
        general form of quadratic programming
            min  q(y)=1/2 ||y-y_proj||^2_2
            st  A_eq y - b_eq = 0
                A_ineq y - b_ineq <= 0
        """
        device, dtype = y.device, y.dtype
        B, n = y.shape
        m_eq = A_eq.shape[0]
        m_i = A_ineq.shape[0]

        A_eq = A_eq.to(device=device, dtype=dtype)
        b_eq = b_eq.to(device=device, dtype=dtype)
        A_ineq = A_ineq.to(device=device, dtype=dtype)
        b_ineq = b_ineq.to(device=device, dtype=dtype)

        y_proj = torch.empty_like(y)
        # save a few things for backward
        saved = {'A_eq': A_eq, 'A_ineq': A_ineq, 'tol': tol}

        for i in range(B):  # for each single case in one batch
            xi = y[i]
            # 1) initialize active set
            viol = A_ineq @ xi - b_ineq  # initial point xi
            W = set((viol >= -tol).nonzero(as_tuple=False).view(-1).tolist())

            for it in range(max_iters):
                # build active‐constraint matrices
                if len(W) > 0:
                    A_W = A_ineq[list(W)]
                    b_W = b_ineq[list(W)]
                    A_act = torch.cat([A_eq, A_W], dim=0)   # (m_eq+|W| × n)
                    b_act = torch.cat([b_eq, b_W], dim=0)   # (m_eq+|W|,)
                else:
                    A_act = A_eq
                    b_act = b_eq

                m_act = A_act.shape[0]

                # Schur complement solve:  M λ = A_act y - b_act
                # with M = A_act A_actᵀ + eps·I
                M = A_act @ A_act.T                       # (m_act × m_act)
                if m_act > 0:
                    #M = M + eps * torch.eye(m_act, device=device, dtype=dtype)
                    rhs = (A_act @ xi) - b_act               # (m_act,)
                    #v_star = torch.linalg.solve(M, rhs)         # (m_act,)
                    v_star = torch.linalg.pinv(M) @ rhs
                    y_star = xi - A_act.T @ v_star                # (n,)
                else:
                    y_star = xi.clone()

                # check multipliers on the *inequality* part
                v_i = v_star[m_eq:]  # these correspond to A_W
                if v_i.numel() == 0 or v_i.min() >= -tol:
                    # check all violations
                    viol_all = A_ineq @ y_star - b_ineq
                    max_viol, j_max = viol_all.max(0)
                    if max_viol <= tol:
                        # done (all inequality constraints are satisfied)
                        break
                    else:
                        W.add(int(j_max))
                else:
                    # drop one with most negative multiplier
                    j_drop = int(v_i.argmin().item())
                    # map j_drop back to the original index in A_ineq
                    # if W is a list, drop W[j_drop]; if a set, track ordering
                    W_list = list(W)
                    W.remove(W_list[j_drop])

            y_proj[i] = y_star

        # save for backward
        ctx.saved_data = saved
        ctx.save_for_backward(y, y_proj)
        return y_proj

    @staticmethod
    def backward(ctx, grad_y):
        """
        The derivative of the projection is the projection of the gradient
        onto the tangent space of the active constraints.  Concretely,
        one can resolve a small KKT system with grad_x as the "rhs" and
        zeroing out the components in the normal cone.  For brevity, here
        we simply pass the gradient through the same projection operator
        (i.e. the Jacobian is symmetric idempotent), which is correct
        for the Euclidean projection onto a convex set.
        """
        y, y_proj = ctx.saved_tensors
        A_eq = ctx.saved_data['A_eq']
        A_ineq = ctx.saved_data['A_ineq']
        tol = ctx.saved_data['tol']

        # **simplest** consistent backward: project grad_x in the same way
        grad_y = InequalityConstraintFunction.forward(None, grad_y, A_eq, torch.zeros_like(A_eq[:, 0]),
                                                      A_ineq, torch.zeros(A_ineq.shape[0], device=grad_y.device),
                                                      tol=tol, max_iters=20)
        # no grads for the A/b’s
        return grad_y, None, None, None, None, None, None


class ConstraintWithIneq(nn.Module):
    def __init__(self, A_eq, b_eq, A_ineq, b_ineq, tol=1e-8, max_iters=50, eps=1e-8):
        super().__init__()
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.A_ineq = A_ineq
        self.b_ineq = b_ineq
        self.tol = tol
        self.max_iters = max_iters
        self.eps = eps

    def forward(self, y):
        return InequalityConstraintFunction.apply(
            y,
            self.A_eq, self.b_eq,
            self.A_ineq, self.b_ineq,
            self.tol, self.max_iters, self.eps
        )


class ConstraintNonlin(nn.Module):
    def __init__(self, eq_funcs, ineq_funcs, n_iters=3, tol=1e-8, eps=1e-8):
        super().__init__()
        self.eq_funcs = eq_funcs     # list of callables ci(y) == 0
        self.ineq_funcs = ineq_funcs   # list of callables ci(y) >= 0
        self.n_iters = n_iters
        self.proj = ConstraintWithIneq(None, None, None, None,
                                       tol=tol, max_iters=50, eps=eps)

    def forward(self, y0, y_target):
        y_k = y0
        for _ in range(self.n_iters):
            # 1) evaluate and linearize
            cE = torch.stack([f(y_k) for f in self.eq_funcs], dim=-1)  # (B, m_eq)
            cI = torch.stack([f(y_k) for f in self.ineq_funcs], dim=-1)  # (B, m_i)
            JE = torch.stack([
                torch.autograd.grad(cE[:, j].sum(), y_k, create_graph=True)[0]
                for j in range(cE.shape[-1])
            ], dim=1)  # (B,m_eq,n)
            JI = torch.stack([
                torch.autograd.grad(cI[:, j].sum(), y_k, create_graph=True)[0]
                for j in range(cI.shape[-1])
            ], dim=1)  # (B,m_i,n)

            # 2) form A_eq, b_eq, A_ineq, b_ineq
            b_eq = (JE @ y_k.unsqueeze(-1)).squeeze(-1) - cE
            b_ineq = -(JI @ y_k.unsqueeze(-1)).squeeze(-1) + cI

            # 3) call your projection
            y_k = self.proj(y_target, JE, b_eq, -JI, b_ineq)

        return y_k


def generate_bc0(batch_size, channels, height, width):
    n_boundary_points = 0
    n_boundary_points = 2 * (height + width) * channels - 4 * channels
    n_total_points = 0
    n_total_points = height * width * channels
    
    A = torch.zeros((n_boundary_points, n_total_points))
    b = torch.zeros((batch_size, n_boundary_points))

    boundary_indices = []
    for c in range(channels):
        offset = c * height * width
        boundary_indices.extend(offset + np.arange(width))
        boundary_indices.extend(offset + (height - 1) * width + np.arange(width))
        boundary_indices.extend(offset + np.arange(0, height * width, width))
        boundary_indices.extend(offset + np.arange(width - 1, height * width, width))
    """
    A = torch.zeros((int(n_boundary_points/2)+2, n_total_points))
    b = torch.zeros((batch_size, int(n_boundary_points/2)+2))

    boundary_indices = []
    for c in range(channels):
        offset = c * height * width
        boundary_indices.extend(offset + np.arange(width))
        boundary_indices.extend(offset + (height - 1) * width + np.arange(width))
    """
    boundary_indices = list(set(boundary_indices))
    
    for row_idx, col_idx in enumerate(boundary_indices):
        A[row_idx, col_idx] = 1
    #scale = torch.linalg.norm(A)
    
    return A, b


def neumann(x, gx, direction='normal', pos=[1, 1, 1, 1]):
    
    """
    ------
    Parameters:
        batch_size (int):
        channels (int): output channel
        height
        width
        gx (func): the boundary solves dy/dx = g(x)
        direction: normal/tangential
        pos: which boundary is being constrained, up bottom left right, 1 is apply, 0 is not apply
    """
    batch_size, channels, height, width = x.shape
    n_total_points = height * width * channels
    n_boundary_points = ((pos[0]+pos[1])*width + (pos[2]+pos[3])*height) * channels - (pos[0]+pos[1])*(pos[2]+pos[3]) * channels

    A = torch.zeros((n_boundary_points, n_total_points))
    b = torch.zeros((batch_size, n_boundary_points))
    
    boundary_indices = []
    neighbor_indices = []
    boundary_x_coords = []
    boundary_y_coords = []
    
    for c in range(channels):
        offset = c * height * width
        
        if pos[0] == 1:  # Top
            for y in range(width):
                boundary_idx = offset + y
                neighbor_idx = offset + width + y
                boundary_indices.append(boundary_idx)
                neighbor_indices.append(neighbor_idx)
                boundary_x_coords.append(0)
                boundary_y_coords.append(y)
        
        if pos[1] == 1:  # Bottom
            for y in range(width):
                boundary_idx = offset + (height - 1) * width + y
                neighbor_idx = offset + (height - 2) * width + y
                boundary_indices.append(boundary_idx)
                neighbor_indices.append(neighbor_idx)
                boundary_x_coords.append(height-1)
                boundary_y_coords.append(y)
        
        if pos[2] == 1:
            for x in range(height):
                boundary_idx = offset + x * width
                neighbor_idx = offset + x * width + 1
                boundary_indices.append(boundary_idx)
                neighbor_indices.append(neighbor_idx)
                boundary_x_coords.append(x)
                boundary_y_coords.append(0)
        
        if pos[3] == 1:
            for x in range(height):
                boundary_idx = offset + x * width + (width - 1)
                neighbor_idx = offset + x * width + (width - 2)
                boundary_indices.append(boundary_idx)
                neighbor_indices.append(neighbor_idx)
                boundary_x_coords.append(x)
                boundary_y_coords.append(width-1)

    constrained_points = set()
    row_idx = 0
    hx = 1/width
    hy = 1/height
    for b_idx, n_idx, x, y in zip(boundary_indices, neighbor_indices, 
                                  boundary_x_coords, boundary_y_coords):
        if b_idx in constrained_points:
            continue
        A[row_idx, n_idx] = 1
        A[row_idx, b_idx] = -1  # outward
        b[:, row_idx] = gx(hx*x, hy*y)  # inputs are two int and output is one int
        constrained_points.add(b_idx)
        row_idx += 1
    return A, b
