import torch
import torch.nn as nn
import numpy as np


class ConstraintFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, A, b):
        """
        Solve the optimization problem.
        ------
        Parameters:
            y (torch.Tensor): Input tensor `y` of shape (batch_size, input_dim).
            b (torch.Tensor): Input tensor `b` of shape (batch_size, output_dim).

        Output:
            y_star (torch.Tensor): Solution `y^*` of shape (batch_size, input_dim).
            v_star (torch.Tensor): Solution `v^*` of shape (batch_size, output_dim).
        """
        batch_size = y.shape[0]
        input_dim = A.shape[1]
        output_dim = A.shape[0]

        # Create the block matrix
        Id = 2 * torch.eye(input_dim, device=y.device, dtype=y.dtype)  # 2I
        zero_block = torch.zeros((output_dim, output_dim), device=y.device, dtype=y.dtype)
        top_block = torch.cat([Id, A.T], dim=1)
        bottom_block = torch.cat([A, zero_block], dim=1)
        block_matrix = torch.cat([top_block, bottom_block], dim=0)

        # Create the right-hand side vector
        rhs = torch.cat([2 * y, b], dim=1)  # Shape: (batch_size, input_dim + output_dim)

        # Solve for each batch
        y_star_v_star = []
        for i in range(batch_size):
            solution = torch.linalg.solve(block_matrix, rhs[i])
            y_star_v_star.append(solution)

        y_star_v_star = torch.stack(y_star_v_star, dim=0)

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

        zero_block = torch.zeros((batch_size, output_dim), device=y_star.device, dtype=y_star.dtype)
        rhs_grad = torch.cat([grad_output, zero_block], dim=1)  # Shape: (batch_size, input_dim + output_dim)
        grad_solution = torch.linalg.solve(block_matrix.unsqueeze(0).expand(batch_size, -1, -1), rhs_grad.unsqueeze(-1))
        grad_y = grad_solution[:, :input_dim, 0]

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
        self.output_dim = A.shape[0]

    def forward(self, y):
        return ConstraintFunction.apply(y, self.A, self.b)


"""
class ConstraintLayer(nn.Module):
    def __init__(self, A, b):
        
        Initialize the ConstraintLayer.

        ------
        parameters:
            A (torch.Tensor): The matrix `A` in the optimization problem of Ay=b.
            b (torch.Tensor): The matrix 'b' in the optimization problem.
        
        super().__init__()
        self.A = A
        self.b = b

    def forward(self, y, f=True):
        
        Solve the optimization problem.
        ------
        Parameters:
            y (torch.Tensor): Input tensor `y` of shape (batch_size, input_dim).

        Output:
            y_star (torch.Tensor): Solution `y^*` of shape (batch_size, input_dim).

        
        A = self.A
        b = self.b
        batch_size = y.shape[0]
        input_dim = A.shape[1]
        output_dim = A.shape[0]

        # Create the block matrix
        Id = 2 * torch.eye(input_dim, device=y.device, dtype=y.dtype)  # 2I
        zero_block = torch.zeros((output_dim, output_dim), device=y.device, dtype=y.dtype)
        top_block = torch.cat([Id, A.T], dim=1)
        bottom_block = torch.cat([A, zero_block], dim=1)
        block_matrix = torch.cat([top_block, bottom_block], dim=0)

        # Create the right-hand side vector
        if f:
            rhs = torch.cat([2 * y, b], dim=1)  # Shape: (batch_size, input_dim + output_dim)
        else:
            zero_block = torch.zeros((batch_size, output_dim), device=y.device, dtype=y.dtype)
            rhs = torch.cat([y, zero_block], dim=1)  # Shape: (batch_size, input_dim + output_dim)

        # Solve for each batch
        y_star_v_star = []
        for i in range(batch_size):
            solution = torch.linalg.solve(block_matrix, rhs[i])
            y_star_v_star.append(solution)

        y_star_v_star = torch.stack(y_star_v_star, dim=0)

        # Extract y_star and v_star
        y_star = y_star_v_star[:, :input_dim]
        v_star = y_star_v_star[:, input_dim:]

        #self.save_for_backward(A, y_star, v_star, block_matrix)
        return y_star

    def backward(self, grad_output):

        return self.forward(grad_output, f=False)#, None, None
"""

def generate_bc0(height, width, channels):
    n_boundary_points = 2 * (height + width) * channels - 4 * channels
    n_total_points = height * width * channels

    A = torch.zeros((n_boundary_points, n_total_points))
    b = torch.zeros((n_boundary_points,))

    boundary_indices = []
    for c in range(channels):
        offset = c * height * width
        boundary_indices.extend(offset + np.arange(width))
        boundary_indices.extend(offset + (height - 1) * width + np.arange(width))
        boundary_indices.extend(offset + np.arange(0, height * width, width))
        boundary_indices.extend(offset + np.arange(width - 1, height * width, width))

    boundary_indices = list(set(boundary_indices))

    for row_idx, col_idx in enumerate(boundary_indices):
        A[row_idx, col_idx] = 1
    return A, b


def neumann_linear(height, width, channels, alpha=0, beta=0, scale=1.0,
                   direction='normal', function=):

    n_boundary_points = 2 * (height + width) * channels - 4 * channels
    n_total_points = height * width * channels
    
    A = torch.zeros((n_boundary_points, n_total_points))
    b = torch.zeros((n_boundary_points,))
    
    boundary_indices = []
    neighbor_indices = []
    boundary_x_coords = []
    
    for c in range(channels):
        offset = c * height * width
        
        # Top boundary (y=0)
        for x in range(width):
            boundary_idx = offset + x
            neighbor_idx = offset + width + x
            boundary_indices.append(boundary_idx)
            neighbor_indices.append(neighbor_idx)
            boundary_x_coords.append(x)
        
        # Bottom boundary (y=height-1)
        for x in range(width):
            boundary_idx = offset + (height - 1) * width + x
            neighbor_idx = offset + (height - 2) * width + x
            boundary_indices.append(boundary_idx)
            neighbor_indices.append(neighbor_idx)
            boundary_x_coords.append(x)
        
        # Left boundary (x=0)
        for y in range(height):
            boundary_idx = offset + y * width
            neighbor_idx = offset + y * width + 1
            boundary_indices.append(boundary_idx)
            neighbor_indices.append(neighbor_idx)
            boundary_x_coords.append(y)
        
        # Right boundary (x=width-1)
        for y in range(height):
            boundary_idx = offset + y * width + (width - 1)
            neighbor_idx = offset + y * width + (width - 2)
            boundary_indices.append(boundary_idx)
            neighbor_indices.append(neighbor_idx)
            boundary_x_coords.append(y)

    for row_idx, (b_idx, n_idx, x) in enumerate(zip(boundary_indices, neighbor_indices, boundary_x_coords)):
        A[row_idx, n_idx] = 1
        A[row_idx, b_idx] = -1
        b[row_idx] = (alpha * x + beta) * scale

    return A, b
