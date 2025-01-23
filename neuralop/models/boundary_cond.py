from models import FNO
import torch
import torch.nn as nn
import numpy as np


class ConstraintFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(clayer, y, A, b):
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

        # Create the block matrix
        Id = 2 * torch.eye(clayer.input_dim, device=y.device, dtype=y.dtype)  # 2I
        zero_block = torch.zeros((clayer.output_dim, clayer.output_dim), device=y.device, dtype=y.dtype)
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
        y_star = y_star_v_star[:, :clayer.input_dim]
        v_star = y_star_v_star[:, clayer.input_dim:]

        clayer.save_for_backward(A, y_star, v_star)
        return y_star


    @staticmethod
    def backward(ctx, grad_output):


        return grad_input, grad_weight, grad_bias


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
        A = self.A
        b = self.b
        batch_size = y.shape[0]

        # Create the block matrix
        Id = 2 * torch.eye(self.input_dim, device=y.device, dtype=y.dtype)  # 2I
        zero_block = torch.zeros((self.output_dim, self.output_dim), device=y.device, dtype=y.dtype)
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
        y_star = y_star_v_star[:, :self.input_dim]
        v_star = y_star_v_star[:, self.input_dim:]

        return y_star
    
    def backward():


def generate_constraint(height, width, channels):
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

    for row_idx, col_idx in enumerate(boundary_indices):
        A[row_idx, col_idx] = 1
    return A, b