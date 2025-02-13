import torch
import torch.nn as nn
from boundary_cond import ConstraintLayer, generate_bc0
from neuralop.models.fno import FNO
import sys


"""
print("A shape:", A.shape)  # Should be (n_boundary_points, n_total_points)
print("b shape:", b.shape)  # Should be (n_boundary_points,)

# Check if A has only 0s and 1s
print("Unique values in A:", torch.unique(A))

# Check if b is all zeros
#print(A)
print("b values:", b)
"""
"""
model = FNO(n_modes=(16, 16),
             in_channels=1, 
             out_channels=1,
             hidden_channels=16,
             constraint=False)
print("Model instantiated!")
x = torch.randn(1, 1, 10, 10)  # Example input
try:
    output = model(x)
    print("Forward pass completed!")
    print(output)
except Exception as e:
    print(f"Error during forward pass: {e}")
"""

x = torch.randn(1, 1, 10, 10)  # Example input
p, q, m, n = x.shape
A, b = generate_bc0(p, q, m, n)
model2 = FNO(n_modes=(16, 16),
             in_channels=1,
             out_channels=1,
             hidden_channels=16,
             constraint=True)
try:
    output2 = model2(x)
    print("Forward pass completed!")
    print(output2)
except Exception as e:
    print(f"Error during forward pass: {e}")

#assert torch.allclose(output, output2, atol=1e-4)
sys.stdout.flush()