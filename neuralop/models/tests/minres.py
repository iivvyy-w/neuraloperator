import os
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from neuralop.data.datasets    import load_darcy_flow_small
from neuralop.models.boundary_cond import ConstraintLayer, generate_bc0


device = torch.device("cpu")
stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outdir = f"MINRESconvergence_{stamp}"
os.makedirs(outdir, exist_ok=True)

_, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000, batch_size=32,
    test_resolutions=[32], n_tests=[10],
    test_batch_sizes=[32],
)
data_processor = data_processor.to(device)
test_ds = test_loaders[32].dataset

raw = test_ds[0]
data = data_processor.preprocess(raw, batched=False)
y4d = data['y'].to(device)         # shape = (p, q, m, n)
p, q, m, n = y4d.shape

A, b = generate_bc0(p, q, m, n)
layer = ConstraintLayer(A, b).to(device)

y_vec = y4d[0, 0].reshape(1, m*n).to(device)
y_vec = y_vec.detach().requires_grad_(True)

F_y = layer(y_vec)

v = torch.ones_like(F_y)     # the direction vector
if y_vec.grad is not None:
    y_vec.grad.zero_()       # clear any old gradient
F_y.backward(v)              # calls ConstraintFunction.backward
Jv = y_vec.grad              # this is J_F(y)[v]

h_values = np.logspace(3, -3, num=20)
errors = []

for h in h_values:
    F_y_ph = layer(y_vec + h * v)
    delta = F_y_ph - F_y - h * Jv
    errors.append(delta.norm().item())

errors = np.array(errors)

# 8) Plot & save
fig, ax = plt.subplots()
ax.loglog(h_values, errors, marker='o', label='Truncation error $E(h)$')
ax.loglog(h_values, h_values**2, marker='x', label='$O(h^2)$ ref')
ax.set_xlabel('h')
ax.set_ylabel('Error $E(h)$')
ax.legend()
ax.grid(True)
fig.suptitle('FD Truncation Error vs. $O(h^2)$')

outpath = os.path.join(outdir, "fd_convergence.png")
fig.savefig(outpath, dpi=300, bbox_inches='tight')
plt.show()

print(f"Saved figure to {outpath}")