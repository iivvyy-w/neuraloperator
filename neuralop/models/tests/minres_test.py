import os, datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

from neuralop.data.datasets import load_darcy_flow_small
from neuralop.models.boundary_cond import ConstraintLayer, generate_bc0

device = torch.device("cuda")
stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outdir = f"MINRESconvergence_{stamp}"
os.makedirs(outdir, exist_ok=True)

_, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000, batch_size=32,
    test_resolutions=[32], n_tests=[10],
    test_batch_sizes=[32],
)
data_processor = data_processor.to(device)
raw = test_loaders[32].dataset[0]
data = data_processor.preprocess(raw, batched=False)
y4d = data['y'].to(device)           # shape (p, q, m, n)
p, q, m, n = y4d.shape

A, b = generate_bc0(p, q, m, n)
layer = ConstraintLayer(A, b).to(device)

y_vec = y4d[0, 0].reshape(1, m*n).requires_grad_(True)   # shape (1, input_dim)

F_y = layer(y_vec)                                    # F(y)
v = torch.ones_like(F_y)                            # direction = all‐ones
y_vec.grad = None                                     # clear any old grad
F_y.backward(v)                                        # invokes ConstraintFunction.backward
Jv = y_vec.grad.clone()                              # now holds J_F(y)[v]

h_values = np.logspace(3, -3, num=20)
errors_fd = []   # forward‐difference remainder
errors_cd = []   # central‐difference error

for h in h_values:
    # forward remainder R(h) = F(y+h) - F(y) - h Jv
    F_ph = layer(y_vec + h*v)
    R_h = F_ph - F_y - h*Jv
    errors_fd.append(R_h.norm().item())

    # central‐difference test D_c(h) = [F(y+h)-F(y-h)]/(2h)
    F_mh = layer(y_vec - h*v)
    D_c = (F_ph - F_mh) / 2
    errors_cd.append((D_c - h*Jv).norm().item())

errors_fd = np.array(errors_fd)
errors_cd = np.array(errors_cd)

fig, ax = plt.subplots()
ax.loglog(h_values, errors_fd, marker='o', label='Forward‐remainder $O(h^2)$')
ax.loglog(h_values, errors_cd, marker='x', label='Central‐diff error $O(h^2)$')
ax.loglog(h_values, h_values**2, linestyle='--', label='$h^2$ reference')
ax.set_xlabel('h')
ax.set_ylabel('Error')
ax.grid(True, which='both')
ax.legend()
fig.suptitle('Second‐order Convergence Checks')
outpath = os.path.join(outdir, 'fd_convergence.png')
fig.savefig(outpath, dpi=300, bbox_inches='tight')
print(f"Figure saved to {outpath}")
plt.show()
