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

batch_size = 1
p = 252
m = 4096
A = torch.randn(p, m, device=device)
y = torch.randn(batch_size, m, device=device)
b = torch.randn(batch_size, p, device=device)

cl = ConstraintLayer(A, b).to(device)

y_vec = y.requires_grad_(True)   # shape (1, input_dim)

F_y = cl(y_vec)                                    # F(y)
v = torch.ones_like(F_y)                            # direction = all‐ones
y_vec.grad = None                                     # clear any old grad
F_y.backward(v)                                        # invokes ConstraintFunction.backward
Jv = y_vec.grad.clone()                              # now holds J_F(y)[v]

h_values = np.logspace(3, -5, num=30)
errors_fd = []   # forward‐difference remainder
errors_cd = []   # central‐difference error

for h in h_values:
    # forward remainder R(h) = F(y+h) - F(y) - h Jv
    F_ph = cl(y_vec + h*v)
    R_h = F_ph - F_y - h*Jv
    R_h = R_h/h
    errors_fd.append(R_h.norm().item())

    # central‐difference test D_c(h) = [F(y+h)-F(y-h)]/(2h)
    F_mh = cl(y_vec - h*v)
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
