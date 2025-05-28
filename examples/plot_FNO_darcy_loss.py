import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss


device_name = "cpu"
#"""
if torch.cuda.is_available():
    device_name = "cuda:0"
#"""
device = torch.device(device_name)

from neuralop.models.plot_function import plot_example_result

## Create a folder every time saving the figures
import os
import datetime
import numpy as np

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = f"FNO_DarcyFlowLoss_{timestamp}"
os.makedirs(folder_name, exist_ok=True)

# %%
# Let's load the small Darcy-flow dataset. 
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32, 
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
)
data_processor = data_processor.to(device)


# %%
# We create a simple FNO model

# Record the parameters used to alter the parameters in the following training and testing
"""
constraint = False  # whether to apply constraint on the FNO
constraint_type = 'zero'  # what kind of constraint to apply
constraint_which = None  # on which side of the boundary to apply constraint
constraint_g = None  # What is g(x) on the neumann problem
x_32 = False  # whether to apply the zero boundary on given data for testing
x_16 = False
y_32 = False   # whether to set the boundary of ground truth to zero
y_16 = False

#def g_linear(a, b):
#   return torch.e**(-a*b)
"""

model_unconstraint = FNO(n_modes=(16, 16),
                         in_channels=1, 
                         out_channels=1,
                         hidden_channels=32, 
                         projection_channel_ratio=2,
                         constraint=False)
model_unconstraint = model_unconstraint.to(device)

model_soft = FNO(n_modes=(16, 16),
                 in_channels=1,
                 out_channels=1,
                 hidden_channels=32,
                 projection_channel_ratio=2,
                 constraint=False)
model_soft = model_soft.to(device)

model_constraint = FNO(n_modes=(16, 16),
                       in_channels=1, 
                       out_channels=1,
                       hidden_channels=32, 
                       projection_channel_ratio=2,
                       constraint=True,
                       constraint_type='zero')
model_constraint = model_constraint.to(device)


def training(model, lr=8e-3, n_epochs=20, mask=False, soft=False, soft_weight=0.1):
    n_params = count_model_params(model)
    print(f'\nOur model has {n_params} parameters.')
    sys.stdout.flush()

    optimizer = AdamW(model.parameters(),
                      lr=lr,
                      weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100) #changed
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=n_epochs)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=200)

    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2, mask=mask, soft=soft, soft_weight=soft_weight)

    train_loss = h1loss
    eval_losses = {'h1': h1loss, 'l2': l2loss}

    print('\n### MODEL ###\n', model)
    print('\n### OPTIMIZER ###\n', optimizer)
    print('\n### SCHEDULER ###\n', scheduler)
    print('\n### LOSSES ###')
    print(f'\n * Train: {train_loss}')
    print(f'\n * Test: {eval_losses}')
    sys.stdout.flush()

    trainer = Trainer(model=model, n_epochs=n_epochs,
                      device=device,
                      data_processor=data_processor,
                      wandb_log=False,
                      eval_interval=3,
                      use_distributed=False,
                      verbose=True)

    epoch_metrics, train_errs = trainer.train(train_loader=train_loader,
                                              test_loaders=test_loaders,
                                              optimizer=optimizer,
                                              scheduler=scheduler, 
                                              regularizer=False, 
                                              training_loss=train_loss,
                                              eval_losses=eval_losses)

    return train_errs


train_errs_un = training(model_unconstraint)
#train_errs_con = training(model_constraint)
#train_errs_soft = training(model_soft, soft=True)

error_16_un, total_error_16_un = plot_example_result(test_loaders, 16, data_processor, model_unconstraint, folder_name)
error_32_un, total_error_32_un = plot_example_result(test_loaders, 32, data_processor, model_unconstraint, folder_name)
#error_64_un, total_error_64_un = plot_example_result(test_loaders, 64, data_processor, model_unconstraint, folder_name)
#error_128_un, total_error_128_un = plot_example_result(test_loaders, 128, data_processor, model_unconstraint, folder_name)
#error_16_con, total_error_16_con = plot_example_result(test_loaders, 16, data_processor, model_constraint, folder_name)
#error_32_con, total_error_32_con = plot_example_result(test_loaders, 32, data_processor, model_constraint, folder_name)
#error_64_con, total_error_64_con = plot_example_result(test_loaders, 64, data_processor, model_constraint, folder_name)
#error_128_con, total_error_128_con = plot_example_result(test_loaders, 128, data_processor, model_constraint, folder_name)

#error_16_s, total_error_16_s = plot_example_result(test_loaders, 16, data_processor, model_soft, folder_name)
#error_32_s, total_error_32_s = plot_example_result(test_loaders, 32, data_processor, model_soft, folder_name)

fig, ax = plt.subplots()
ax.plot(train_errs_un, label="FNO")
#ax.plot(train_errs_con, label="FNO-CON")
#ax.plot(train_errs_soft, label="FNO-soft")
ax.set_yscale('log')
ax.set_xlabel("Epoch")
ax.set_ylabel("Relative L2 Error")
ax.set_title("Learning Curve: Relative Error vs. Epochs")
ax.legend()
ax.grid()
fig.savefig(os.path.join(folder_name, "Learning Curve.png"))
fig.show()
