import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

import os
import datetime
import numpy as np
from neuralop.models.errors import average_error

device_name = "cpu"
if torch.cuda.is_available():
    device_name = "cuda:0"
device = torch.device(device_name)


def plot_example_result(test_loaders, resolution, data_processor, model, folder_name):
    """
    input:
    test_loaders: test_loader imported at the beginning
    resolution: 16 / 32
    data_processor:
    model: model_unconstraint/ model_constraint

    """
    test_samples = test_loaders[resolution].dataset

    fig = plt.figure(figsize=(7, 7))
    for index in range(3):
        data = test_samples[index]
        data = data_processor.preprocess(data, batched=False)
        # Input x
        x = data['x']
        # Ground-truth
        y = data['y']
        # Model prediction
        out = model(x.unsqueeze(0), data_processor=data_processor)
        if x.device.type != 'cpu':
            x = x.to('cpu')
            y = y.to('cpu')
            out = out.to('cpu')

        ax = fig.add_subplot(3, 3, index*3 + 1)
        ax.imshow(x[0], cmap='gray')
        if index == 0: 
            ax.set_title('Input x')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(3, 3, index*3 + 2)
        ax.imshow(y.squeeze())
        if index == 0: 
            ax.set_title('Ground-truth y')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(3, 3, index*3 + 3)
        ax.imshow(out.squeeze().detach().numpy())
        if index == 0: 
            ax.set_title('Model prediction')
        plt.xticks([], [])
        plt.yticks([], [])
    
    fig.suptitle(f'Inputs, ground-truth output and prediction ({resolution}x{resolution}).', y=0.98)
    plt.tight_layout()
    fig.show()
    fig.savefig(os.path.join(folder_name, f"{model.constraint}{resolution}.png"))

    error, total_error = average_error(test_samples, data_processor, model, y_0=model.constraint)
    return error, total_error