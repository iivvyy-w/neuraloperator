import torch


def boundary_L2error(y, out):
    """
    ------
    Parameters:
        y: ground truth, tensor shape [batch, out_channels, height, width]
        out: predicted result, tensor shape [batch, out_channels, height, width]
    """

    boundary_mask = torch.zeros_like(y, dtype=torch.bool)
    boundary_mask[:, :, 0, :] = True    # Top row
    boundary_mask[:, :, -1, :] = True   # Bottom row
    boundary_mask[:, :, :, 0] = True    # Left column
    boundary_mask[:, :, :, -1] = True   # Right column

    boundary_y = y[boundary_mask]
    boundary_out = out[boundary_mask]

    return torch.linalg.norm(boundary_out-boundary_y) / (torch.linalg.norm(boundary_y)+1e-8)


def average_error(test_samples, data_processor, model, x_0 = False, y_0=False):
    error = 0
    N = len(test_samples)
    for index in range(N):
        data = test_samples[index]
        data = data_processor.preprocess(data, batched=False)

        x = data['x']
        if x_0:
            x[:, 0, :] = 0
            x[:, -1, :] = 0
            x[:, :, 0] = 0
            x[:, :, -1] = 0
        y = data['y']
        out = model(x.unsqueeze(0))
        if y_0:
            y[:, :, 0, :] = 0
            y[:, :, -1, :] = 0
            y[:, :, :, 0] = 0
            y[:, :, :, -1] = 0
        error += boundary_L2error(y, out)
    print('The relative L-2 error is', error/N)
    return error/N

