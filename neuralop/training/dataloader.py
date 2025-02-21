import torch


class BoundaryZeroDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        """
        A wrapper around an existing dataset to modify only the boundary of y.
        
        Parameters:
        - original_dataset: torch.utils.data.Dataset (The original dataset)
        """
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x = self.original_dataset[idx]['x']
        y = self.original_dataset[idx]['y']  # Get original (x, y)
        
        # Create a modified version of y with boundary set to 0
        y_modified = y.clone()  # Clone to avoid modifying the original dataset
        y_modified[:, 0, :] = 0
        y_modified[:, -1, :] = 0
        y_modified[:, :, 0] = 0
        y_modified[:, :, -1] = 0

        return {'x': x, 'y': y_modified}  # Return x and modified y
