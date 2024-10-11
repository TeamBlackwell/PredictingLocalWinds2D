import torch
import torch.nn as nn


class BasicMLP(nn.Module):
    def __init__(self, local, device):
        super(BasicMLP, self).__init__()

        self.input_size = (
            360 + 2
        )  # 360-dimensional vector + 2 for wind at robot's location
        self.output_size = (
            ((local * 2) + 1) ** 2
        ) * 2  # Adjusted for the additional dimension

        self.fc1 = nn.Linear(self.input_size, 2048, dtype=torch.float32, device=device)
        self.fc2 = nn.Linear(2048, 2048, dtype=torch.float32, device=device)
        self.fc3 = nn.Linear(2048, 2048, dtype=torch.float32, device=device)
        self.fc4 = nn.Linear(2048, self.output_size, dtype=torch.float32, device=device)
        self.relu = nn.ReLU()
        self.local = local

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = x.view(
            -1, self.local * 2 + 1, self.local * 2 + 1, 2
        )  # Reshape to the desired output size
        return x
