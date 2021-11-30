import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class Actor(nn.Module):
    def __init__(self, n_obs: int, n_actions: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256)
        self.fc2 = nn.Linear(256, n_actions)
        self.activation = nn.Softsign()

    def forward(self, obs: Tensor) -> Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.activation(x)