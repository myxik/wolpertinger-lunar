import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class Critic(nn.Module):
    def __init__(self, n_obs: int, act_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256)
        self.fc2 = nn.Linear(256+act_dim, 1)

    def forward(self, obs: Tensor, actions: Tensor) -> Tensor:
        x = F.relu(self.fc1(obs))
        x = torch.cat([x, actions.float()], 1)
        x = self.fc2(x)
        return x