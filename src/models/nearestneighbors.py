import torch

from torch import Tensor
from typing import List


class kNN:
    def __init__(self, action_space: Tensor, k: int) -> None:
        self.action_space = action_space
        self.k = k

    def find_action(self, action: Tensor) -> Tensor:
        dist = torch.norm(self.action_space - action, dim=1, p=None)
        d, idx = dist.topk(self.k, largest=False)
        return self.action_space[idx]