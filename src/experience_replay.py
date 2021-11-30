import torch
import random
import numpy as np

from torch import Tensor
from collections import deque
from typing import List, Any, Tuple


class ExperienceReplay(object):
    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, experience: List[Any]) -> None:
        self.memory.append(experience)

    def sample(self, batch_size: int) -> Tuple[Tensor]:
        data_sample = random.sample(self.memory, batch_size)

        prev_states, next_states, actions = [], [], []
        rewards = torch.zeros(batch_size, )
        for i, (prev_state, action, reward, next_state) in enumerate(data_sample):
            prev_states.append(torch.tensor(prev_state))
            next_states.append(torch.tensor(next_state))
            actions.append(torch.tensor(action))
            rewards[i] = reward

        prev_states = torch.stack(prev_states)
        next_states = torch.stack(next_states)
        actions = torch.stack(actions)
        return (prev_states, actions, rewards, next_states)

    def __len__(self) -> int:
        return len(self.memory)