# import gym
# import numpy as np


# env = gym.make("LunarLanderContinuous-v2")

# obs = env.reset()
# print(f"Shape of observation is {obs.shape}")

# action = env.action_space.sample()
# print(f"Action is {action}")

# obs, rew, done, _ = env.step(np.array([0.7, 0.3]))
# print(f"Reward is {rew}")
# print(f"Done is {done}")
# print(f"Info is {_}")
import torch

data = torch.tensor([[1, 2], [2, 2], [3, 2], [4, 3], [1, 1]]).float()
print(data.shape)
print(torch.tensor([[1, 1]]).shape)
dist = torch.norm(data - torch.tensor([[1, 1]]).float(), dim=1, p=None)
knn = dist.topk(3, largest=False)

print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))
print(f"{data[knn.indices].shape}")