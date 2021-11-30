import torch
import random
import numpy as np
import torch.nn as nn

from src.models.actor import Actor
from src.models.critic import Critic
from src.models.nearestneighbors import kNN


class Wolpertinger:
    def __init__(self, action_space, n_obs, n_actions, n_actions_contracted, gamma, device, logger, tau):
        self.knn = kNN(action_space, n_actions_contracted)
        self.actor = Actor(n_obs, 2)
        self.critic = Critic(n_obs, 2)

        self.actor_target = Actor(n_obs, 2)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target = Critic(n_obs, 2)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor.to(device)
        self.critic.to(device)
        self.actor_target.to(device)
        self.critic_target.to(device)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.criterion = nn.MSELoss()

        self.device = device
        self.logger = logger

        self.gamma = gamma
        self.n_actions_contracted = n_actions_contracted
        self.n_actions = n_actions
        self.tau = tau

    def optimize_policy(self, trainsample, step_num, episode):
        prev_states, actions, rewards, next_states = trainsample
        prev_states = prev_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        
        y_i = rewards.unsqueeze(1) + self.gamma * self.critic_target(next_states, self.wolp_action_target(next_states))
        Q_critic = self.critic(prev_states, actions)

        self.critic_optimizer.zero_grad()
        critic_loss = self.criterion(Q_critic, y_i)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss = self.critic(prev_states, actions).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.logger.add_scalar(f"Loss/{episode}", critic_loss.item(), step_num)
        self.logger.add_scalar(f"Q_i/{episode}", actor_loss.item(), step_num)

        self.update_targets()

    def update_targets(self):
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)
       
    def soft_update(self, target, source, tau_update):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau_update) + param.data * tau_update
            )

    def wolp_action_target(self, next_states):
        Q_i = torch.zeros(next_states.shape[0], 2)
        for i, next_state in enumerate(next_states):
            Q_i[i] = torch.tensor(self.select_action(next_state)).unsqueeze(0)
        return Q_i.to(self.device)

    def select_action(self, obs):
        obs = torch.tensor(obs).unsqueeze(0).to(self.device)
        protoactions = self.actor(obs)

        actions = self.knn.find_action(protoactions)
        Q_i = torch.zeros(actions.shape[0], )
        for i, act in enumerate(actions):
            Q_i[i] = self.critic(obs, act.unsqueeze(0))
        return actions[torch.argmax(Q_i, 0).item()].cpu().numpy()