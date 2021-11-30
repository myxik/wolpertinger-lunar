import gym
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from src.models.wolpertinger import Wolpertinger
from src.experience_replay import ExperienceReplay


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = SummaryWriter()

    env = gym.make("LunarLanderContinuous-v2")
    done = False

    exp_rep = ExperienceReplay(10000)
    batch_size = 128
    action_space = torch.tensor(np.random.uniform(-1, 1, size=(1000, 2))).to(device)

    model = Wolpertinger(action_space, 8, 1000, 5, 0.99, device, logger, 0.2)

    for episode in range(5):
        obs = env.reset()
        done = False

        step_num = 0
        ep_rews = []
        while not done:
            prev_obs = obs
            action = model.select_action(obs)
            obs, rew, done, _ = env.step(action)

            exp_rep.push((prev_obs, action, rew, obs))

            if len(exp_rep) > batch_size:
                trainsample = exp_rep.sample(batch_size)
                model.optimize_policy(trainsample, step_num, episode)
            
            step_num += 1
            ep_rews.append(rew)
            logger.add_scalar("Reward/timestep", rew, step_num)
        logger.add_scalar("Reward/episode", sum(ep_rews), episode)
        print(f"Reward is {sum(ep_rews)}")


if __name__=="__main__":
    run()