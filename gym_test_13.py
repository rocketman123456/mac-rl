from __future__ import annotations

import random
import numpy as np
import torch
from tqdm import tqdm

from framework.agent.reinforce import REINFORCE
from framework.env.vectored_gym_env import VectoredGymEnv
from framework.config.vectored_gym_cfg import VectoredGymCfg

if __name__ == "__main__":
    # plt.rcParams["figure.figsize"] = (10, 5)
    # Create and wrap the environment
    name = "InvertedPendulum-v5"
    # name = "LunarLander-v3"
    cfg = VectoredGymCfg(name, 4, True)
    env = VectoredGymEnv(cfg)

    use_cuda = False
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("mps" if torch.mps.is_available() else "cpu")

    total_num_episodes = int(5000)  # Total number of episodes
    total_num_sub_steps = int(10000)
    # Observation-space of InvertedPendulum-v4 (4)
    obs_space_dims = env.obs_space_dims
    # Action-space of InvertedPendulum-v4 (1)
    action_space_dims = env.action_space_dims

    agent = REINFORCE(device, obs_space_dims, action_space_dims)

    torch.manual_seed(env.seed)
    random.seed(env.seed)
    np.random.seed(env.seed)

    # Reinitialize agent every seed
    reward_over_episodes = []

    for episode in tqdm(range(total_num_episodes)):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = env.reset()

        done = False
        # while not done:
        for step in range(total_num_sub_steps):
            action = agent.sample_action(obs)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = env.step(action)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated.any() or truncated.any()
            if done:
                obs, info = env.reset()

        reward_over_episodes.append(env.get_reward())
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(env.wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    env.close()
