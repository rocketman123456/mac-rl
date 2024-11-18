from __future__ import annotations

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
import gymnasium as gym

from agent.reinforce import REINFORCE

plt.rcParams["figure.figsize"] = (10, 5)

if __name__ == "__main__":
    # Create and wrap the environment
    # env = gym.make(
    #     "InvertedPendulum-v5",
    #     # render_mode="human",
    # )
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: gym.make(
                "InvertedPendulum-v5",
                # render_mode="human",
            )
            for i in range(4)
        ]
    )

    # wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
    wrapped_env = gym.wrappers.vector.RecordEpisodeStatistics(envs, 50)  # Records episode-reward

    use_cuda = False
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("mps" if torch.mps.is_available() else "cpu")

    total_num_episodes = int(5e3)  # Total number of episodes
    # Observation-space of InvertedPendulum-v4 (4)
    # obs_space_dims = env.observation_space.shape[0]
    obs_space_dims = envs.single_observation_space.shape[0]
    # Action-space of InvertedPendulum-v4 (1)
    # action_space_dims = env.action_space.shape[0]
    action_space_dims = envs.single_action_space.shape[0]
    rewards_over_seeds = []

    agent = REINFORCE(device, obs_space_dims, action_space_dims)

    for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Reinitialize agent every seed
        reward_over_episodes = []

        for episode in tqdm(range(total_num_episodes)):
            # gymnasium v26 requires users to set seed while resetting the environment
            obs, info = wrapped_env.reset(seed=seed)

            done = False
            while not done:
                action = agent.sample_action(obs)

                # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
                # These represent the next observation, the reward from the step,
                # if the episode is terminated, if the episode is truncated and
                # additional info from the step
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                agent.rewards.append(reward)

                # End the episode when either truncated or terminated is true
                #  - truncated: The episode duration reaches max number of timesteps
                #  - terminated: Any of the state space values is no longer finite.
                done = terminated.any() or truncated.any()

            reward_over_episodes.append(wrapped_env.return_queue[-1])
            agent.update()

            if episode % 1000 == 0:
                avg_reward = int(np.mean(wrapped_env.return_queue))
                print("Episode:", episode, "Average Reward:", avg_reward)

        rewards_over_seeds.append(reward_over_episodes)

    # plot result
    rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(title="REINFORCE for InvertedPendulum-v4")
    plt.show()
