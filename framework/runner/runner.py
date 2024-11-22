import numpy as np
import torch
import random
from tqdm import tqdm


class PolicyRunner:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

        self.total_num_episodes = int(10000)

        # set random seed
        torch.manual_seed(env.seed)
        random.seed(env.seed)
        np.random.seed(env.seed)

    def learn(self):
        # Reinitialize agent every seed
        reward_over_episodes = []

        for episode in tqdm(range(self.total_num_episodes)):
            # gymnasium v26 requires users to set seed while resetting the environment
            obs, info = self.env.reset()

            done = False
            while not done:
                action = self.agent.sample_action(obs)

                # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
                # These represent the next observation, the reward from the step,
                # if the episode is terminated, if the episode is truncated and
                # additional info from the step
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.agent.rewards.append(reward)

                # End the episode when either truncated or terminated is true
                #  - truncated: The episode duration reaches max number of timesteps
                #  - terminated: Any of the state space values is no longer finite.
                done = terminated.any() or truncated.any()

            reward_over_episodes.append(self.env.get_reward())
            self.agent.update()

            if episode % 1000 == 0:
                avg_reward = int(np.mean(self.env.wrapped_env.return_queue))
                print("Episode:", episode, "Average Reward:", avg_reward)

        self.env.close()

    def save(self):
        pass

    def load(self):
        pass
