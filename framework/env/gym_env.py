from framework.env.basic_env import BasicEnv
import gymnasium as gym


class GymEnv(BasicEnv):

    def __init__(self, name: str, render: bool):
        super().__init__()
        self.name = name
        self.render = render
        if render:
            self.env = gym.make(self.name, render_mode="human")
        else:
            self.env = gym.make(self.name)
        self.wrapped_env = gym.wrappers.vector.RecordEpisodeStatistics(self.env, 50)  # Records episode-reward

        self.obs_space_dims = self.env.observation_space.shape[0]
        self.action_space_dims = self.env.action_space.shape[0]

    def step(self, action):
        return self.wrapped_env.step(action)

    def reset(self):
        return self.wrapped_env.reset(seed=self.seed)

    def get_reward(self):
        return self.wrapped_env.return_queue[-1]

    def close(self):
        self.env.close()
