from framework.env.basic_env import BasicEnv
from framework.config.vectored_gym_cfg import VectoredGymCfg
import gymnasium as gym


class VectoredGymEnv(BasicEnv):

    def __init__(self, cfg: VectoredGymCfg):
        super().__init__()
        self.name = cfg.name
        self.render = cfg.render
        self.num_envs = cfg.num_envs
        self.seed = 42
        self.cfg = cfg

        # create env
        self.envs = gym.vector.AsyncVectorEnv([self.make_env(self.name, self.seed, self.render, i) for i in range(self.num_envs)])
        self.wrapped_env = gym.wrappers.vector.RecordEpisodeStatistics(self.envs, 50)  # Records episode-reward

        self.obs_space_dims = self.envs.single_observation_space.shape[0]
        self.action_space_dims = self.envs.single_action_space.shape[0]

    def step(self, action):
        return self.wrapped_env.step(action)

    def reset(self):
        return self.wrapped_env.reset(seed=self.seed)

    def get_reward(self):
        return self.wrapped_env.return_queue[-1]

    def close(self):
        self.envs.close()

    def make_env(self, env_id, seed, render, idx):
        def _init_():
            if idx == 0 and render:
                env = gym.make(env_id, render_mode="human")
            else:
                env = gym.make(env_id)
            env.reset(seed=seed + idx)  # Set seed for reproducibility
            return env

        return _init_
