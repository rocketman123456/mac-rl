from framework.env.basic_env import BasicEnv
from framework.config.vectored_mujoco_cfg import VectoredMujocoCfg
import gymnasium as gym
import numpy as np


class VectoredMujocoEnv(BasicEnv):

    def __init__(self, cfg: VectoredMujocoCfg):
        super().__init__()
        self.name = cfg.name
        self.render = cfg.render
        self.num_envs = cfg.num_envs
        self.seed = 42
        self.cfg = cfg

        self.default_pos = [0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8]
        self.default_pos = np.array(self.default_pos)

        # create env
        self.envs = gym.vector.AsyncVectorEnv([self.make_env(self.name, self.seed, self.render, i) for i in range(self.num_envs)])
        self.wrapped_env = gym.wrappers.vector.RecordEpisodeStatistics(self.envs, 50)  # Records episode-reward

        self.obs_space_dims = self.envs.single_observation_space.shape[0]
        self.action_space_dims = self.envs.single_action_space.shape[0]

    def step(self, action):
        # action = np.zeros_like(action) + self.default_pos
        action = action + self.default_pos
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
                env = gym.make(
                    env_id,
                    render_mode="human",
                    xml_file="./mujoco_menagerie/unitree_go1/scene.xml",
                    forward_reward_weight=1,
                    ctrl_cost_weight=0.05,
                    contact_cost_weight=5e-4,
                    healthy_reward=1,
                    main_body=1,
                    healthy_z_range=(0.195, 0.75),
                    include_cfrc_ext_in_observation=True,
                    exclude_current_positions_from_observation=False,
                    reset_noise_scale=0.1,
                    frame_skip=25,
                    max_episode_steps=1000,
                )
            else:
                env = gym.make(
                    env_id,
                    xml_file="./mujoco_menagerie/unitree_go1/scene.xml",
                    forward_reward_weight=1,
                    ctrl_cost_weight=0.05,
                    contact_cost_weight=5e-4,
                    healthy_reward=1,
                    main_body=1,
                    healthy_z_range=(0.195, 0.75),
                    include_cfrc_ext_in_observation=True,
                    exclude_current_positions_from_observation=False,
                    reset_noise_scale=0.1,
                    frame_skip=25,
                    max_episode_steps=1000,
                )
            env.reset(seed=seed + idx)  # Set seed for reproducibility
            return env

        return _init_
