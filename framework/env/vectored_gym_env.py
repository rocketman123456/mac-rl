from framework.env.basic_env import BasicEnv
import gymnasium as gym


def make_env(env_id, seed, idx, render):
    def _init_():
        if idx == 0 and render:
            env = gym.make(env_id, render_mode="human")
        else:
            env = gym.make(env_id)
        env.reset(seed=seed + idx)  # Set seed for reproducibility
        return env

    return _init_


class VectoredGymEnv(BasicEnv):

    def __init__(self, name: str, num_envs: int, render: bool):
        super().__init__()
        self.name = name
        self.render = render
        self.num_envs = num_envs
        self.seed = 42
        self.envs = gym.vector.AsyncVectorEnv([make_env(name, self.seed, i, render) for i in range(num_envs)])

    def step(self):
        pass

    def reset(self):
        pass
