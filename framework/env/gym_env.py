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

    def step(self):
        pass

    def reset(self):
        pass
