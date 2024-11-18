from framework.config.basic_cfg import BasicCfg
import torch


class VectoredGymCfg(BasicCfg):

    def __init__(self, name: str, num_envs: int, render: bool):
        super().__init__()
        self.num_envs = 10
        self.num_obs = 4
        self.num_action = 1

        self.name = name
        self.num_envs = num_envs
        self.render = render

        self.use_cuda = False
        if self.use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
