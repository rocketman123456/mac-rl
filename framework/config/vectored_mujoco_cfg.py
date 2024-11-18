from framework.config.basic_cfg import BasicCfg
import torch


class VectoredMujocoCfg(BasicCfg):

    def __init__(self, name: str, xml_path: str, num_envs: int, render: bool):
        super().__init__()
        self.num_envs = 10
        self.num_obs = 4
        self.num_action = 1

        self.name = name
        self.num_envs = num_envs
        self.render = render
        self.xml_path = xml_path

        self.use_cuda = False
        if self.use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
