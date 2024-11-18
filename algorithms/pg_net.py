import torch
import torch.nn as nn
import torch.nn.functional as F


class PGNet(nn.Module):
    def __init__(self, device: torch.device, input_dim: int, output_dim: int, hidden_dim: int = 128):
        """初始化q网络，为全连接网络
        input_dim: 输入的特征数即环境的状态维度
        output_dim: 输出的动作维度
        """
        super(PGNet, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(device)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, output_dim).to(device)  # 输出层

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
