import torch
import random
import torch.nn as nn
import torch.nn.functional as F

class DQNNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, obs_dim, act_dim, lr=1e-3, gamma=0.99):
        self.net = DQNNet(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
    
    def select_action(self, obs, epsilon):
        """
        输入：
          obs: numpy array 或 Tensor，表示当前状态
          epsilon: float，探索率，∈ [0,1]
        返回：
          action: int，选中的动作
        """
        # 0. 处理 obs，使其成为 [1, obs_dim] 的 Tensor
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.unsqueeze(0)  # shape [1, obs_dim]
        
        # 1. 用网络预测 Q 值
        q_values = self.net(obs)  # shape [1, act_dim]
        q_values = q_values.squeeze(0)  # shape [act_dim]
        
        # 2. ε-贪心：随机探索或贪心选择
        if random.random() < epsilon:
            action = random.randrange(q_values.shape[0])
        else:
            action = torch.argmax(q_values).item()
        
        return action
    
    def update(self, batch):
        """
        输入一个 batch 的 (obs, act, rew, next_obs, done)，
        计算 DQN loss 并做优化
        """
        # TODO: 实现 DQN 的 update
        pass
