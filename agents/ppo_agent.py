import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # 输出 logits

class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)

class PPOAgent:
    def __init__(self, obs_dim, act_dim, clip_ratio=0.2, lr=3e-4, gamma=0.99, lam=0.95):
        self.policy_net = PolicyNet(obs_dim, act_dim)
        self.value_net  = ValueNet(obs_dim)
        self.clip_ratio = clip_ratio
        self.gamma      = gamma
        self.lam        = lam
        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optim  = torch.optim.Adam(self.value_net.parameters(),  lr=lr)

    def select_action(self, obs):
        """
        返回 action, log_prob, value（用于后续 GAE）
        """
        # TODO
        pass

    def compute_gae(self, rewards, values, dones):
        """
        输入 rewards, values, dones 列表，计算 GAE advantages 和 returns
        """
        # TODO
        pass

    def update(self, obs, acts, logps, advs, rets):
        """
        使用一个 batch 的数据做一次 PPO clipping 更新
        """
        # TODO
        pass
