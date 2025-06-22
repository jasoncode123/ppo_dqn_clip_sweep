import torch
import random
import torch.nn as nn
import torch.nn.functional as F

# 定义 DQN 网络
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
        输入:
          obs: numpy array、Tensor，或 (obs, info) 元组
          epsilon: float, 探索率
        """
        # 1) 兼容 Gym 0.26+ API，把 (obs, info) 拆开
        if isinstance(obs, (tuple, list)):
            obs = obs[0]

        # 2) 转成 Tensor 并加 batch 维度
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [1, obs_dim]

        # 3) 前向 + ε-greedy
        q_values = self.net(obs)           # [1, act_dim]
        q_values = q_values.squeeze(0)     # [act_dim]
        if random.random() < epsilon:
            return random.randrange(q_values.shape[0])
        else:
            return torch.argmax(q_values).item()
        
    def update(self, obs, acts, rewards, next_obs, dones):
        """
        对一个 batch 的 (obs, acts, rewards, next_obs, dones) 做一次 DQN 更新
        - obs, next_obs: numpy arrays, shape [N, obs_dim]
        - acts:           numpy array of ints, shape [N]
        - rewards:        numpy array of floats, shape [N]
        - dones:          numpy array of 0/1,    shape [N]
        返回：
          dict 包含 'q_loss'（float）和 'q_value_mean'（float）
        """
        # 1. 转 Tensor
        obs_t      = torch.as_tensor(obs,      dtype=torch.float32)
        acts_t     = torch.as_tensor(acts,     dtype=torch.int64)
        rews_t     = torch.as_tensor(rewards,  dtype=torch.float32)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32)
        dones_t    = torch.as_tensor(dones,    dtype=torch.float32)

        # 2. 当前 Q(s,a)
        q_values = self.net(obs_t)                              # [N, act_dim]
        q_s_a    = q_values.gather(1, acts_t.unsqueeze(-1)).squeeze(-1)  # [N]

        # 3. 计算 target：r + γ * max_a' Q(s',a') * (1 - done)
        with torch.no_grad():
            q_next    = self.net(next_obs_t)                    # [N, act_dim]
            q_next_max = q_next.max(dim=1).values               # [N]
            target     = rews_t + self.gamma * q_next_max * (1 - dones_t)  # [N]

        # 4. MSE loss
        loss = F.mse_loss(q_s_a, target)

        # 5. 反向 + 更新
        # 5. 反向 + 更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 6. 返回指标
        return {
            "q_loss":       loss.item(),
            "q_value_mean": q_s_a.mean().item()
        }