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
    def __init__(self, obs_dim, act_dim,
                 lr=1e-4, gamma=0.99,
                 target_update_freq=500):
        # 在线网络 & 目标网络
        self.online_net = DQNNet(obs_dim, act_dim)
        self.target_net = DQNNet(obs_dim, act_dim)
        # 同步参数
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.gamma = gamma

        self.update_count = 0
        self.target_update_freq = target_update_freq

    def select_action(self, obs, epsilon):
        # 拆包新版 Gym API
        if isinstance(obs, (tuple, list)):
            obs = obs[0]
        # to Tensor + batch dim
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        if obs_t.dim()==1:
            obs_t = obs_t.unsqueeze(0)
        # forward
        q = self.online_net(obs_t).squeeze(0)
        if random.random() < epsilon:
            return random.randrange(q.shape[0])
        else:
            return torch.argmax(q).item()

    def update(self, obs, acts, rewards, next_obs, dones):
        """
        Double DQN + Target Network
        obs, next_obs: list of np.array, length N
        acts, rewards, dones: list of scalars, length N
        """
        # 1) 转 Tensor
        obs_t      = torch.as_tensor(obs,      dtype=torch.float32)
        acts_t     = torch.as_tensor(acts,     dtype=torch.int64)
        rews_t     = torch.as_tensor(rewards,  dtype=torch.float32)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32)
        dones_t    = torch.as_tensor(dones,    dtype=torch.float32)

        # 2) Q(s,a) from online
        q_values = self.online_net(obs_t)  # [N, A]
        q_s_a    = q_values.gather(1, acts_t.unsqueeze(-1)).squeeze(-1)

        # 3) Double DQN target:
        #    actions from online_net, Q-values from target_net
        with torch.no_grad():
            next_q_online = self.online_net(next_obs_t)           # [N, A]
            best_next_actions = next_q_online.argmax(dim=1)       # [N]
            next_q_target = self.target_net(next_obs_t)           # [N, A]
            q_next = next_q_target.gather(1, best_next_actions.unsqueeze(-1)).squeeze(-1)
            target = rews_t + self.gamma * q_next * (1 - dones_t) # [N]

        # 4) MSE loss
        loss = F.mse_loss(q_s_a, target)

        # 5) backward
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（可选）
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 6) update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {
            "q_loss":       loss.item(),
            "q_value_mean": q_s_a.mean().item()
        }
