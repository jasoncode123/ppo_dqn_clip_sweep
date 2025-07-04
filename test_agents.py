import gym
import numpy as np
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent

env = gym.make("CartPole-v1")
obs ,_= env.reset()

# 测试 PPOAgent
ppo = PPOAgent(env.observation_space.shape[0], env.action_space.n)
a, lp, v = ppo.select_action(obs)
print("PPO 采样结果 → action:", a, "log_prob:", lp, "value:", v)

# 测试 DQNAgent 的 select_action（可选）
dqn = DQNAgent(env.observation_space.shape[0], env.action_space.n)
# 这里我们暂时用 epsilon=0，纯 greedy
# 原来是
# action_dqn = dqn.select_action(obs, epsilon=0.0)
# print("DQN greedy action:", action_dqn)

# 改成下面，方便看随机和贪心行为
action_dqn_greedy = dqn.select_action(obs, epsilon=0.0)
action_dqn_random = dqn.select_action(obs, epsilon=1.0)
print("DQN greedy action:", action_dqn_greedy)
print("DQN random action:", action_dqn_random)

print("\n=== PPO Update Test ===")
N = 8
obs_b = np.random.randn(N, env.observation_space.shape[0])
acts_b = np.random.randint(0, env.action_space.n, size=N)
oldlps = np.random.randn(N) * -1.0
advs   = np.random.randn(N)
rets   = np.random.randn(N)
metrics = ppo.update(obs_b, acts_b, oldlps, advs, rets)
print(metrics)


# ------------------ DQN 更新测试 ------------------
print("\n=== DQN Update Test ===")
N = 8
obs_b      = np.random.randn(N, env.observation_space.shape[0])
acts_b     = np.random.randint(0, env.action_space.n, size=N)
rews_b     = np.random.randn(N)
next_obs_b = np.random.randn(N, env.observation_space.shape[0])
dones_b    = np.random.randint(0, 2, size=N)

dqn_metrics = dqn.update(obs_b, acts_b, rews_b, next_obs_b, dones_b)
print(dqn_metrics)
