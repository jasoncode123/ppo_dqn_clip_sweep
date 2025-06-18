import gym
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
