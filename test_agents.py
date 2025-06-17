import gym
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent

env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

ppo = PPOAgent(obs_dim, act_dim)
dqn = DQNAgent(obs_dim, act_dim)
print("PPOAgent:", ppo)
print("DQNAgent:", dqn)
