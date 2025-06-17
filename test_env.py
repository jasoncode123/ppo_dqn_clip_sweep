import gym
import torch

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs = env.reset()
    print("CartPole-v1 reset obs:", obs)
    net = torch.nn.Linear(env.observation_space.shape[0], env.action_space.n)
    print("Policy net created:", net)
