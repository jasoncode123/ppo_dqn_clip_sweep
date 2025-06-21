# train/dqn.py
import gym
import numpy as np
import random
from utils.replay_buffer import ReplayBuffer
from agents.dqn_agent import DQNAgent

def train_dqn(env_name="CartPole-v1",
              total_epochs=50,
              steps_per_epoch=4000,
              gamma=0.99,
              lr=1e-3,
              replay_size=10000,
              batch_size=64,
              eps_start=1.0,
              eps_end=0.1):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = DQNAgent(obs_dim, act_dim, lr=lr, gamma=gamma)
    buffer = ReplayBuffer(replay_size)

    ep_returns = []
    ep_lens    = []
    global_step = 0

    for epoch in range(1, total_epochs+1):
        o, done, ep_ret, ep_len = env.reset(), False, 0, 0

        for t in range(steps_per_epoch):
            # 线性衰减 ε
            frac = min(1.0, global_step / (total_epochs * steps_per_epoch))
            eps  = eps_start + frac * (eps_end - eps_start)

            a = agent.select_action(o, epsilon=eps)
            o2, r, d, *_ = env.step(a)
            o_val  = o[0] if isinstance(o, tuple) else o
            o2_val = o2[0] if isinstance(o2, tuple) else o2

            buffer.push((o_val, a, r, o2_val, d))

            o, ep_ret, ep_len, done = o2, ep_ret + r, ep_len + 1, d
            global_step += 1

            # 只有当缓冲区足够大时才开始更新
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                obs_b, act_b, rew_b, next_b, done_b = zip(*batch)
                metrics = agent.update(obs_b, act_b, rew_b, next_b, done_b)

            if done:
                ep_returns.append(ep_ret)
                ep_lens.append(ep_len)
                o, done, ep_ret, ep_len = env.reset(), False, 0, 0

        avg_ret = np.mean(ep_returns[-10:]) if ep_returns else 0
        avg_len = np.mean(ep_lens[-10:]) if ep_lens else 0

        print(f"Epoch {epoch:3d} | "
              f"AvgRet {avg_ret:.2f} | "
              f"AvgLen {avg_len:.2f} | "
              f"QLoss {metrics['q_loss']:.3f} | "
              f"QVal {metrics['q_value_mean']:.3f} | "
              f"ε {eps:.3f}")

    return agent

if __name__ == "__main__":
    train_dqn()
