# train/ppo.py
import gym
import numpy as np
import torch
from agents.ppo_agent import PPOAgent
from utils.rl_utils import compute_gae

def train_ppo(env_name="CartPole-v1",
              total_epochs=50,
              steps_per_epoch=4000,
              gamma=0.99,
              lam=0.95,
              clip_ratio=0.2,
              lr=3e-4,
              train_iters=80,
              batch_size=64):
    # 创建环境和 Agent
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = PPOAgent(obs_dim, act_dim,
                     clip_ratio=clip_ratio, lr=lr,
                     gamma=gamma, lam=lam)

    ep_returns = []
    ep_lens    = []

    for epoch in range(1, total_epochs+1):
        # 准备 buffers
        obs_buf, act_buf, rew_buf = [], [], []
        val_buf, logp_buf, done_buf = [], [], []

        o, done, ep_ret, ep_len = env.reset(), False, 0, 0
        # 收集一批数据
        for t in range(steps_per_epoch):
            a, logp, v = agent.select_action(o)
            # 兼容 Gym API
            o_val = o[0] if isinstance(o, tuple) else o

            obs_buf.append(o_val)
            act_buf.append(a)
            val_buf.append(v)
            logp_buf.append(logp)
            done_buf.append(done)

            o2, r, d, *_ = env.step(a)
            rew_buf.append(r)
            ep_ret += r
            ep_len += 1

            o, done = o2, d
            # episode 结束或到达 horizon
            if d or (t == steps_per_epoch-1):
                ep_returns.append(ep_ret)
                ep_lens.append(ep_len)
                # 计算末状态 value
                if done:
                    last_val = 0
                else:
                    o2_val = o2[0] if isinstance(o2, tuple) else o2
                    last_val = agent.value_net(
                        torch.tensor(o2_val, dtype=torch.float32).unsqueeze(0)
                    ).item()
                val_buf.append(last_val)
                # 计算 GAE advantages 和 returns
                advs, rets = compute_gae(rew_buf, val_buf[:-1], done_buf,
                                         gamma=gamma, lam=lam)
                # 重置 episode
                o, done, ep_ret, ep_len = env.reset(), False, 0, 0

        # 转成 numpy array
        obs_arr  = np.array(obs_buf)
        act_arr  = np.array(act_buf)
        logp_arr = np.array(logp_buf)
        advs_arr = np.array(advs); rets_arr = np.array(rets)
        # 标准化优势
        advs_arr = (advs_arr - advs_arr.mean()) / (advs_arr.std() + 1e-8)

        # 多轮更新
        for _ in range(train_iters):
            idx = np.random.randint(0, len(obs_arr), size=batch_size)
            metrics = agent.update(obs_arr[idx],
                                   act_arr[idx],
                                   logp_arr[idx],
                                   advs_arr[idx],
                                   rets_arr[idx])

        # 打印日志
        print(f"Epoch {epoch:3d} | "
              f"AvgRet {np.mean(ep_returns[-10:]):.2f} | "
              f"AvgLen {np.mean(ep_lens[-10:]):.2f} | "
              f"PolL {metrics['policy_loss']:.3f} | "
              f"ValL {metrics['value_loss']:.3f} | "
              f"Ent {metrics['entropy']:.3f}")

    return agent

if __name__ == "__main__":
    train_ppo()
