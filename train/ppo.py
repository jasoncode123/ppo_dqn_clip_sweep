# train/ppo.py
import os
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
    # 指标收集
    ret_list, len_list = [], []
    pol_list, val_list, ent_list = [], [], []

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = PPOAgent(obs_dim, act_dim,
                     clip_ratio=clip_ratio, lr=lr,
                     gamma=gamma, lam=lam)

    for epoch in range(1, total_epochs+1):
        # —— 1. 收集一整个 epoch 的数据 ——
        obs_buf, act_buf, rew_buf = [], [], []
        val_buf, logp_buf, done_buf = [], [], []

        o, done, ep_ret, ep_len = env.reset(), False, 0, 0
        ep_returns, ep_lens = [], []

        for t in range(steps_per_epoch):
            a, logp, v = agent.select_action(o)
            o_val = o[0] if isinstance(o, tuple) else o

            obs_buf.append(o_val)
            act_buf.append(a)
            val_buf.append(v)
            logp_buf.append(logp)

            # 兼容新旧 Gym API
            step_ret = env.step(a)
            if len(step_ret) == 5:
                o2, r, term, trunc, _ = step_ret
                done = term or trunc
            else:
                o2, r, done, _ = step_ret

            rew_buf.append(r)
            done_buf.append(done)

            ep_ret += r
            ep_len += 1
            o = o2

            if done:
                ep_returns.append(ep_ret)
                ep_lens.append(ep_len)
                # Reset episode
                o, done, ep_ret, ep_len = env.reset(), False, 0, 0

        # —— 2. 在 epoch 末尾补一个最后 state 的 value ——
        # （如果上一步 done=True，那 value=0；否则正常估计）
        if done:
            last_val = 0.0
        else:
            o_val = o[0] if isinstance(o, tuple) else o
            last_val = agent.value_net(
                torch.tensor(o_val, dtype=torch.float32).unsqueeze(0)
            ).item()
        val_buf.append(last_val)

        # —— 3. 计算全程的 GAE advantages & returns ——
        advs, rets = compute_gae(rew_buf, val_buf, done_buf, gamma, lam)

        # —— 4. 转成 numpy（并标准化 advantages）——
        obs_arr  = np.array(obs_buf)
        act_arr  = np.array(act_buf)
        logp_arr = np.array(logp_buf)
        advs_arr = np.array(advs)
        rets_arr = np.array(rets)
        advs_arr = (advs_arr - advs_arr.mean()) / (advs_arr.std() + 1e-8)

        # —— 5. 多轮 PPO 更新 ——
        for _ in range(train_iters):
            idx = np.random.randint(0, len(obs_arr), size=batch_size)
            metrics = agent.update(obs_arr[idx],
                                   act_arr[idx],
                                   logp_arr[idx],
                                   advs_arr[idx],
                                   rets_arr[idx])

        # —— 6. 记录 & 打印 ——
        avg_ret = np.mean(ep_returns[-10:]) if ep_returns else 0
        avg_len = np.mean(ep_lens[-10:])    if ep_lens else 0

        ret_list.append(avg_ret)
        len_list.append(avg_len)
        pol_list.append(metrics['policy_loss'])
        val_list.append(metrics['value_loss'])
        ent_list.append(metrics['entropy'])

        print(f"Epoch {epoch:3d} | "
              f"AvgRet {avg_ret:.2f} | "
              f"AvgLen {avg_len:.2f} | "
              f"PolL {metrics['policy_loss']:.3f} | "
              f"ValL {metrics['value_loss']:.3f} | "
              f"Ent {metrics['entropy']:.3f}")

    # —— 7. 保存指标到文件 ——
    os.makedirs("data", exist_ok=True)
    np.savez("data/ppo_metrics.npz",
             ret=ret_list,
             length=len_list,
             pol=pol_list,
             val=val_list,
             ent=ent_list)

    return agent

if __name__ == "__main__":
    train_ppo()
