# utils/rl_utils.py
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE).
    返回 (advantages, returns)，两个 list 与 rewards 同长度。
    """
    T = len(rewards)
    advantages = [0.0] * T
    returns    = [0.0] * T
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        returns[t]    = gae + values[t]
        next_value    = values[t]
    return advantages, returns
