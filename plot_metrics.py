# plot_metrics.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# 创建带时间戳的子目录
run_id = time.strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join("data", run_id)
os.makedirs(out_dir, exist_ok=True)

# 加载数据
ppo = np.load("data/ppo_metrics.npz")
dqn = np.load("data/dqn_metrics.npz")
epochs = np.arange(1, len(ppo["ret"])+1)

# 1) Average Return
plt.figure()
plt.plot(epochs, ppo["ret"], label="PPO")
plt.plot(epochs, dqn["ret"], label="DQN")
plt.xlabel("Epoch"); plt.ylabel("Avg Return")
plt.title("Average Return per Epoch")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(out_dir, "avg_return.png"))

# 2) Episode Length
plt.figure()
plt.plot(epochs, ppo["length"], label="PPO")
plt.plot(epochs, dqn["length"], label="DQN")
plt.xlabel("Epoch"); plt.ylabel("Avg Episode Length")
plt.title("Avg Episode Length per Epoch")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(out_dir, "avg_length.png"))

# 3) Loss Comparison
plt.figure()
plt.plot(epochs, ppo["pol"], label="PPO Policy Loss")
plt.plot(epochs, ppo["val"], label="PPO Value Loss")
plt.plot(epochs, dqn["ql"], label="DQN Q Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(out_dir, "losses.png"))

# 4) PPO Entropy
plt.figure()
plt.plot(epochs, ppo["ent"], label="PPO Entropy")
plt.xlabel("Epoch"); plt.ylabel("Entropy")
plt.title("PPO Entropy per Epoch")
plt.grid(True)
plt.savefig(os.path.join(out_dir, "entropy.png"))

print(f"Plots saved in: {out_dir}")
