# plot_metrics.py
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot PPO/DQN metrics with timestamped output folder"
    )
    parser.add_argument(
        "--algo", choices=["ppo","dqn"], required=True,
        help="Which algorithm metrics to plot"
    )
    parser.add_argument(
        "--clip_ratio", type=float, default=None,
        help="If algo==ppo, specify clip ratio for naming"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) 生成输出目录名
    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.algo == "ppo":
        if args.clip_ratio is None:
            raise ValueError("Must provide --clip_ratio when plotting PPO")
        folder = f"ppo_clip{args.clip_ratio:.2f}_{ts}"
        metrics = np.load("train/data/ppo_metrics.npz")
        curves = {
            "Average Return":   metrics["ret"],
            "Episode Length":   metrics["length"],
            "Policy Loss":      metrics["pol"],
            "Value Loss":       metrics["val"],
            "Entropy":          metrics["ent"],
        }
    else:  # dqn
        folder = f"dqn_{ts}"
        metrics = np.load("train/data/dqn_metrics.npz")
        curves = {
            "Average Return":   metrics["ret"],
            "Episode Length":   metrics["length"],
            "Q Loss":           metrics["ql"],
            "Q Value Mean":     metrics["qv"],
        }

    out_dir = os.path.join("data", folder)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving plots to {out_dir}")

    epochs = np.arange(1, len(next(iter(curves.values())))+1)

    # 为每个曲线单独画图
    for name, values in curves.items():
        plt.figure()
        plt.plot(epochs, values, label=name)
        plt.title(name + " per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.grid(True)
        plt.legend()
        filename = name.lower().replace(" ","_") + ".png"
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()

if __name__ == "__main__":
    main()
