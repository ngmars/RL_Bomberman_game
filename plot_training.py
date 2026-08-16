#!/usr/bin/env python3
"""Plot training metrics and Q-values logged by agent_code/my_agent/train.py."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def rolling_mean(series, window):
    return series.rolling(window, min_periods=1).mean()


def plot_training(metrics_path: Path, output_path: Path, window: int = 20):
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No metrics file at {metrics_path}. "
            "Run training first with --train 1 to generate it."
        )

    df = pd.read_csv(metrics_path)
    if df.empty:
        raise ValueError(f"{metrics_path} is empty.")

    df["reward_ma"] = rolling_mean(df["episode_reward"], window)
    df["score_ma"] = rolling_mean(df["game_score"], window)

    has_q = "max_q" in df.columns
    if has_q:
        df["max_q_ma"] = rolling_mean(df["max_q"], window)
        if "q_chosen" in df.columns:
            df["q_chosen_ma"] = rolling_mean(df["q_chosen"], window)

    n_rows = 3 if has_q else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True)

    axes[0].plot(df["round"], df["episode_reward"], alpha=0.25, color="C0", label="reward / round")
    axes[0].plot(df["round"], df["reward_ma"], color="C0", label=f"{window}-round avg")
    axes[0].set_ylabel("Episode reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["round"], df["game_score"], alpha=0.25, color="C1", label="game score / round")
    axes[1].plot(df["round"], df["score_ma"], color="C1", label=f"{window}-round avg")
    axes[1].set_ylabel("Game score (coins)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if has_q:
        q_action_cols = [
            ("q_up", "UP"),
            ("q_right", "RIGHT"),
            ("q_down", "DOWN"),
            ("q_left", "LEFT"),
            ("q_wait", "WAIT"),
            ("q_bomb", "BOMB"),
        ]
        for col, label in q_action_cols:
            if col in df.columns:
                axes[2].plot(df["round"], rolling_mean(df[col], window), label=label, alpha=0.8)

        ax2 = axes[2].twinx()
        ax2.plot(df["round"], df["max_q_ma"], color="black", linewidth=2, linestyle="--", label="max Q")
        if "q_chosen_ma" in df.columns:
            ax2.plot(df["round"], df["q_chosen_ma"], color="gray", linewidth=2, linestyle=":", label="Q(chosen)")

        axes[2].set_ylabel("Q per action")
        ax2.set_ylabel("max Q / Q(chosen)")
        axes[2].set_xlabel("Training round")
        lines1, labels1 = axes[2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[2].legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
        axes[2].grid(True, alpha=0.3)
    else:
        axes[1].set_xlabel("Training round")

    fig.suptitle("Q-Learning Training Progress", fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot my_agent training metrics.")
    parser.add_argument(
        "--metrics",
        default="agent_code/my_agent/training_metrics.csv",
        help="Path to training_metrics.csv",
    )
    parser.add_argument(
        "--output",
        default="agent_code/my_agent/training_curves.png",
        help="Where to save the plot",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling average window (rounds)",
    )
    args = parser.parse_args()
    plot_training(Path(args.metrics), Path(args.output), args.window)


if __name__ == "__main__":
    main()
