import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

PANELS = [
    ("score", "Game score", True),
    ("reward", "Total reward", True),
    ("steps", "Steps survived", True),
    ("loss", "Mean loss (log scale)", True),
    ("mean_q", "Mean Q(s, a)", True),
    ("epsilon", "Epsilon", False),
]


def load_metrics(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["epsilon"])
    df["episode"] = range(1, len(df) + 1)
    df["new_run"] = df["round"] == 1
    return df

def draw(fig, axes, df, window):
    for ax, (column, title, smooth) in zip(axes.flat, PANELS):
        ax.clear()
        values = df[column]
        if smooth:
            ax.plot(df["episode"], values, color="C0", alpha=0.25, linewidth=0.8)
            values = values.rolling(window, min_periods=1).mean()
        ax.plot(df["episode"], values, color="C0", linewidth=2)

        for start in df.loc[df["new_run"], "episode"].iloc[1:]:
            ax.axvline(start, color="grey", linestyle="--", linewidth=0.8)

        ax.set_title(title)
        ax.grid(alpha=0.3)

    
    if df["loss"].notna().any():
        axes[1, 0].set_yscale("log")
    for ax in axes[1]:
        ax.set_xlabel("Training round (all runs)")
    fig.suptitle(f"DQN training, {len(df)} rounds (thick line: {window}-round average)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="agent_code/dqn_agent/training_metrics.csv")
    parser.add_argument("--output", default="agent_code/dqn_agent/training_curves.png")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    if not args.live:
        draw(fig, axes, load_metrics(args.metrics), args.window)
        fig.tight_layout()
        fig.savefig(args.output, dpi=120)
        print(f"Saved {args.output}")
        return

    def refresh(_frame):
        try:
            df = load_metrics(args.metrics)
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
            return
        if df.empty:
            return
        draw(fig, axes, df, args.window)
        fig.tight_layout()

    animation = FuncAnimation(fig, refresh, interval=args.interval * 1000, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()