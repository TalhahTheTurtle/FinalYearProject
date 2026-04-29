"""
Plot the action distribution from recorded demos.

Usage:
    python scripts/plot_action_dist.py --demo-dir data/demos/world1_1
    python scripts/plot_action_dist.py --demo-dir data/demos/world1_1 --out report/action_dist.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


ACTION_LABELS = ["+".join(a) if a != ["NOOP"] else "NOOP" for a in SIMPLE_MOVEMENT]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo-dir", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    demo_dir = Path(args.demo_dir)
    files = sorted(demo_dir.glob("episode_*.npz"))
    if not files:
        raise SystemExit(f"No episode_*.npz files found in {demo_dir}")

    all_actions = []
    for f in files:
        data = np.load(f)
        all_actions.append(data["actions"])
    all_actions = np.concatenate(all_actions)

    n_actions = len(SIMPLE_MOVEMENT)
    counts = np.bincount(all_actions, minlength=n_actions)
    percentages = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(n_actions), percentages, color="#4C72B0", edgecolor="white", linewidth=0.8)

    for bar, pct, count in zip(bars, percentages, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{pct:.1f}%\n({count:,})",
            ha="center", va="bottom", fontsize=8
        )

    ax.set_xticks(range(n_actions))
    ax.set_xticklabels(ACTION_LABELS, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Frequency (%)")
    ax.set_title(f"Action Distribution in Demos  ({len(files)} episodes, {len(all_actions):,} frames)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(percentages) * 1.2)

    fig.tight_layout()

    out = Path(args.out) if args.out else demo_dir / "action_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
