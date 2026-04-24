"""
Inspect recorded demo files before training BC.

Usage:
    python scripts/inspect_demos.py --demo-dir data/demos/world1_1

Reports:
    - Per-episode: frames, x_pos, flag_get, score
    - Aggregate: total frames, total minutes of gameplay at 15 Hz
    - Action distribution across all episodes
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo-dir", type=str, required=True)
    args = p.parse_args()

    demo_dir = Path(args.demo_dir)
    if not demo_dir.is_absolute():
        demo_dir = ROOT / demo_dir
    files = sorted(demo_dir.glob("episode_*.npz"))
    if not files:
        raise SystemExit(f"No episode_*.npz files in {demo_dir}")

    print(f"[inspect] directory: {demo_dir}")
    print(f"[inspect] found {len(files)} episode files\n")

    print(f"{'episode':>10}  {'frames':>7}  {'x_pos':>6}  {'flag':>5}  {'score':>6}")
    print("-" * 45)

    total_frames = 0
    flag_count = 0
    all_actions = []
    x_positions = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        n = data["observations"].shape[0]
        x = int(data["x_pos"])
        flag = bool(data["flag_get"])
        score = int(data["score"])
        print(f"{f.stem:>10}  {n:>7}  {x:>6}  {str(flag):>5}  {score:>6}")
        total_frames += n
        flag_count += 1 if flag else 0
        all_actions.extend(data["actions"].tolist())
        x_positions.append(x)

    # Aggregate
    print("-" * 45)
    print(f"total frames:          {total_frames:,}")
    print(f"approx. gameplay time: {total_frames / 15 / 60:.1f} min (at 15 Hz decision rate)")
    print(f"flag reaches:          {flag_count}/{len(files)}")
    print(f"x_pos:                 min={min(x_positions)}, max={max(x_positions)}, mean={sum(x_positions)/len(x_positions):.0f}")

    # Action distribution
    print("\n[inspect] action distribution across all frames:")
    counts = Counter(all_actions)
    total = sum(counts.values())
    for a in range(len(SIMPLE_MOVEMENT)):
        c = counts.get(a, 0)
        name = "+".join(SIMPLE_MOVEMENT[a]) if SIMPLE_MOVEMENT[a] != ["NOOP"] else "NOOP"
        bar = "#" * int(40 * c / total) if total > 0 else ""
        print(f"  {a} {name:<18} {c:>6d} ({100*c/total:5.1f}%)  {bar}")


if __name__ == "__main__":
    main()
