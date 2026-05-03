"""
Record a video of a trained model playing Super Mario Bros.

Usage:
    python scripts/record_run.py --run-dir logs/ppo_default --episodes 3

Saves:
    <run-dir>/videos/episode_0.mp4, episode_1.mp4, ...

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs import make_env
from agents.utils import load_config


def record_episode(model, env, max_steps: int = 4000) -> tuple[list, dict]:
    """Run one episode, collect raw rgb frames, return (frames, final_info)."""
    frames = []
    obs = env.reset()
    done = False
    steps = 0
    last_info: dict = {}

    while not done and steps < max_steps:
        # .render on the underlying NES env returns the raw RGB frame
        frame = env.render(mode="rgb_array")
        if frame is not None:
            frames.append(frame.copy())
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(int(action))
        last_info = info
        steps += 1

    return frames, last_info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--model", type=str, default="best/best_model.zip")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--level", type=str, default=None, help="Override level, e.g. 1-2")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = run_dir / model_path
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    cfg = load_config(run_dir / "config.yaml")
    env_cfg = dict(cfg["env"])

    if args.level is not None:
        w, s = args.level.split("-")
        env_cfg["world"] = int(w)
        env_cfg["stage"] = int(s)

    env = make_env(**env_cfg)

    print(f"[record] loading model {model_path}")
    model = PPO.load(str(model_path))

    out_dir = run_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        frames, info = record_episode(model, env)
        flag = info.get("flag_get", False)
        x = info.get("x_pos", 0)
        suffix = "flag" if flag else f"x{x}"
        out_path = out_dir / f"episode_{ep}_{env_cfg['world']}-{env_cfg['stage']}_{suffix}.mp4"
        imageio.mimsave(out_path, frames, fps=args.fps, codec="libx264", quality=8)
        print(f"  [{ep}] steps={len(frames)}  flag_get={flag}  x={x}  -> {out_path}")

    env.close()


if __name__ == "__main__":
    main()
