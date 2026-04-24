"""
Evaluate a trained PPO agent and produce a metrics CSV.

Usage:
    # Evaluate best model on the level it was trained on
    python scripts/evaluate.py --run-dir logs/ppo_default

    # Evaluate on several levels (generalisation test)
    python scripts/evaluate.py --run-dir logs/ppo_default --levels 1-1 1-2 1-3 1-4

    # Use the final model instead of best
    python scripts/evaluate.py --run-dir logs/ppo_default --model final_model.zip

The CSV records one row per episode with:
    level, episode, reward, length, flag_get, final_x_pos, final_score
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs import make_env
from agents.utils import load_config


def parse_levels(level_strs: list[str]) -> list[tuple[int, int]]:
    """Parse ['1-1', '1-2', ...] into [(1, 1), (1, 2), ...]."""
    out = []
    for s in level_strs:
        parts = s.split("-")
        if len(parts) != 2:
            raise ValueError(f"Bad level spec: {s} (expected 'W-S')")
        out.append((int(parts[0]), int(parts[1])))
    return out


def evaluate_on_level(
    model: PPO,
    world: int,
    stage: int,
    env_cfg: dict,
    n_episodes: int,
    deterministic: bool,
) -> list[dict]:
    """Run n_episodes on (world, stage), return one metrics dict per episode."""
    # Override world/stage in the env config
    this_cfg = {**env_cfg, "world": world, "stage": stage}
    env = make_env(**this_cfg)

    results = []
    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward, ep_len = 0.0, 0
        done = False
        last_info: dict = {}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            # env is a single env, so action is a scalar-ish numpy value
            # and step returns 4-tuple (classic gym API)
            obs, reward, done, info = env.step(int(action))
            ep_reward += float(reward)
            ep_len += 1
            last_info = info

        results.append({
            "level": f"{world}-{stage}",
            "episode": ep,
            "reward": round(ep_reward, 2),
            "length": ep_len,
            "flag_get": bool(last_info.get("flag_get", False)),
            "final_x_pos": int(last_info.get("x_pos", 0)),
            "final_score": int(last_info.get("score", 0)),
        })

    env.close()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True,
                   help="Directory produced by train_ppo.py, e.g. logs/ppo_default")
    p.add_argument("--model", type=str, default="best/best_model.zip",
                   help="Path relative to --run-dir, or absolute")
    p.add_argument("--levels", type=str, nargs="+", default=None,
                   help="Levels to evaluate on, e.g. 1-1 1-2 1-3. Default: training level only.")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--stochastic", action="store_true",
                   help="Use stochastic (non-deterministic) action sampling")
    p.add_argument("--out", type=str, default=None,
                   help="Output CSV path. Defaults to <run-dir>/eval.csv")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run-dir does not exist: {run_dir}")

    # Resolve model path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = run_dir / model_path
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    # Load the config that was saved alongside the run so we eval with the
    # same preprocessing that was used for training.
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(
            f"config.yaml not found in {run_dir}. Did this run finish training?"
        )
    cfg = load_config(cfg_path)

    # Default to evaluating on the trained level
    if args.levels is None:
        levels = [(cfg["env"]["world"], cfg["env"]["stage"])]
    else:
        levels = parse_levels(args.levels)

    print(f"[evaluate] loading model: {model_path}")
    model = PPO.load(str(model_path))

    out_csv = Path(args.out) if args.out else (run_dir / "eval.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    env_cfg = cfg["env"]
    deterministic = not args.stochastic

    for (w, s) in levels:
        print(f"[evaluate] level {w}-{s}, {args.episodes} episodes (deterministic={deterministic})")
        results = evaluate_on_level(
            model, w, s, env_cfg,
            n_episodes=args.episodes,
            deterministic=deterministic,
        )
        # Summary for this level
        xs = [r["final_x_pos"] for r in results]
        rws = [r["reward"] for r in results]
        lens = [r["length"] for r in results]
        n_flag = sum(r["flag_get"] for r in results)
        n = len(results)

        import statistics as _st
        x_stdev = _st.stdev(xs) if n > 1 else 0.0
        print(
            f"  reward:   mean={_st.mean(rws):+.1f}  median={_st.median(rws):+.1f}  max={max(rws):+.1f}"
        )
        print(
            f"  x_pos:    mean={_st.mean(xs):.0f}  median={_st.median(xs):.0f}  "
            f"max={max(xs)}  stdev={x_stdev:.0f}"
        )
        print(
            f"  ep_len:   mean={_st.mean(lens):.0f}  max={max(lens)}"
        )
        print(f"  flag:     {n_flag}/{n}  ({100*n_flag/n:.0f}%)")
        all_results.extend(results)

    # Write CSV
    fieldnames = list(all_results[0].keys())
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"[evaluate] wrote {len(all_results)} rows to {out_csv}")


if __name__ == "__main__":
    main()
