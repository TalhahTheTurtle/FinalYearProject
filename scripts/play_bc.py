"""
Evaluate a trained Behaviour Cloning policy.

Identical output format to scripts/evaluate.py so you can directly compare:
    BC-only vs PPO-from-scratch vs BC+PPO-hybrid

Usage:
    python scripts/play_bc.py --run-dir logs/bc_default --episodes 20
    python scripts/play_bc.py --run-dir logs/bc_default --stochastic
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.bc import NatureCNNPolicy
from agents.utils import load_config
from envs import make_env


def sample_action(logits: torch.Tensor, deterministic: bool) -> int:
    if deterministic:
        return int(logits.argmax(dim=-1).item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def evaluate_on_level(model, device, world, stage, env_cfg, n_episodes, deterministic):
    # Override world/stage
    this_cfg = {**env_cfg, "world": world, "stage": stage}
    env = make_env(**this_cfg)
    model.eval()

    results = []
    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward, ep_len = 0.0, 0
        done = False
        last_info: dict = {}

        while not done:
            # obs is (4, 84, 84) uint8
            obs_t = torch.from_numpy(np.asarray(obs)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(obs_t)
            action = sample_action(logits, deterministic)
            obs, reward, done, info = env.step(action)
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
            "time_to_flag": ep_len if last_info.get("flag_get", False) else None,
        })

    env.close()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--model", type=str, default="bc_model.pt",
                   help="Path (absolute, or relative to --run-dir) to the .pt file")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--stochastic", action="store_true",
                   help="Sample from the policy instead of taking argmax")
    p.add_argument("--levels", type=str, nargs="+", default=None)
    p.add_argument("--env-config", type=str, default="configs/ppo_default.yaml",
                   help="Config to take env-preprocessing settings from (world/stage overridden by --levels)")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = run_dir / model_path
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    # Use env config from a ppo config so eval preprocessing matches training
    env_cfg = load_config(args.env_config)["env"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=device)
    n_actions = int(ckpt.get("n_actions", env_cfg.get("n_actions", 7)))

    model = NatureCNNPolicy(n_actions=n_actions).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[play_bc] loaded model: {model_path}  (val_acc={ckpt.get('val_acc', '?')})")

    if args.levels is None:
        levels = [(env_cfg["world"], env_cfg["stage"])]
    else:
        levels = []
        for s in args.levels:
            w, st = s.split("-")
            levels.append((int(w), int(st)))

    deterministic = not args.stochastic
    all_results: list[dict] = []

    from agents.metrics import summarise

    for (w, s) in levels:
        print(f"[play_bc] level {w}-{s}, {args.episodes} episodes (deterministic={deterministic})")
        results = evaluate_on_level(
            model, device, w, s, env_cfg, args.episodes, deterministic,
        )
        summarise(results)
        all_results.extend(results)

    out_csv = Path(args.out) if args.out else (run_dir / "play_bc_eval.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"[play_bc] wrote {len(all_results)} rows to {out_csv}")


if __name__ == "__main__":
    main()
