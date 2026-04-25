"""
Train BC+PPO hybrid: load BC weights into PPO, then fine-tune with PPO.

Usage:
    python scripts/train_hybrid.py --config configs/hybrid_default.yaml
    python scripts/train_hybrid.py --config configs/hybrid_default.yaml --bc-ckpt logs/bc_no_class_weights/bc_model.pt --run-name hybrid_500k_v1

This is structurally `train_ppo.py` plus one extra step: the BC weight
transplant happens immediately after the PPO model is constructed, before
`.learn()`.

Design decisions baked in:
    1. Same env preprocessing as the PPO baseline. Otherwise comparison is
       not apples-to-apples.
    2. Lower default LR than from-scratch PPO (1e-4 instead of 2.5e-4) so
       PPO doesn't immediately blow up the BC initialisation.
    3. Lower entropy coefficient than from-scratch PPO. BC starts the policy
       in a sensible region; we don't need as much exploration noise.

Run the result through scripts/evaluate.py exactly like a from-scratch PPO
model -- the training pipeline produces a normal SB3 .zip checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from envs import make_vec_env
from agents.callbacks import make_callbacks
from agents.utils import load_config, set_seeds, write_run_metadata
from agents.transplant import transplant_bc_into_ppo


def build_train_and_eval_envs(cfg: dict):
    env_kwargs = dict(cfg["env"])
    vec_cfg = cfg["vec_env"]
    train_env = make_vec_env(
        n_envs=vec_cfg["n_envs"],
        use_subproc=vec_cfg["use_subproc"],
        base_seed=vec_cfg["base_seed"],
        **env_kwargs,
    )
    eval_env = make_vec_env(
        n_envs=1,
        use_subproc=False,
        base_seed=vec_cfg["base_seed"] + 10_000,
        **env_kwargs,
    )
    return train_env, eval_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/hybrid_default.yaml")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--bc-ckpt", type=str, default=None,
                   help="Override cfg.bc_ckpt (path to a BC .pt file)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    cfg = load_config(args.config)

    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.timesteps is not None:
        cfg["train"]["total_timesteps"] = args.timesteps
    if args.bc_ckpt is not None:
        cfg["bc_ckpt"] = args.bc_ckpt
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed

    bc_ckpt = cfg.get("bc_ckpt")
    if not bc_ckpt:
        raise SystemExit(
            "No BC checkpoint specified. Set `bc_ckpt:` in the config or pass "
            "--bc-ckpt /path/to/bc_model.pt"
        )
    bc_ckpt_path = Path(bc_ckpt)
    if not bc_ckpt_path.is_absolute():
        bc_ckpt_path = ROOT / bc_ckpt_path
    if not bc_ckpt_path.exists():
        raise SystemExit(f"BC checkpoint not found: {bc_ckpt_path}")

    run_name = cfg["run_name"]
    seed = cfg["train"]["seed"]
    total_timesteps = cfg["train"]["total_timesteps"]

    log_root = ROOT / cfg["train"]["log_dir"] / run_name
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"[train_hybrid] run dir: {log_root}")

    set_seeds(seed)
    write_run_metadata(log_root, cfg)

    print("[train_hybrid] building environments...")
    train_env, eval_env = build_train_and_eval_envs(cfg)
    print(f"  obs space:    {train_env.observation_space}")
    print(f"  action space: {train_env.action_space}")

    tb_dir = log_root / "tb"
    new_logger = configure(str(tb_dir), ["stdout", "tensorboard"])

    ppo_cfg = dict(cfg["ppo"])
    policy_str = ppo_cfg.pop("policy", "CnnPolicy")
    print(f"[train_hybrid] constructing PPO ({policy_str}) with hybrid hyperparameters")

    model = PPO(
        policy_str,
        train_env,
        seed=seed,
        device=args.device,
        **ppo_cfg,
    )
    model.set_logger(new_logger)

    # ---- BC weight transplant (the whole point of this script) ----
    print(f"[train_hybrid] transplanting BC weights from {bc_ckpt_path}")
    transplant_info = transplant_bc_into_ppo(bc_ckpt_path, model, verbose=True)

    # Save what we did so the report can cite the exact BC checkpoint used.
    import json
    with (log_root / "transplant_info.json").open("w") as f:
        # tuples in transferred -> turn into lists
        json.dump({
            "transferred": [list(t) for t in transplant_info["transferred"]],
            "skipped_count": len(transplant_info["skipped"]),
            "bc_val_acc": transplant_info["bc_val_acc"],
            "bc_epoch_at_save": transplant_info["bc_epoch_at_save"],
            "bc_ckpt_path": str(bc_ckpt_path),
        }, f, indent=2, default=str)

    callbacks = make_callbacks(
        eval_env=eval_env,
        log_dir=log_root,
        eval_freq=cfg["train"]["eval_freq"],
        n_eval_episodes=cfg["train"]["eval_episodes"],
        save_freq=cfg["train"]["save_freq"],
        deterministic_eval=cfg["eval"]["deterministic"],
        verbose=1,
    )

    print(f"[train_hybrid] starting fine-tune for {total_timesteps:,} env steps")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[train_hybrid] interrupted -- saving current model before exit")
    finally:
        final_path = log_root / "final_model.zip"
        model.save(str(final_path))
        print(f"[train_hybrid] final model saved to {final_path}")
        train_env.close()
        eval_env.close()

    print(f"[train_hybrid] done. tensorboard --logdir {log_root.parent}")


if __name__ == "__main__":
    main()
