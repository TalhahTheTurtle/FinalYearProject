"""
il + PPO hybrid with value-head warmup.


The warmup phase runs ~50k env steps (configurable) where ONLY the value
head is updated. This gives PPO a critic with sensible value estimates
before any policy update is allowed, preventing the policy collapse seen
in the naive hybrid.

Usage:
    python scripts/train_hybrid_warmup.py --config configs/hybrid_warmup.yaml
"""
from __future__ import annotations

import argparse
import json
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
from agents.value_warmup import warmup_value_head


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
    p.add_argument("--config", type=str, default="configs/hybrid_warmup.yaml")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--bc-ckpt", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--skip-warmup", action="store_true",
                   help="Disable warmup (equivalent to plain train_hybrid.py)")
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
        raise SystemExit("No BC checkpoint specified. Set `bc_ckpt:` in the config or pass --bc-ckpt")
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
    print(f"[train_hybrid_warmup] run dir: {log_root}")

    set_seeds(seed)
    write_run_metadata(log_root, cfg)

    print("[train_hybrid_warmup] building environments...")
    train_env, eval_env = build_train_and_eval_envs(cfg)
    print(f"  obs space:    {train_env.observation_space}")
    print(f"  action space: {train_env.action_space}")

    tb_dir = log_root / "tb"
    new_logger = configure(str(tb_dir), ["stdout", "tensorboard"])

    ppo_cfg = dict(cfg["ppo"])
    policy_str = ppo_cfg.pop("policy", "CnnPolicy")
    print(f"[train_hybrid_warmup] constructing PPO ({policy_str}) with hybrid hyperparameters")
    model = PPO(policy_str, train_env, seed=seed, device=args.device, **ppo_cfg)
    model.set_logger(new_logger)

    # ---- BC weight transplant ----
    print(f"[train_hybrid_warmup] transplanting BC weights from {bc_ckpt_path}")
    transplant_info = transplant_bc_into_ppo(bc_ckpt_path, model, verbose=True)
    with (log_root / "transplant_info.json").open("w") as f:
        json.dump({
            "transferred": [list(t) for t in transplant_info["transferred"]],
            "skipped_count": len(transplant_info["skipped"]),
            "bc_val_acc": transplant_info["bc_val_acc"],
            "bc_epoch_at_save": transplant_info["bc_epoch_at_save"],
            "bc_ckpt_path": str(bc_ckpt_path),
        }, f, indent=2, default=str)

    # ---- Value-head warmup (the new bit) ----
    if not args.skip_warmup:
        warmup_cfg = cfg.get("warmup", {})
        n_warmup_steps = warmup_cfg.get("n_warmup_steps", 50_000)
        warmup_lr = warmup_cfg.get("lr", 1e-3)
        rollout_size = warmup_cfg.get("rollout_size", 2048)
        minibatch_size = warmup_cfg.get("minibatch_size", 256)
        n_grad_passes = warmup_cfg.get("n_grad_passes_per_rollout", 4)

        print(f"[train_hybrid_warmup] starting value-head warmup ({n_warmup_steps:,} env steps)")
        warmup_history = warmup_value_head(
            model=model,
            n_warmup_steps=n_warmup_steps,
            rollout_size=rollout_size,
            minibatch_size=minibatch_size,
            n_grad_passes_per_rollout=n_grad_passes,
            lr=warmup_lr,
        )
        with (log_root / "warmup_history.json").open("w") as f:
            json.dump(warmup_history, f, indent=2, default=str)

        # Save a "post-warmup" checkpoint so we can A/B test the warmed-up
        # model directly without needing to re-run.
        post_warmup_path = log_root / "post_warmup_model.zip"
        model.save(str(post_warmup_path))
        print(f"[train_hybrid_warmup] saved post-warmup model to {post_warmup_path}")
    else:
        print("[train_hybrid_warmup] WARMUP SKIPPED (--skip-warmup); behaving like train_hybrid.py")

    # ---- Main PPO training ----
    callbacks = make_callbacks(
        eval_env=eval_env,
        log_dir=log_root,
        eval_freq=cfg["train"]["eval_freq"],
        n_eval_episodes=cfg["train"]["eval_episodes"],
        save_freq=cfg["train"]["save_freq"],
        deterministic_eval=cfg["eval"]["deterministic"],
        verbose=1,
    )

    print(f"[train_hybrid_warmup] starting PPO fine-tune for {total_timesteps:,} env steps")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[train_hybrid_warmup] interrupted -- saving current model")
    finally:
        final_path = log_root / "final_model.zip"
        model.save(str(final_path))
        print(f"[train_hybrid_warmup] final model saved to {final_path}")
        train_env.close()
        eval_env.close()

    print(f"[train_hybrid_warmup] done. tensorboard --logdir {log_root.parent}")


if __name__ == "__main__":
    main()
