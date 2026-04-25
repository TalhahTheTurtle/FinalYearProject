"""
BC + PPO concurrent training (the full pipeline).

Pipeline:
    1. Build HybridPPO (PPO + concurrent BC auxiliary loss).
    2. Transplant BC weights from a pre-trained checkpoint.
    3. Warm up the value head with the policy frozen.
    4. Call model.learn() -- BC loss fires on every gradient step throughout.

The BC auxiliary loss is annealed linearly from bc_coef_start to bc_coef_end
over the course of training, so the policy gradually transitions from
"stay near demos" to "explore freely with PPO".

Usage:
    python scripts/train_hybrid_concurrent.py
    python scripts/train_hybrid_concurrent.py --config configs/hybrid_concurrent.yaml
    python scripts/train_hybrid_concurrent.py --bc-ckpt logs/bc_default/bc_model.pt --run-name my_run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3.common.logger import configure

from envs import make_vec_env
from agents.callbacks import make_callbacks
from agents.utils import load_config, set_seeds, write_run_metadata
from agents.transplant import transplant_bc_into_ppo
from agents.value_warmup import warmup_value_head
from agents.hybrid_ppo import HybridPPO


def build_envs(cfg: dict):
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
    p.add_argument("--config", default="configs/hybrid_concurrent.yaml")
    p.add_argument("--run-name", default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--bc-ckpt", default=None, help="Path to BC .pt checkpoint")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-warmup", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.timesteps:
        cfg["train"]["total_timesteps"] = args.timesteps
    if args.bc_ckpt:
        cfg["bc_ckpt"] = args.bc_ckpt
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed

    bc_ckpt = cfg.get("bc_ckpt")
    if not bc_ckpt:
        raise SystemExit("No BC checkpoint. Set bc_ckpt: in config or pass --bc-ckpt.")
    bc_ckpt_path = Path(bc_ckpt)
    if not bc_ckpt_path.is_absolute():
        bc_ckpt_path = ROOT / bc_ckpt_path
    if not bc_ckpt_path.exists():
        raise SystemExit(f"BC checkpoint not found: {bc_ckpt_path}")

    demo_dir = cfg.get("demo_dir", "data/demos/world1_1")
    demo_dir_path = Path(demo_dir)
    if not demo_dir_path.is_absolute():
        demo_dir_path = ROOT / demo_dir_path

    run_name = cfg["run_name"]
    seed = cfg["train"]["seed"]
    total_timesteps = cfg["train"]["total_timesteps"]

    log_root = ROOT / cfg["train"]["log_dir"] / run_name
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"[train_hybrid_concurrent] run dir: {log_root}")

    set_seeds(seed)
    write_run_metadata(log_root, cfg)

    print("[train_hybrid_concurrent] building environments...")
    train_env, eval_env = build_envs(cfg)

    tb_dir = log_root / "tb"
    new_logger = configure(str(tb_dir), ["stdout", "tensorboard"])

    ppo_cfg = dict(cfg["ppo"])
    policy_str = ppo_cfg.pop("policy", "CnnPolicy")
    bc_cfg = cfg.get("bc_aux", {})

    print(f"[train_hybrid_concurrent] constructing HybridPPO ({policy_str})")
    model = HybridPPO(
        policy_str,
        train_env,
        seed=seed,
        device=args.device,
        demo_dir=demo_dir_path,
        bc_coef_start=bc_cfg.get("bc_coef_start", 0.5),
        bc_coef_end=bc_cfg.get("bc_coef_end", 0.05),
        bc_batch_size=bc_cfg.get("bc_batch_size", 256),
        **ppo_cfg,
    )
    model.set_logger(new_logger)

    # ---- BC weight transplant ----
    print(f"[train_hybrid_concurrent] transplanting BC weights from {bc_ckpt_path}")
    transplant_info = transplant_bc_into_ppo(bc_ckpt_path, model, verbose=True)
    with (log_root / "transplant_info.json").open("w") as f:
        json.dump({
            "transferred": [list(t) for t in transplant_info["transferred"]],
            "skipped_count": len(transplant_info["skipped"]),
            "bc_val_acc": transplant_info["bc_val_acc"],
            "bc_epoch_at_save": transplant_info["bc_epoch_at_save"],
            "bc_ckpt_path": str(bc_ckpt_path),
        }, f, indent=2, default=str)

    # ---- Value-head warmup ----
    if not args.skip_warmup:
        warmup_cfg = cfg.get("warmup", {})
        print(f"[train_hybrid_concurrent] starting value-head warmup ({warmup_cfg.get('n_warmup_steps', 50_000):,} steps)")
        warmup_history = warmup_value_head(
            model=model,
            n_warmup_steps=warmup_cfg.get("n_warmup_steps", 50_000),
            rollout_size=warmup_cfg.get("rollout_size", 2048),
            minibatch_size=warmup_cfg.get("minibatch_size", 256),
            n_grad_passes_per_rollout=warmup_cfg.get("n_grad_passes_per_rollout", 4),
            lr=warmup_cfg.get("lr", 1e-3),
        )
        with (log_root / "warmup_history.json").open("w") as f:
            json.dump(warmup_history, f, indent=2, default=str)
        post_warmup_path = log_root / "post_warmup_model.zip"
        model.save(str(post_warmup_path))
        print(f"[train_hybrid_concurrent] post-warmup model saved to {post_warmup_path}")

    # ---- Main PPO+BC training ----
    callbacks = make_callbacks(
        eval_env=eval_env,
        log_dir=log_root,
        eval_freq=cfg["train"]["eval_freq"],
        n_eval_episodes=cfg["train"]["eval_episodes"],
        save_freq=cfg["train"]["save_freq"],
        deterministic_eval=cfg["eval"]["deterministic"],
        verbose=1,
    )

    print(f"[train_hybrid_concurrent] starting HybridPPO for {total_timesteps:,} steps")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[train_hybrid_concurrent] interrupted -- saving current model")
    finally:
        final_path = log_root / "final_model.zip"
        model.save(str(final_path))
        print(f"[train_hybrid_concurrent] final model saved to {final_path}")
        train_env.close()
        eval_env.close()

    print(f"[train_hybrid_concurrent] done. tensorboard --logdir {log_root.parent}")


if __name__ == "__main__":
    main()
