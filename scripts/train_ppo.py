"""
Train a PPO agent on Super Mario Bros.

Usage:
    python scripts/train_ppo.py --config configs/ppo_default.yaml
    python scripts/train_ppo.py --config configs/ppo_default.yaml --timesteps 1_000_000
    python scripts/train_ppo.py --config configs/ppo_default.yaml --run-name ppo_1_1_seedA

The --timesteps and --run-name flags override the YAML for quick experiments
without editing the file.

Launch TensorBoard to watch training:
    tensorboard --logdir logs/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_schedule_fn

from envs import make_vec_env
from agents.callbacks import make_callbacks
from agents.utils import load_config, set_seeds, write_run_metadata


def build_train_and_eval_envs(cfg: dict):
    """Build a vectorised training env and a single-env vec for eval."""
    env_kwargs = dict(cfg["env"])
    vec_cfg = cfg["vec_env"]

    train_env = make_vec_env(
        n_envs=vec_cfg["n_envs"],
        use_subproc=vec_cfg["use_subproc"],
        base_seed=vec_cfg["base_seed"],
        **env_kwargs,
    )

    # Eval env is a single non-parallel env. We use n_envs=1 DummyVecEnv so
    # the SB3 .predict() / .step() API matches what the training loop expects,
    # without paying the subprocess overhead for a single environment.
    eval_env = make_vec_env(
        n_envs=1,
        use_subproc=False,
        base_seed=vec_cfg["base_seed"] + 10_000,
        **env_kwargs,
    )

    return train_env, eval_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/ppo_default.yaml")
    p.add_argument("--run-name", type=str, default=None, help="Override run_name in config")
    p.add_argument("--timesteps", type=int, default=None, help="Override train.total_timesteps")
    p.add_argument("--seed", type=int, default=None, help="Override train.seed")
    p.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    p.add_argument("--resume-from", type=str, default=None, help="Path to a .zip checkpoint to fine-tune from")
    p.add_argument("--level", type=str, default=None, help="Override level, e.g. 1-3")
    args = p.parse_args()

    # ---- Load config ----
    cfg = load_config(args.config)

    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.timesteps is not None:
        cfg["train"]["total_timesteps"] = args.timesteps
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.level is not None:
        w, s = args.level.split("-")
        cfg["env"]["world"] = int(w)
        cfg["env"]["stage"] = int(s)

    run_name = cfg["run_name"]
    seed = cfg["train"]["seed"]
    total_timesteps = cfg["train"]["total_timesteps"]

    # ---- Set up run directory ----
    log_root = ROOT / cfg["train"]["log_dir"] / run_name
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"[train_ppo] run dir: {log_root}")

    set_seeds(seed)
    write_run_metadata(log_root, cfg)

    # ---- Build envs ----
    print("[train_ppo] building environments...")
    train_env, eval_env = build_train_and_eval_envs(cfg)
    print(f"  n_envs (train): {train_env.num_envs}")
    print(f"  obs space:      {train_env.observation_space}")
    print(f"  action space:   {train_env.action_space}")

    # ---- Configure SB3 logger -> TensorBoard + stdout ----
    tb_dir = log_root / "tb"
    new_logger = configure(str(tb_dir), ["stdout", "tensorboard"])

    # ---- PPO model ----
    ppo_cfg = dict(cfg["ppo"])
    policy_str = ppo_cfg.pop("policy", "CnnPolicy")

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        print(f"[train_ppo] fine-tuning from {resume_path}")
        model = PPO.load(str(resume_path), env=train_env, device=args.device)
        model.learning_rate = get_schedule_fn(ppo_cfg.get("learning_rate", model.learning_rate))
        model.clip_range = get_schedule_fn(ppo_cfg.get("clip_range", model.clip_range))
        model.ent_coef = ppo_cfg.get("ent_coef", model.ent_coef)
    else:
        model = PPO(
            policy_str,
            train_env,
            seed=seed,
            device=args.device,
            **ppo_cfg,
        )
    model.set_logger(new_logger)

    # ---- Callbacks ----
    callbacks = make_callbacks(
        eval_env=eval_env,
        log_dir=log_root,
        eval_freq=cfg["train"]["eval_freq"],
        n_eval_episodes=cfg["train"]["eval_episodes"],
        save_freq=cfg["train"]["save_freq"],
        deterministic_eval=cfg["eval"]["deterministic"],
        verbose=1,
    )

    # ---- Train ----
    reset_timesteps = args.resume_from is None
    print(f"[train_ppo] starting training for {total_timesteps:,} env steps")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=reset_timesteps)
    except KeyboardInterrupt:
        print("\n[train_ppo] interrupted -- saving current model before exit")
    finally:
        final_path = log_root / "final_model.zip"
        model.save(str(final_path))
        print(f"[train_ppo] final model saved to {final_path}")

        train_env.close()
        eval_env.close()

    print(f"[train_ppo] done. tensorboard --logdir {log_root.parent}")


if __name__ == "__main__":
    main()
