"""
Smoke test for PPO training.

Runs PPO for 10,000 env steps (~1-2 minutes on a 3060 Ti) to verify that the
entire training pipeline works before you kick off a real 200k / 1M / 4M run.

What this catches:
    - Import errors
    - Env construction failures
    - Observation-space / action-space mismatch with the policy
    - Minibatch shape errors in PPO update
    - Logger / TensorBoard misconfiguration
    - Eval callback errors

What this does NOT tell you:
    - Whether PPO actually learns (too short). For that, run the 200k config.

Usage:
    python scripts/smoke_test_ppo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs import make_vec_env
from agents.callbacks import make_callbacks
from agents.utils import set_seeds


def main():
    print("[smoke] building envs (2 parallel for speed)...")
    set_seeds(0)

    train_env = make_vec_env(
        n_envs=2,
        use_subproc=False,  # DummyVecEnv simpler for a smoke test
        base_seed=0,
        world=1,
        stage=1,
    )
    eval_env = make_vec_env(
        n_envs=1,
        use_subproc=False,
        base_seed=10_000,
        world=1,
        stage=1,
    )

    print(f"  obs space:    {train_env.observation_space}")
    print(f"  action space: {train_env.action_space}")

    print("[smoke] constructing PPO with tiny batch...")
    model = PPO(
        "CnnPolicy",
        train_env,
        learning_rate=2.5e-4,
        n_steps=128,            # smaller rollout for quick iteration
        batch_size=64,
        n_epochs=2,
        gamma=0.9,
        verbose=1,
        device="auto",
    )

    log_root = ROOT / "logs" / "smoke_test"
    log_root.mkdir(parents=True, exist_ok=True)

    callbacks = make_callbacks(
        eval_env=eval_env,
        log_dir=log_root,
        eval_freq=5_000,
        n_eval_episodes=2,
        save_freq=10_000,
        verbose=1,
    )

    print("[smoke] training for 10,000 steps...")
    model.learn(total_timesteps=10_000, callback=callbacks)

    print("[smoke] PPO pipeline works end-to-end.")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
