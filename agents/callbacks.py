"""
Custom SB3 callbacks for Mario PPO training.

Provides:
    - MarioEvalCallback: periodic evaluation with Mario-specific metrics
      (flag-reach rate, final x-position, episode length) in addition to
      the usual mean return.
    - CheckpointCallback: already provided by SB3, we re-export it for tidy
      imports.
    - make_callbacks(): convenience function that wires the whole lot together
      from a config dict.

Author: Talhah Anwar
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import VecEnv


class MarioEvalCallback(BaseCallback):
    """
    Evaluate the current policy every `eval_freq` env steps.

    Logs to TensorBoard:
        eval/mean_reward       - mean episodic return across eval episodes
        eval/mean_ep_length    - mean episode length (steps)
        eval/flag_reach_rate   - fraction of episodes that reached the flag
        eval/mean_x_pos        - mean x-position at episode end
        eval/max_x_pos         - best x-position across eval episodes

    Also saves a best-model checkpoint whenever mean_reward improves.

    Why a custom eval callback?
    SB3's built-in EvalCallback only tracks return and episode length. The
    interesting Mario-specific metrics (flag-reach rate, distance travelled)
    require peeking at `info` dicts, which SB3's default doesn't do.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        best_model_save_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_model_save_path = Path(best_model_save_path) if best_model_save_path else None
        if self.best_model_save_path:
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)
        self.best_mean_reward: float = -np.inf
        # Track the next total-env-steps at which to run eval. Using a threshold
        # instead of modulo ensures evals always fire even when num_timesteps
        # increments by n_envs per rollout step and never lands exactly on a
        # multiple of eval_freq.
        self._next_eval_at = eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval_at:
            return True
        # Advance the threshold first so a slow eval doesn't cause us to miss
        # subsequent scheduled evals.
        self._next_eval_at += self.eval_freq

        rewards, ep_lengths, flags, final_x = self._run_eval()

        mean_reward = float(np.mean(rewards))
        mean_ep_length = float(np.mean(ep_lengths))
        flag_reach_rate = float(np.mean(flags))
        mean_x = float(np.mean(final_x))
        max_x = float(np.max(final_x))

        # Time-to-flag, conditional on completion. NaN if zero completions in
        # this eval batch (TensorBoard handles NaN by not plotting that point).
        completed_lengths = [l for l, f in zip(ep_lengths, flags) if f]
        time_to_flag_mean = float(np.mean(completed_lengths)) if completed_lengths else float("nan")

        # Log to SB3's logger (which routes to TensorBoard + stdout)
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", mean_ep_length)
        self.logger.record("eval/flag_reach_rate", flag_reach_rate)
        self.logger.record("eval/mean_x_pos", mean_x)
        self.logger.record("eval/max_x_pos", max_x)
        self.logger.record("eval/time_to_flag_mean", time_to_flag_mean)

        if self.verbose >= 1:
            print(
                f"[eval @ {self.num_timesteps:>9} steps]  "
                f"return={mean_reward:+.1f}  "
                f"ep_len={mean_ep_length:.0f}  "
                f"flag_rate={flag_reach_rate:.1%}  "
                f"mean_x={mean_x:.0f}  "
                f"max_x={max_x:.0f}"
            )

        # Save best model by mean eval return
        if self.best_model_save_path and mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            save_path = self.best_model_save_path / "best_model.zip"
            self.model.save(str(save_path))
            if self.verbose >= 1:
                print(f"  new best model saved: {save_path}")

        return True

    def _run_eval(self):
        """Run N eval episodes and return per-episode metrics."""
        rewards, lengths, flags, final_x = [], [], [], []

        # We ignore the vec structure and just drive env 0 sequentially.
        # Eval is cheap so we don't need parallelism here, and it keeps metric
        # extraction from info dicts clean.
        obs = self.eval_env.reset()
        ep_reward, ep_len = 0.0, 0
        ep_count = 0

        while ep_count < self.n_eval_episodes:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, done, infos = self.eval_env.step(action)
            ep_reward += float(reward[0])
            ep_len += 1

            # For a VecEnv of size 1, done[0] signals episode boundary.
            if done[0]:
                # On episode end SB3's VecEnv auto-resets and stores the *final*
                # info of the completed episode in infos[0]["terminal_observation"]-
                # adjacent fields. gym-super-mario-bros keeps flag_get / x_pos in
                # the normal info dict at the done step, which VecEnv preserves.
                info = infos[0]
                rewards.append(ep_reward)
                lengths.append(ep_len)
                flags.append(bool(info.get("flag_get", False)))
                final_x.append(int(info.get("x_pos", 0)))
                ep_reward, ep_len = 0.0, 0
                ep_count += 1

        return rewards, lengths, flags, final_x


def make_callbacks(
    eval_env: VecEnv,
    log_dir: Path,
    eval_freq: int,
    n_eval_episodes: int,
    save_freq: int,
    deterministic_eval: bool = True,
    verbose: int = 1,
) -> CallbackList:
    """
    Wire up the standard training callbacks.

    Note on `eval_freq` vs `save_freq`:
      - Both are in *total* env steps (summed across parallel envs).
      - SB3's CheckpointCallback expects `save_freq` to be per-env-step-rate,
        so we divide by vec_size implicitly via how SB3 calls it internally.
        In practice, pass the intended total step count and SB3 handles it.
    """
    log_dir = Path(log_dir)
    ckpt_dir = log_dir / "ckpts"
    best_dir = log_dir / "best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    eval_cb = MarioEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic_eval,
        best_model_save_path=str(best_dir),
        verbose=verbose,
    )

    # SB3 interprets CheckpointCallback's save_freq as a per-env rate in its
    # internal loop. We effectively call back every save_freq *total* env steps
    # by setting it equal to save_freq // n_envs elsewhere if desired. For
    # simplicity we pass save_freq directly here; for n_envs=8 this checkpoints
    # roughly every 8 * save_freq total steps.
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, save_freq // max(1, eval_env.num_envs)),
        save_path=str(ckpt_dir),
        name_prefix="ppo_ckpt",
        verbose=verbose,
    )

    return CallbackList([eval_cb, ckpt_cb])
