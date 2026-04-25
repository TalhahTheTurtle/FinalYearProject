"""
HybridPPO: PPO with a concurrent behaviour-cloning auxiliary loss.

The standard sequential approach (train BC -> transplant weights -> fine-tune PPO)
loses the imitation signal the moment PPO starts. HybridPPO keeps it alive: during
every PPO gradient update, a minibatch is sampled from the recorded human demos and
a negative-log-likelihood (BC) loss is added to the PPO objective.

  total_loss = ppo_loss + bc_coef(t) * bc_loss

bc_coef is annealed linearly from bc_coef_start (high, near the start of training)
to bc_coef_end (low, near the end) so that the policy can still diverge from the
demonstrations when PPO finds better strategies.

Designed to be used on top of the existing transplant pipeline:
    1. Build PPO model.
    2. Transplant BC weights into it (agents.transplant.transplant_bc_into_ppo).
    3. Optionally run value-head warmup (agents.value_warmup.warmup_value_head).
    4. Call model.learn() -- the concurrent BC loss fires automatically.

Compatible with SB3 1.8.0 + gym 0.21.0 + CnnPolicy on (4, 84, 84) observations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


# ---------------------------------------------------------------------------
# Demo sampler
# ---------------------------------------------------------------------------

class DemoSampler:
    """
    Loads all episode_*.npz files from demo_dir and provides random minibatch
    sampling for the BC auxiliary loss.

    Observations are stored as uint8 (raw env output). SB3's policy.evaluate_actions
    runs preprocess_obs internally, which converts uint8 -> float32 / 255.0, so
    we pass uint8 tensors directly without pre-dividing.
    """

    def __init__(self, demo_dir: str | Path):
        demo_dir = Path(demo_dir)
        files = sorted(demo_dir.glob("episode_*.npz"))
        if not files:
            raise FileNotFoundError(f"No episode_*.npz files found in {demo_dir}")

        obs_chunks, act_chunks = [], []
        for f in files:
            d = np.load(f, allow_pickle=True)
            obs_chunks.append(d["observations"])   # (T, 4, 84, 84) uint8
            act_chunks.append(d["actions"])         # (T,) int

        self.obs = np.concatenate(obs_chunks, axis=0)       # (N, 4, 84, 84)
        self.actions = np.concatenate(act_chunks, axis=0)   # (N,) int
        self._n = len(self.actions)
        print(f"[DemoSampler] loaded {self._n:,} frames from {len(files)} episodes in {demo_dir}")

    def sample(self, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        idx = np.random.randint(0, self._n, size=n)
        obs_t = torch.as_tensor(self.obs[idx], device=device)      # uint8
        act_t = torch.as_tensor(self.actions[idx], dtype=torch.long, device=device)
        return obs_t, act_t


# ---------------------------------------------------------------------------
# HybridPPO
# ---------------------------------------------------------------------------

class HybridPPO(PPO):
    """
    PPO with a concurrent BC auxiliary loss injected into every gradient step.

    Extra constructor args (all others are forwarded to PPO):
        demo_dir       : path to directory of episode_*.npz demo files.
        bc_coef_start  : BC loss weight at the very start of training (default 0.5).
        bc_coef_end    : BC loss weight at the very end of training  (default 0.05).
        bc_batch_size  : demo minibatch size per PPO gradient step   (default 256).

    If demo_dir is None, behaves exactly like vanilla PPO (no BC loss).
    """

    def __init__(
        self,
        *args,
        demo_dir: Optional[str | Path] = None,
        bc_coef_start: float = 0.5,
        bc_coef_end: float = 0.05,
        bc_batch_size: int = 256,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bc_coef_start = bc_coef_start
        self.bc_coef_end = bc_coef_end
        self.bc_batch_size = bc_batch_size
        self._demo_sampler: Optional[DemoSampler] = (
            DemoSampler(demo_dir) if demo_dir is not None else None
        )

    def _current_bc_coef(self) -> float:
        """Linear schedule: bc_coef_start at progress=1.0 -> bc_coef_end at 0.0."""
        progress = self._current_progress_remaining  # 1.0 -> 0.0 over training
        return self.bc_coef_end + (self.bc_coef_start - self.bc_coef_end) * progress

    # ------------------------------------------------------------------
    # Override train() to inject BC loss.
    # This is a copy of SB3 1.8.0 PPO.train() with a single extra block
    # (marked with "# ---- BC AUX LOSS") after the main PPO loss is built.
    # ------------------------------------------------------------------
    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses = [], [], []
        bc_losses: list[float] = []
        clip_fractions = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())

                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # ---- BC AUX LOSS ----
                bc_coef = self._current_bc_coef()
                if self._demo_sampler is not None and bc_coef > 0:
                    bc_obs, bc_acts = self._demo_sampler.sample(self.bc_batch_size, self.device)
                    _, bc_log_prob, _ = self.policy.evaluate_actions(bc_obs, bc_acts)
                    bc_loss = -bc_log_prob.mean()
                    loss = loss + bc_coef * bc_loss
                    bc_losses.append(bc_loss.item())
                # ---- end BC AUX LOSS ----

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if bc_losses:
            self.logger.record("train/bc_loss", np.mean(bc_losses))
            self.logger.record("train/bc_coef", self._current_bc_coef())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
