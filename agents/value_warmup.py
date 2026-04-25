"""
Value-head warmup for BC -> PPO transfer.

After transplanting BC weights into PPO, only `value_net` is randomly
initialised. PPO's advantage = (return - value); a noisy value head
produces noisy advantages that destroy BC's policy in the first few
updates. Warmup pretrains value_net on rollouts collected by the frozen
BC-transplanted policy.

Implementation notes:
    - Uses SB3's own `RolloutBuffer` and `model.collect_rollouts` so episodic
      auto-reset is handled correctly. Driving the env manually from outside
      SB3 leaks reset semantics and crashes nes-py with
      "cannot step in a done environment".
    - During warmup, every parameter is frozen except value_net. SB3's
      collect_rollouts uses no-grad inference so the freezing doesn't matter
      for collection; it matters during the gradient pass.
    - We do not run SB3's PPO update (that would touch the policy). We run
      our own minibatch loop over the buffer's stored states + returns and
      update only `value_net`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback


_VALUE_HEAD_KEYS = ("value_net.weight", "value_net.bias")


def _set_value_head_only_trainable(policy):
    for name, param in policy.named_parameters():
        param.requires_grad = name in _VALUE_HEAD_KEYS


def _set_all_trainable(policy):
    for param in policy.parameters():
        param.requires_grad = True


class _NoOpCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True


def _collect_rollout_into_buffer(model: PPO, buffer: RolloutBuffer, n_rollout_steps: int) -> bool:
    """
    Drive the env through SB3's own collection path. Auto-resets dead
    sub-envs correctly and computes bootstrapped returns/advantages.
    """
    buffer.reset()
    cb = _NoOpCallback()
    cb.init_callback(model)
    return bool(
        model.collect_rollouts(
            env=model.get_env(),
            callback=cb,
            rollout_buffer=buffer,
            n_rollout_steps=n_rollout_steps,
        )
    )


def warmup_value_head(
    model: PPO,
    n_warmup_steps: int = 50_000,
    rollout_size: int = 2048,
    minibatch_size: int = 256,
    n_grad_passes_per_rollout: int = 4,
    gamma: Optional[float] = None,
    lr: float = 1e-3,
    log_fn=print,
) -> dict:
    """
    Warmup the value head only. Everything else is frozen.

    Args:
        model         : SB3 PPO with BC weights already transplanted.
        n_warmup_steps: total env steps across the warmup (all envs combined).
        rollout_size  : env steps per rollout, total across n_envs.
                        Per-env steps = rollout_size // n_envs.
    """
    if gamma is None:
        gamma = model.gamma

    env = model.get_env()
    n_envs = env.num_envs
    device = model.device
    policy = model.policy

    steps_per_env = max(1, rollout_size // n_envs)
    rollout_total = steps_per_env * n_envs

    log_fn(
        f"[warmup] starting value-head warmup: "
        f"target ~{n_warmup_steps:,} env steps, "
        f"rollout={rollout_total} ({steps_per_env} per env x {n_envs} envs), "
        f"gamma={gamma}, lr={lr}"
    )

    _set_value_head_only_trainable(policy)
    value_params = [p for n, p in policy.named_parameters() if n in _VALUE_HEAD_KEYS]
    optimiser = torch.optim.Adam(value_params, lr=lr)

    buffer = RolloutBuffer(
        buffer_size=steps_per_env,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        gamma=gamma,
        gae_lambda=model.gae_lambda,
        n_envs=n_envs,
    )

    # SB3's collect_rollouts uses several internal attributes that are normally
    # initialised by model.learn() before its first rollout (model._last_obs,
    # model._last_episode_starts, model.ep_info_buffer, model.num_timesteps,
    # etc.). Since we're calling collect_rollouts directly, run the same setup
    # via _setup_learn. Calling it with total_timesteps=n_warmup_steps gets us
    # the full initialisation; SB3 doesn't enforce that we actually train for
    # exactly that many steps when we invoke collect_rollouts manually.
    setup_cb = _NoOpCallback()
    model._setup_learn(
        total_timesteps=n_warmup_steps,
        callback=setup_cb,
        reset_num_timesteps=True,
        tb_log_name="warmup",
        progress_bar=False,
    )

    history = []
    env_steps_collected = 0
    iteration = 0

    while env_steps_collected < n_warmup_steps:
        iteration += 1

        ok = _collect_rollout_into_buffer(model, buffer, steps_per_env)
        if not ok:
            log_fn("[warmup] collect_rollouts returned False; stopping warmup early")
            break

        epoch_losses: list[float] = []
        for _pass in range(n_grad_passes_per_rollout):
            for batch in buffer.get(minibatch_size):
                values = policy.predict_values(batch.observations).reshape(-1)
                target = batch.returns.reshape(-1)
                loss = F.mse_loss(values, target)

                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                epoch_losses.append(float(loss.item()))

        env_steps_collected += rollout_total
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        ret_t = torch.as_tensor(buffer.returns).reshape(-1).float()
        log_fn(
            f"[warmup]   iter {iteration:2d}  "
            f"env_steps={env_steps_collected:>7,}  "
            f"value_loss={mean_loss:.4f}  "
            f"return_mean={float(ret_t.mean()):+.2f}  "
            f"return_std={float(ret_t.std()):.2f}"
        )
        history.append({
            "iteration": iteration,
            "env_steps": env_steps_collected,
            "value_loss": mean_loss,
            "return_mean": float(ret_t.mean()),
            "return_std": float(ret_t.std()),
        })

    _set_all_trainable(policy)
    log_fn("[warmup] done. all parameters unfrozen for main PPO training.")

    return {
        "history": history,
        "total_env_steps": env_steps_collected,
        "iterations": iteration,
    }
