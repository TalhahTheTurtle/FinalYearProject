"""
Environment wrappers for Super Mario Bros.

The wrapping order matters. We apply them in this sequence:

    raw NES env  (240x256 RGB, 60 FPS, ~256 action combos)
        |
        v  JoypadSpace          -> discrete action set (7 or 12 actions)
        v  SkipFrame            -> repeat action N frames, return max-pooled obs (reduces decision freq)
        v  EpisodicLife         -> end episode on life loss (speeds up learning signal)
        v  RewardShaping        -> custom reward on top of gym-super-mario-bros's built-in
        v  GrayScaleObservation -> RGB -> grayscale
        v  ResizeObservation    -> 240x256 -> 84x84
        v  FrameStack           -> stack 4 consecutive frames -> shape (4, 84, 84)

The final observation is what the CNN policy sees. 84x84 grayscale + 4-frame stack is
the NatureCNN input shape from Mnih et al. (2015).

Author: Talhah Anwar
"""
from __future__ import annotations

import numpy as np
import gym
from gym import spaces
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


# ---------------------------------------------------------------------------
# Action repeat / frame skip
# ---------------------------------------------------------------------------
class SkipFrame(gym.Wrapper):
    """
    Repeat the chosen action for `skip` frames and return the max-pooled
    observation across the last two frames.

    Why max-pool? NES games (like Atari) flicker sprites every other frame,
    so a single frame can drop enemies/items. Max-pooling the last two frames
    of the skip window avoids that.

    Rewards are summed across the skipped frames so no reward signal is lost.
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info


# ---------------------------------------------------------------------------
# Life-based episode termination
# ---------------------------------------------------------------------------
class EpisodicLifeEnv(gym.Wrapper):
    """
    End the episode (from the agent's perspective) on any life loss,
    but only actually reset the underlying env when lives reach zero.

    This is the standard Atari trick. It gives a much denser learning signal:
    Mario dying to an enemy is now a clear terminal negative event, rather
    than being absorbed into a longer multi-life episode.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._lives = 0
        self._was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._was_real_done = done
        # gym-super-mario-bros exposes `life` in info
        lives = info.get("life", self._lives)
        if 0 < lives < self._lives:
            done = True
        self._lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self._was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # No-op step to advance past the death animation
            obs, _, _, info = self.env.step(0)
            self._lives = info.get("life", self._lives)
        self._lives = self.env.unwrapped._life if hasattr(self.env.unwrapped, "_life") else self._lives
        return obs


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------
class MarioRewardShaping(gym.Wrapper):
    """
    On top of gym-super-mario-bros' built-in reward, add shaping signals
    that speed up early learning.

    The built-in reward is already reasonable:
        r = (x_pos change) + (clock penalty) + (death penalty)
    capped to [-15, 15] per step.

    We add:
        - small per-step time penalty to discourage idling
        - bonus for flag-get (level completion)
        - penalty for falling into pits (detected via y-position)

    All shaping magnitudes are kept small relative to the built-in reward
    so the agent still optimises mostly for the original signal. If you want
    to ablate reward shaping in the final report, pass `enabled=False`.
    """

    def __init__(
        self,
        env: gym.Env,
        enabled: bool = True,
        flag_bonus: float = 50.0,
        step_penalty: float = 0.0,
        death_penalty: float = -15.0,
    ):
        super().__init__(env)
        self.enabled = enabled
        self.flag_bonus = flag_bonus
        self.step_penalty = step_penalty
        self.death_penalty = death_penalty
        self._prev_life = None

    def reset(self, **kwargs):
        self._prev_life = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self.enabled:
            # Constant small penalty per step (off by default)
            reward -= self.step_penalty

            # Big bonus for reaching the flag
            if info.get("flag_get", False):
                reward += self.flag_bonus

            # Explicit death penalty (in addition to built-in), based on life drop
            life = info.get("life", 2)
            if self._prev_life is None:
                self._prev_life = life
            if life < self._prev_life:
                reward += self.death_penalty
            self._prev_life = life

        return obs, reward, done, info


# ---------------------------------------------------------------------------
# Observation conversion for SB3 compatibility
# ---------------------------------------------------------------------------
class LazyFramesToArray(gym.ObservationWrapper):
    """
    `FrameStack` returns `LazyFrames` objects. SB3 expects numpy arrays.
    Also transpose to (C, H, W) because that's what torch CNNs want.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space = env.observation_space
        # FrameStack shape is (n_stack, H, W) after GrayScale; we keep that order
        # because SB3's NatureCNN policy expects (C, H, W).
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=old_space.shape,
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.asarray(observation, dtype=np.uint8)
