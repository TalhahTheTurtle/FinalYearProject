"""
Factory for building Super Mario Bros environments with a consistent
preprocessing pipeline.

Usage:
    env = make_env(world=1, stage=1, action_set="simple")
    obs = env.reset()  # shape (4, 84, 84), uint8

For PPO training you'll want a vectorised env:
    vec = make_vec_env(n_envs=8, world=1, stage=1)

IMPORTANT: the *same* `make_env` function must be used for PPO training,
BC training, demo recording, and evaluation. If the preprocessing differs
between phases, observations drift and BC-pretrained weights become useless.

Author: Talhah Anwar
"""
from __future__ import annotations

from typing import Optional

import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from .wrappers import SkipFrame, EpisodicLifeEnv, MarioRewardShaping, LazyFramesToArray


ACTION_SETS = {
    "right_only": RIGHT_ONLY,      # 5 actions, easiest
    "simple": SIMPLE_MOVEMENT,     # 7 actions, standard
    "complex": COMPLEX_MOVEMENT,   # 12 actions, hardest
}


def make_env(
    world: int = 1,
    stage: int = 1,
    version: int = 0,
    action_set: str = "simple",
    frame_skip: int = 4,
    frame_stack: int = 4,
    resize_shape: int = 84,
    reward_shaping: bool = True,
    episodic_life: bool = True,
    seed: Optional[int] = None,
) -> gym.Env:
    """
    Build a single wrapped Super Mario Bros environment.

    Parameters
    ----------
    world, stage : which level (e.g. world=1, stage=1 -> SuperMarioBros-1-1-v0)
    version      : 0 = standard, 1 = downsampled, 2 = pixel, 3 = rectangle.
                   Use 0 for training; use 3 if you want a radically simpler
                   visual input for faster prototyping.
    action_set   : "right_only" | "simple" | "complex"
    frame_skip   : action-repeat factor (4 is standard)
    frame_stack  : number of stacked frames (4 is standard, gives velocity info)
    resize_shape : H and W after resize (84 is standard)
    reward_shaping, episodic_life : toggle wrappers for ablations
    seed         : integer seed (optional, for reproducibility)
    """
    if action_set not in ACTION_SETS:
        raise ValueError(f"action_set must be one of {list(ACTION_SETS)}")

    env_id = f"SuperMarioBros-{world}-{stage}-v{version}"
    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, ACTION_SETS[action_set])

    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)

    env = SkipFrame(env, skip=frame_skip)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    env = MarioRewardShaping(env, enabled=reward_shaping)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=resize_shape)
    env = FrameStack(env, num_stack=frame_stack)
    env = LazyFramesToArray(env)

    return env


def _make_single_env_fn(rank: int, base_seed: int, env_kwargs: dict):
    """
    Return a zero-arg callable that builds a single wrapped Mario env.

    SubprocVecEnv requires picklable zero-arg factories. Using a module-level
    factory function (not a nested closure that captures locals) is the safe
    pattern across Python versions and WSL.
    """
    def _init():
        return make_env(seed=base_seed + rank, **env_kwargs)
    return _init


def make_vec_env(
    n_envs: int = 8,
    use_subproc: bool = True,
    base_seed: int = 0,
    **env_kwargs,
):
    """
    Build a vectorised env for PPO training.

    SubprocVecEnv runs each env in its own process, which is what you want
    for PPO on CPU-bound envs like Mario. On WSL this usually works fine.
    If you hit pickling issues, set use_subproc=False to fall back to DummyVecEnv.
    """
    fns = [
        _make_single_env_fn(rank=i, base_seed=base_seed, env_kwargs=env_kwargs)
        for i in range(n_envs)
    ]
    vec_cls = SubprocVecEnv if use_subproc else DummyVecEnv
    vec = vec_cls(fns)
    vec = VecMonitor(vec)  # logs episode returns/lengths
    return vec
