"""
making environments with preprocessing

make_env(world=1, stage=1,action_set = "simple")
obs = env.reset()

for the PPO training ill use a vectorised env
vec = make_vec_env(n_envs=8, world=1,stage=1)





"""
from __future__ import annotations

from typing import Optional

import gym
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from .wrappers import SkipFrame, EpisodicLifeEnv, MarioRewardShaping, LazyFramesToArray, StuckDetector


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
    flag_bonus: float = 50.0,
    step_penalty: float = 0.0,
    death_penalty: float = -15.0,
    stuck_threshold: int = 30,
    seed: Optional[int] = None,
) -> gym.Env:
    """
    Building a single wrapped Super Mario Bros environment.

    Parameters
    ----------
    world, stage : which level (e.g. world=1, stage=1 -> SuperMarioBros-1-1-v0)

    version      : 0 = standard, 1 = downsampled, 2 = pixel, 3 = rectangle.
                   Use 0 for training, 3 if you want a radically simpler
                   visual input for faster prototyping.

    action_set   : "right_only" | "simple" | "complex"

    frame_skip   : action-repeat factor (4 because standard)

    frame_stack  : number of stacked frames (4, thats the standarad

    resize_shape : H and W after resize (84 because standard)

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
    env = MarioRewardShaping(
        env,
        enabled=reward_shaping,
        flag_bonus=flag_bonus,
        step_penalty=step_penalty,
        death_penalty=death_penalty,
    )
    env = StuckDetector(env, stuck_threshold=stuck_threshold)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=resize_shape)
    env = FrameStack(env, num_stack=frame_stack)
    env = LazyFramesToArray(env)

    return env


def _make_single_env_fn(rank: int, base_seed: int, env_kwargs: dict):
    """
    Return a zero-arg callable that builds a single wrapped Mario env.
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

    SubprocVecEnv runs each env in its own process
    """
    fns = [
        _make_single_env_fn(rank=i, base_seed=base_seed, env_kwargs=env_kwargs)
        for i in range(n_envs)
    ]
    vec_cls = SubprocVecEnv if use_subproc else DummyVecEnv
    vec = vec_cls(fns)
    vec = VecMonitor(vec)  # logs episode returns/lengths
    return vec
