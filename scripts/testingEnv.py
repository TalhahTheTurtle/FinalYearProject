

import argparse
import sys
from pathlib import Path

import numpy as np


"""
running a random agent for one episode and save a GIF
I verify that:
1. The Mario env installs and loads

2. The wrapper pipeline produces the expected observation shape
3. Episodes terminate cleanly
4. We can render and save a video

Run:
python scripts/testingEnv.py --world 1 --stage 1

testing script, this should work
"""

# Make the project importable when running the script directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs import make_env  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--action-set", choices=["right_only", "simple", "complex"], default="simple")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-gif", type=str, default="logs/sanity_check.gif")
    p.add_argument("--no-gif", action="store_true", help="Skip GIF rendering (faster)")
    args = p.parse_args()

    print(f"Building env: World {args.world}-{args.stage}, actions={args.action_set}")
    env = make_env(
        world=args.world,
        stage=args.stage,
        action_set=args.action_set,
        seed=args.seed,
    )
    print(f"  observation_space: {env.observation_space}")
    print(f"  action_space:      {env.action_space}")

    obs = env.reset()
    print(f"  obs.shape={obs.shape}, obs.dtype={obs.dtype}")
    assert obs.dtype == np.uint8, f"Unexpected obs dtype: {obs.dtype}"
    assert obs.ndim == 3, f"Expected 3D obs (stack, H, W), got {obs.shape}"

    # For the GIF need raw RGB frames, the wrapped env returns preprocessed.
    # We access the underlying render() method which still returns the original.
    frames = []
    total_reward = 0.0
    done = False
    steps = 0

    while not done and steps < args.max_steps:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        if not args.no_gif:
            # .render(mode="rgb_array") goes through all wrappers but most are no-op for render
            frame = env.render(mode="rgb_array")
            if frame is not None:
                frames.append(frame)

    env.close()

    print(f"\nEpisode finished after {steps} steps")
    print(f"  total reward: {total_reward:.2f}")
    print(f"  final info:   x_pos={info.get('x_pos')}, flag_get={info.get('flag_get')}, life={info.get('life')}")

    if not args.no_gif and frames:
        import imageio
        out_path = ROOT / args.save_gif
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Downsample every 2nd frame to keep GIF small
        imageio.mimsave(out_path, frames[::2], fps=30)
        print(f"  GIF saved to: {out_path}")

    print("WORKING")


if __name__ == "__main__":
    main()
