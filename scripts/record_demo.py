"""
Record human gameplay of Super Mario Bros for IL.

Controls:
    Arrow keys or WASD : move left / right, duck, look up
    J : A button (jump)
    K : B button (run / fireball)
    ESC: quit recording session
    R : discard current episode and restart
    P : pause/unpause
    Enter: manually end current episode (as if died)


Usage:
    python scripts/record_demo.py --world 1 --stage 1
    python scripts/record_demo.py --world 1 --stage 1 --out data/demos/world1_1/
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Suppress the pygame audio init -- WSL has no ALSA and we don't need sound.
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs import make_env
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT



# Given the keys held this frame, pick the best matching action.
# Priorities: (right > left), (A = jump), (B = run).
def keys_to_action(right: bool, left: bool, a_btn: bool, b_btn: bool) -> int:
    # Right-moving group
    if right:
        if a_btn and b_btn:
            return 4
        if a_btn:
            return 2
        if b_btn:
            return 3
        return 1
    # Left
    if left:
        return 6
    # Jump in place
    if a_btn:
        return 5
    return 0  # NOOP


# ---------------------------------------------------------------------------
# Pygame HUD
# ---------------------------------------------------------------------------
class Recorder:
    # Logical canvas size. The NES frame is 256x240; we upscale 2x to 512x480.
    GAME_W, GAME_H = 512, 480
    HUD_H = 120
    WIN_W, WIN_H = GAME_W, GAME_H + HUD_H

    def __init__(self, world: int, stage: int, out_dir: Path):
        self.world = world
        self.stage = stage
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Build env with the same wrappers as training so state-action alignment
        # is guaranteed. Eval-style (no episodic-life bite) so a single life-loss
        # doesn't cut an episode short while the user is trying to complete the
        # level.
        self.env = make_env(
            world=world,
            stage=stage,
            episodic_life=False,       # let the full 3-life game play out
            reward_shaping=True,       # match training reward curve
        )

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIN_W, self.WIN_H))
        pygame.display.set_caption(
            f"Mario Demo Recorder  -  World {world}-{stage}  -  J=A K=B  R=restart  P=pause  ESC=quit"
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 14)

        # Session state
        self.paused = False
        self.episode_idx = self._count_existing_episodes()
        self.session_started_at = time.time()
        self.session_frames_recorded = 0

    def _count_existing_episodes(self) -> int:
        """Resume episode numbering after any existing demos in out_dir."""
        existing = sorted(self.out_dir.glob("episode_*.npz"))
        if not existing:
            return 0
        # Parse the numeric suffix from each filename
        nums = []
        for p in existing:
            try:
                n = int(p.stem.split("_")[1])
                nums.append(n)
            except (IndexError, ValueError):
                pass
        return (max(nums) + 1) if nums else 0

    # -----------------------------------------------------------------------
    def _read_keys(self) -> tuple[int, dict]:
        """Read current key state, return (action_index, key_flags_for_HUD)."""
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        a_btn = keys[pygame.K_j]           # jump
        b_btn = keys[pygame.K_k]           # run / fireball

        action = keys_to_action(right, left, a_btn, b_btn)
        return action, dict(right=right, left=left, a=a_btn, b=b_btn)

    # -----------------------------------------------------------------------
    def _draw_frame(self, rgb_frame: np.ndarray, info: dict, action: int, keys: dict, hud_extra: str = ""):
        """Render the game frame + HUD to the window."""
        # Game view: the emulator gives us a 240x256 RGB frame
        if rgb_frame is not None:
            # pygame surfaces want (W, H, 3); numpy is (H, W, 3) -> transpose
            surf = pygame.surfarray.make_surface(rgb_frame.swapaxes(0, 1))
            surf = pygame.transform.scale(surf, (self.GAME_W, self.GAME_H))
            self.screen.blit(surf, (0, 0))
        else:
            self.screen.fill((20, 20, 20), rect=(0, 0, self.GAME_W, self.GAME_H))

        # HUD background
        pygame.draw.rect(self.screen, (25, 25, 30), (0, self.GAME_H, self.WIN_W, self.HUD_H))

        action_name = SIMPLE_MOVEMENT[action] if action < len(SIMPLE_MOVEMENT) else ["?"]
        action_str = "+".join(action_name) if action_name != ["NOOP"] else "NOOP"

        lines = [
            f"Episode #{self.episode_idx}   Session frames: {self.session_frames_recorded}",
            f"action: {action:>2}  ({action_str})",
            f"x_pos={info.get('x_pos', '?')}  score={info.get('score', '?')}  life={info.get('life', '?')}  coins={info.get('coins', '?')}",
        ]
        if hud_extra:
            lines.append(hud_extra)

        for i, line in enumerate(lines):
            surf = self.font.render(line, True, (230, 230, 230))
            self.screen.blit(surf, (10, self.GAME_H + 8 + i * 22))

        # Key-press indicator (bottom right)
        pressed_str = (
            ("<" if keys["left"] else "-") +
            (">" if keys["right"] else "-") +
            ("A" if keys["a"] else "-") +
            ("B" if keys["b"] else "-")
        )
        surf = self.font.render(pressed_str, True, (120, 220, 120) if any(keys.values()) else (90, 90, 90))
        self.screen.blit(surf, (self.WIN_W - 90, self.GAME_H + 8))

        if self.paused:
            surf = self.font.render("PAUSED  (press P)", True, (255, 220, 60))
            self.screen.blit(surf, (self.WIN_W - 230, self.GAME_H + 40))

        pygame.display.flip()

    # -----------------------------------------------------------------------
    def _handle_session_events(self) -> Optional[str]:
        """Check discrete key events (press, not hold). Returns a command or None."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.key == pygame.K_r:
                    return "restart"
                if event.key == pygame.K_p:
                    return "pause"
                if event.key == pygame.K_RETURN:
                    return "end_episode"
        return None

    # -----------------------------------------------------------------------
    def _save_episode(
        self,
        observations: list,
        actions: list,
        rewards: list,
        dones: list,
        final_info: dict,
    ) -> Path:
        path = self.out_dir / f"episode_{self.episode_idx:04d}.npz"
        np.savez_compressed(
            path,
            observations=np.stack(observations, axis=0).astype(np.uint8),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=bool),
            world=self.world,
            stage=self.stage,
            x_pos=final_info.get("x_pos", 0),
            flag_get=final_info.get("flag_get", False),
            score=final_info.get("score", 0),
        )
        return path

    # -----------------------------------------------------------------------
    def run(self):
        """Main session loop. Runs one episode per iteration until user quits."""
        print(f"[record] output dir: {self.out_dir}")
        print(f"[record] starting at episode index {self.episode_idx}")
        print("[record] controls: arrows/WASD move, J=jump, K=run/fireball, R=restart, P=pause, ESC=quit, Enter=end")

        running = True
        while running:
            obs = self.env.reset()
            # Grab an rgb frame for the preview. render() goes through wrappers
            # but most pass through; the underlying NES env returns RGB.
            rgb = self.env.render(mode="rgb_array")
            last_info: dict = {}

            observations = [obs]
            actions: list[int] = []
            rewards: list[float] = []
            dones: list[bool] = []

            done = False
            discard = False

            # Give user a brief moment to get ready on each new episode
            start_countdown = time.time() + 1.0
            while time.time() < start_countdown:
                self._draw_frame(rgb, {}, 0, dict(right=False, left=False, a=False, b=False),
                                 hud_extra="Get ready...")
                cmd = self._handle_session_events()
                if cmd == "quit":
                    running = False
                    break
                self.clock.tick(15)
            if not running:
                break

            # Main episode loop
            while not done:
                cmd = self._handle_session_events()
                if cmd == "quit":
                    running = False
                    discard = True
                    break
                if cmd == "restart":
                    discard = True
                    break
                if cmd == "pause":
                    self.paused = not self.paused
                if cmd == "end_episode":
                    done = True  # user ends early
                    break

                if self.paused:
                    self._draw_frame(rgb, last_info, 0,
                                     dict(right=False, left=False, a=False, b=False))
                    self.clock.tick(15)
                    continue

                action, keys = self._read_keys()
                obs, reward, done, info = self.env.step(action)
                rgb = self.env.render(mode="rgb_array")

                observations.append(obs)
                actions.append(int(action))
                rewards.append(float(reward))
                dones.append(bool(done))
                last_info = info
                self.session_frames_recorded += 1

                self._draw_frame(rgb, info, action, keys)
                # ~15 FPS matches the agent's decision rate (frame_skip=4 at 60fps)
                self.clock.tick(15)

            # Episode loop ended: decide whether to save or discard
            if discard:
                print(f"  [ep {self.episode_idx}] discarded ({len(actions)} frames)")
                # Don't advance episode_idx on discard
                continue

            if len(actions) < 10:
                print(f"  [ep {self.episode_idx}] too short ({len(actions)} frames), skipping")
                continue

            # observations currently has T+1 entries (initial obs + one per step).
            # For BC we want exactly T frames paired with T actions. Drop the final obs.
            observations = observations[:-1]
            path = self._save_episode(observations, actions, rewards, dones, last_info)
            print(
                f"  [ep {self.episode_idx}] saved {len(actions)} frames  "
                f"x_pos={last_info.get('x_pos', '?')}  "
                f"flag={last_info.get('flag_get', False)}  "
                f"-> {path.name}"
            )
            self.episode_idx += 1

        # Session wrap-up
        self.env.close()
        pygame.quit()
        dur = time.time() - self.session_started_at
        print(
            f"\n[record] session done. duration={dur:.0f}s, "
            f"{self.session_frames_recorded} total frames recorded, "
            f"{self.episode_idx} episodes saved."
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--out", type=str, default=None,
                   help="Output directory. Default: data/demos/world<W>_<S>/")
    args = p.parse_args()

    out_dir = Path(args.out) if args.out else (ROOT / "data" / "demos" / f"world{args.world}_{args.stage}")
    Recorder(args.world, args.stage, out_dir).run()


if __name__ == "__main__":
    main()
