"""
Behaviour Cloning for Super Mario Bros.

Trains a CNN policy to predict human actions from preprocessed observations.
The architecture mirrors SB3's NatureCNN so that BC-trained weights can be
transplanted into a PPO policy in Phase 4 without any compatibility hacks.

Key design choices (each affects BC quality a lot):
  1. **Class imbalance handling.** Human Mario play is dominated by "right"
     and "right+A". Raw cross-entropy loss learns to output those two actions
     and ignore everything else. We use a class-weighted loss where weights
     are inversely proportional to action frequency in the dataset.
  2. **Stratified train/val split.** We split per-episode (not per-frame) to
     avoid leakage: adjacent frames are near-duplicates, so per-frame split
     produces artificially high val accuracy.
  3. **Early stopping on val loss.** Without it, BC overfits hard after ~5-10
     epochs on a 20-minute dataset.

Usage (via scripts/train_bc.py, which wraps this module).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Architecture: NatureCNN (matches SB3 defaults so weights are transplantable)
# ---------------------------------------------------------------------------
class NatureCNNPolicy(nn.Module):
    """
    CNN + linear head, matching the architecture SB3 uses inside CnnPolicy.
    Input : (B, 4, 84, 84) uint8 observations (we'll /255 inside forward).
    Output: logits over n_actions discrete actions.
    """

    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        # For 84x84 input with this conv stack the flattened dim is 64*7*7 = 3136
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            feat_dim = self.features(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(inplace=True),
        )
        self.action_head = nn.Linear(512, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs comes in as uint8; convert to float and scale
        x = obs.float() / 255.0
        x = self.features(x)
        x = self.fc(x)
        return self.action_head(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DemoDataset(Dataset):
    """
    Loads one or more .npz demo files into a contiguous array of (obs, action)
    pairs. Keeps everything in CPU memory because even 1hr of demos is only
    ~50k frames at 84x84x4 uint8 = ~140 MB, which fits comfortably.
    """

    def __init__(self, episode_files: list[Path]):
        self.episode_files = list(episode_files)
        obs_chunks: list[np.ndarray] = []
        act_chunks: list[np.ndarray] = []

        for f in self.episode_files:
            data = np.load(f, allow_pickle=True)
            obs_chunks.append(data["observations"])
            act_chunks.append(data["actions"])

        if not obs_chunks:
            raise RuntimeError("No demo files provided (empty list)")

        self.observations = np.concatenate(obs_chunks, axis=0)   # (N, 4, 84, 84)
        self.actions = np.concatenate(act_chunks, axis=0)        # (N,)

        assert self.observations.shape[0] == self.actions.shape[0]

    def __len__(self) -> int:
        return self.actions.shape[0]

    def __getitem__(self, idx: int):
        return self.observations[idx], int(self.actions[idx])


def split_episodes(episode_files: list[Path], val_frac: float = 0.15, seed: int = 0):
    """Episode-level split so consecutive frames don't leak across train/val."""
    rng = np.random.RandomState(seed)
    files = list(episode_files)
    rng.shuffle(files)
    n_val = max(1, int(round(len(files) * val_frac)))
    val_files = files[:n_val]
    train_files = files[n_val:]
    return train_files, val_files


def class_weights(actions: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights, capped to keep rare noise from dominating.
    Returns a tensor of shape (n_classes,).
    """
    counts = Counter(actions.tolist())
    # Start with 1 for every class so any class not present still gets some weight
    freq = np.array([counts.get(c, 0) + 1 for c in range(n_classes)], dtype=np.float64)
    inv = 1.0 / freq
    # Normalise so weights average to 1
    weights = inv * (n_classes / inv.sum())
    # Cap to avoid wild weights when a class appears just once or twice
    weights = np.clip(weights, 0.1, 5.0)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_bc(
    demo_dir: Path,
    n_actions: int,
    save_path: Path,
    batch_size: int = 128,
    lr: float = 3e-4,
    epochs: int = 30,
    val_frac: float = 0.15,
    early_stop_patience: int = 4,
    device: str = "auto",
    seed: int = 0,
    use_class_weights: bool = True,
    log_fn=print,
):
    """
    Train a BC policy on all .npz files in demo_dir.

    Saves the best-by-val-loss model state dict to save_path.
    """
    demo_dir = Path(demo_dir)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Gather episode files
    episode_files = sorted(demo_dir.glob("episode_*.npz"))
    if not episode_files:
        raise RuntimeError(f"No demo files found in {demo_dir}")
    log_fn(f"[bc] found {len(episode_files)} episode files in {demo_dir}")

    train_files, val_files = split_episodes(episode_files, val_frac=val_frac, seed=seed)
    log_fn(f"[bc] train episodes: {len(train_files)}, val episodes: {len(val_files)}")

    train_ds = DemoDataset(train_files)
    val_ds = DemoDataset(val_files)
    log_fn(f"[bc] train frames: {len(train_ds)}, val frames: {len(val_ds)}")

    # Report action distribution
    counts = Counter(train_ds.actions.tolist())
    total = sum(counts.values())
    log_fn("[bc] train action distribution:")
    for a in range(n_actions):
        c = counts.get(a, 0)
        log_fn(f"     action {a}: {c:>6d}  ({100*c/total:5.1f}%)")

    # Loss: cross-entropy, optionally weighted
    if use_class_weights:
        w = class_weights(train_ds.actions, n_actions).to(device_t)
        log_fn(f"[bc] using class weights: {w.cpu().numpy().round(3).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    # Model + optim
    model = NatureCNNPolicy(n_actions=n_actions).to(device_t)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # Dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
    )

    best_val_loss = float("inf")
    patience_left = early_stop_patience
    history = []

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss_sum, train_correct, train_n = 0.0, 0, 0
        for obs, act in train_loader:
            obs = obs.to(device_t, non_blocking=True)
            act = act.to(device_t, non_blocking=True)

            logits = model(obs)
            loss = criterion(logits, act)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            bs = obs.size(0)
            train_loss_sum += float(loss.item()) * bs
            train_correct += int((logits.argmax(dim=1) == act).sum().item())
            train_n += bs

        train_loss = train_loss_sum / train_n
        train_acc = train_correct / train_n

        # ---- Val ----
        model.eval()
        val_loss_sum, val_correct, val_n = 0.0, 0, 0
        with torch.no_grad():
            for obs, act in val_loader:
                obs = obs.to(device_t, non_blocking=True)
                act = act.to(device_t, non_blocking=True)
                logits = model(obs)
                loss = criterion(logits, act)
                bs = obs.size(0)
                val_loss_sum += float(loss.item()) * bs
                val_correct += int((logits.argmax(dim=1) == act).sum().item())
                val_n += bs
        val_loss = val_loss_sum / val_n
        val_acc = val_correct / val_n

        log_fn(
            f"  epoch {epoch:2d}  "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f}  |  "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # ---- Early stopping on val_loss ----
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_left = early_stop_patience
            torch.save({
                "state_dict": model.state_dict(),
                "n_actions": n_actions,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, save_path)
            log_fn(f"     new best val_loss, saved to {save_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                log_fn(f"[bc] early stop after {epoch} epochs (no val improvement)")
                break

    return history
