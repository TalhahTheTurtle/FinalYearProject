"""
Utility helpers: config loading, run setup, reproducibility.

The reason this exists as its own module: every script (train_ppo, evaluate,
eventually train_bc, train_hybrid) needs the same three things:
    1. Load a YAML config and optionally merge CLI overrides.
    2. Set seeds deterministically.
    3. Capture a metadata snapshot (git commit, system info, config hash) so
       that months from now you can tell which code + which hyperparameters
       produced a particular log directory.

All three go in the report as part of methodology.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> dict:
    """Load a YAML config into a plain dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a dict")
    return cfg


def config_hash(cfg: dict) -> str:
    """
    Short stable hash of a config, useful for naming runs that only differ
    in hyperparameters. Order-independent on nested dicts.
    """
    serialised = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialised).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seeds(seed: int) -> None:
    """Seed Python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode slows things down a lot; don't enable by default.
    # Leave this commented unless you explicitly need bit-exact reproducibility.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# System / git metadata
# ---------------------------------------------------------------------------
def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.decode().strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return bool(out.strip())
    except Exception:
        return None


def system_info() -> dict:
    """Snapshot of system info. Written to logs for reproducibility."""
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_name,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
    }


def write_run_metadata(run_dir: Path, cfg: dict) -> None:
    """Dump config + system info + config hash into run_dir/metadata.json."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": cfg,
        "config_hash": config_hash(cfg),
        "system": system_info(),
    }
    with (run_dir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2, default=str)
    # Also write the raw config for easy re-use
    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
