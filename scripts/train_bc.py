"""
Train a Behaviour Cloning policy on recorded demos.

Usage:
    python scripts/train_bc.py --config configs/bc_default.yaml
    python scripts/train_bc.py --config configs/bc_default.yaml --epochs 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.bc import train_bc
from agents.utils import load_config, set_seeds, write_run_metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/bc_default.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--demo-dir", type=str, default=None,
                   help="Override cfg.bc.demo_dir")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.epochs is not None:
        cfg["bc"]["epochs"] = args.epochs
    if args.demo_dir is not None:
        cfg["bc"]["demo_dir"] = args.demo_dir

    set_seeds(cfg["bc"]["seed"])
    run_dir = ROOT / cfg["bc"]["log_dir"] / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(run_dir, cfg)

    print(f"[train_bc] run dir: {run_dir}")
    save_path = run_dir / "bc_model.pt"

    demo_dir = Path(cfg["bc"]["demo_dir"])
    if not demo_dir.is_absolute():
        demo_dir = ROOT / demo_dir

    history = train_bc(
        demo_dir=demo_dir,
        n_actions=cfg["bc"]["n_actions"],
        save_path=save_path,
        batch_size=cfg["bc"]["batch_size"],
        lr=cfg["bc"]["lr"],
        epochs=cfg["bc"]["epochs"],
        val_frac=cfg["bc"]["val_frac"],
        early_stop_patience=cfg["bc"]["early_stop_patience"],
        device=cfg["bc"].get("device", "auto"),
        seed=cfg["bc"]["seed"],
        use_class_weights=cfg["bc"].get("use_class_weights", True),
    )

    # Save training history for plotting
    with (run_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    print(f"[train_bc] done. best model at {save_path}")


if __name__ == "__main__":
    main()
