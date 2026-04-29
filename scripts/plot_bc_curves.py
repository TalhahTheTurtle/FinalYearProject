"""
Plot BC training curves (loss and accuracy) from a history.json file.

Usage:
    python scripts/plot_bc_curves.py --history logs/bc_no_class_weights/history.json
    python scripts/plot_bc_curves.py --history logs/bc_no_class_weights/history.json --out report/bc_curves.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=str, required=True)
    p.add_argument("--out", type=str, default=None, help="Output path. Default: same dir as history.")
    args = p.parse_args()

    history_path = Path(args.history)
    with history_path.open() as f:
        history = json.load(f)

    epochs      = [e["epoch"]      for e in history]
    train_loss  = [e["train_loss"] for e in history]
    val_loss    = [e["val_loss"]   for e in history]
    train_acc   = [e["train_acc"]  for e in history]
    val_acc     = [e["val_acc"]    for e in history]

    best_epoch = min(history, key=lambda e: e["val_loss"])["epoch"]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Behaviour Cloning Training Curves", fontsize=13, fontweight="bold")

    # ---- Loss ----
    ax_loss.plot(epochs, train_loss, label="Train loss", color="#4C72B0", linewidth=1.8)
    ax_loss.plot(epochs, val_loss,   label="Val loss",   color="#DD8452", linewidth=1.8)
    ax_loss.axvline(best_epoch, color="grey", linestyle="--", linewidth=1, label=f"Best val (epoch {best_epoch})")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.legend(framealpha=0.9)
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_loss.grid(True, alpha=0.3)

    # ---- Accuracy ----
    ax_acc.plot(epochs, [a * 100 for a in train_acc], label="Train acc", color="#4C72B0", linewidth=1.8)
    ax_acc.plot(epochs, [a * 100 for a in val_acc],   label="Val acc",   color="#DD8452", linewidth=1.8)
    ax_acc.axvline(best_epoch, color="grey", linestyle="--", linewidth=1, label=f"Best val (epoch {best_epoch})")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Action Prediction Accuracy")
    ax_acc.legend(framealpha=0.9)
    ax_acc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_acc.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()

    out = Path(args.out) if args.out else history_path.parent / "bc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
