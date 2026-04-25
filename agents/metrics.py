"""
Shared metric helpers for evaluation.

Project success hierarchy (decided in Phase 3):
    1. Completion rate         (% of episodes that reach the flag)
    2. Time-to-flag, on completed runs only (mean episode length)
    3. Final x-position        (diagnostic for incomplete runs)

Score is intentionally ignored.

This module exists so that scripts/evaluate.py (PPO) and scripts/play_bc.py
(BC) print the same numbers in the same format. Anything that needs a
"summary block" should call summarise().
"""
from __future__ import annotations

import statistics as _st
from typing import Optional


def time_to_flag(episode_record: dict) -> Optional[int]:
    """Length of the episode if it reached the flag, else None."""
    if episode_record.get("flag_get", False):
        return int(episode_record["length"])
    return None


def annotate(records: list[dict]) -> list[dict]:
    """Add `time_to_flag` field to each record (None if not completed)."""
    for r in records:
        r["time_to_flag"] = time_to_flag(r)
    return records


def summarise(records: list[dict], label: str = "") -> dict:
    """
    Print a project-aligned summary block. Returns the computed stats dict.

    Output prioritises completion rate, then conditional time, then x-pos.
    """
    if not records:
        print("  (no records)")
        return {}

    n = len(records)
    n_flag = sum(1 for r in records if r["flag_get"])
    completion_rate = n_flag / n

    # Time-to-flag: only on completed episodes
    completed_lengths = [r["length"] for r in records if r["flag_get"]]
    if completed_lengths:
        ttf_mean = _st.mean(completed_lengths)
        ttf_median = _st.median(completed_lengths)
        ttf_min = min(completed_lengths)
    else:
        ttf_mean = ttf_median = ttf_min = None

    # X-pos: diagnostic on incomplete runs (and overall)
    xs = [r["final_x_pos"] for r in records]
    incomplete_xs = [r["final_x_pos"] for r in records if not r["flag_get"]]

    # Pretty print
    if label:
        print(f"  -- {label} --")
    print(
        f"  completion: {n_flag}/{n}  ({100*completion_rate:.0f}%)"
    )
    if completed_lengths:
        print(
            f"  time2flag:  mean={ttf_mean:.0f}  median={ttf_median:.0f}  "
            f"best={ttf_min}  (frames, on {len(completed_lengths)} completed)"
        )
    else:
        print("  time2flag:  -- (no completed runs)")

    if incomplete_xs:
        print(
            f"  x_pos (incomplete): mean={_st.mean(incomplete_xs):.0f}  "
            f"median={_st.median(incomplete_xs):.0f}  max={max(incomplete_xs)}"
        )
    print(
        f"  x_pos (all):        mean={_st.mean(xs):.0f}  "
        f"median={_st.median(xs):.0f}  max={max(xs)}"
    )

    return {
        "n_episodes": n,
        "completion_rate": completion_rate,
        "time_to_flag_mean": ttf_mean,
        "time_to_flag_median": ttf_median,
        "time_to_flag_best": ttf_min,
        "x_pos_mean_all": _st.mean(xs),
        "x_pos_max": max(xs),
    }
