"""
BC -> PPO weight transplant.

Loads a Behaviour Cloning checkpoint (NatureCNNPolicy state dict) and copies
the convolutional + MLP-projection + action-head weights into a freshly
constructed SB3 PPO model. The PPO value head is left at default
initialisation: BC has no value function, so PPO learns the value branch
from its own rollouts via the value-loss term.

Why this exists as a separate module:
    - SB3 wraps the CNN inside `policy.features_extractor` and the action
      head inside `policy.action_net`. Naming differs from our flat BC model.
    - We map parameters explicitly (not by `load_state_dict(strict=False)`)
      because silent-mismatch is the worst possible failure mode here. If
      SB3 changes internals, this should crash with a useful error rather
      than train PPO with random conv weights.
    - For Phase 6 ablations, we want a single function that we can call
      from multiple training scripts.

Compatible with stable-baselines3 1.8.0 + CnnPolicy on (4, 84, 84) input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from stable_baselines3 import PPO


# ---------------------------------------------------------------------------
# The mapping: BC keys -> PPO keys
# ---------------------------------------------------------------------------
# BC keys (from agents.bc.NatureCNNPolicy):
#   features.0.weight, features.0.bias    Conv2d 4->32 k=8 s=4
#   features.2.weight, features.2.bias    Conv2d 32->64 k=4 s=2
#   features.4.weight, features.4.bias    Conv2d 64->64 k=3 s=1
#   fc.0.weight,       fc.0.bias          Linear 3136->512
#   action_head.weight, action_head.bias  Linear 512->n_actions
#
# SB3 PPO with CnnPolicy keys (1.8.x):
#   In SB3 1.8 with default CnnPolicy (whether or not share_features_extractor
#   is set in policy_kwargs), the policy state dict can contain THREE feature
#   extractors:
#     features_extractor.*     (the shared / canonical extractor)
#     pi_features_extractor.*  (policy/action branch alias)
#     vf_features_extractor.*  (value branch alias)
#   We transplant BC's conv+MLP weights into ALL THREE. When share is on at
#   runtime they all tie to the same forward pass; when it is not (or when
#   SB3's wiring deviates), each branch still uses correct weights.
#   The action head (`action_net`) is also transplanted directly. The value
#   head (`value_net`) is left at PPO default since BC has no value function.

_BC_TO_PPO_FEATURE_DESTINATIONS = {
    "features.0.weight":  ["features_extractor.cnn.0.weight",     "pi_features_extractor.cnn.0.weight",     "vf_features_extractor.cnn.0.weight"],
    "features.0.bias":    ["features_extractor.cnn.0.bias",       "pi_features_extractor.cnn.0.bias",       "vf_features_extractor.cnn.0.bias"],
    "features.2.weight":  ["features_extractor.cnn.2.weight",     "pi_features_extractor.cnn.2.weight",     "vf_features_extractor.cnn.2.weight"],
    "features.2.bias":    ["features_extractor.cnn.2.bias",       "pi_features_extractor.cnn.2.bias",       "vf_features_extractor.cnn.2.bias"],
    "features.4.weight":  ["features_extractor.cnn.4.weight",     "pi_features_extractor.cnn.4.weight",     "vf_features_extractor.cnn.4.weight"],
    "features.4.bias":    ["features_extractor.cnn.4.bias",       "pi_features_extractor.cnn.4.bias",       "vf_features_extractor.cnn.4.bias"],
    "fc.0.weight":        ["features_extractor.linear.0.weight",  "pi_features_extractor.linear.0.weight",  "vf_features_extractor.linear.0.weight"],
    "fc.0.bias":          ["features_extractor.linear.0.bias",    "pi_features_extractor.linear.0.bias",    "vf_features_extractor.linear.0.bias"],
}

# Action head and the value head (NOT transplanted) live separately.
# Build the final list. Each entry: (BC source key, [PPO destinations]).
# Destinations that aren't actually present in the model at runtime will
# be silently skipped (so the same map works whether the policy has 1, 2,
# or 3 feature extractors -- all real layouts in the SB3 1.x series).
_PARAMETER_MAP = list(_BC_TO_PPO_FEATURE_DESTINATIONS.items()) + [
    ("action_head.weight", ["action_net.weight"]),
    ("action_head.bias",   ["action_net.bias"]),
]


def transplant_bc_into_ppo(bc_ckpt_path: str | Path, ppo_model: PPO, verbose: bool = True) -> dict:
    """
    Copy BC weights into a freshly-constructed PPO model in-place.

    Returns a dict summarising which params were transferred and which were
    left at PPO defaults (the value head primarily).

    Raises:
        KeyError if any expected BC key is missing from the checkpoint
        KeyError if any expected PPO key is missing from the policy state dict
        ValueError on shape mismatch (catches accidental architecture drift)
    """
    bc_ckpt_path = Path(bc_ckpt_path)
    ckpt = torch.load(bc_ckpt_path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise KeyError(f"BC checkpoint at {bc_ckpt_path} has no 'state_dict' key. "
                       f"Top-level keys: {list(ckpt.keys())}")
    bc_sd: dict[str, torch.Tensor] = ckpt["state_dict"]

    ppo_policy = ppo_model.policy
    ppo_sd = ppo_policy.state_dict()

    transferred = []
    skipped = []

    for bc_key, ppo_keys in _PARAMETER_MAP:
        if bc_key not in bc_sd:
            raise KeyError(
                f"Expected BC parameter '{bc_key}' not found in checkpoint. "
                f"Available BC keys: {list(bc_sd.keys())}"
            )
        bc_w = bc_sd[bc_key]

        # Track how many of the candidate destinations actually exist in the
        # current policy. We allow some to be missing (the layout differs
        # across SB3 versions) but require at least one match per BC key,
        # otherwise the BC weights would be silently lost.
        n_matched = 0
        for ppo_key in ppo_keys:
            if ppo_key not in ppo_sd:
                continue
            ppo_w = ppo_sd[ppo_key]
            if bc_w.shape != ppo_w.shape:
                raise ValueError(
                    f"Shape mismatch for {bc_key} -> {ppo_key}: "
                    f"BC has {bc_w.shape}, PPO expects {ppo_w.shape}"
                )
            ppo_sd[ppo_key] = bc_w.to(ppo_w.device, dtype=ppo_w.dtype)
            transferred.append((bc_key, ppo_key))
            n_matched += 1

        if n_matched == 0:
            raise KeyError(
                f"None of the candidate destinations for BC parameter '{bc_key}' "
                f"were found in the policy state dict.\n"
                f"  Tried: {ppo_keys}\n"
                f"  Policy has (first 30 keys): {list(ppo_sd.keys())[:30]}"
            )

    # Identify what we did NOT transplant
    transplanted_ppo_keys = set(p for _, ppo_keys in _PARAMETER_MAP for p in ppo_keys)
    for k in ppo_sd:
        if k not in transplanted_ppo_keys:
            skipped.append(k)

    # Apply
    ppo_policy.load_state_dict(ppo_sd, strict=True)

    if verbose:
        print(f"[transplant] BC checkpoint: {bc_ckpt_path}")
        print(f"[transplant]   BC val_acc at save:   {ckpt.get('val_acc', '?')}")
        print(f"[transplant]   BC epoch at save:     {ckpt.get('epoch', '?')}")
        print(f"[transplant] transferred {len(transferred)} parameter tensors:")
        for bk, pk in transferred:
            print(f"     {bk:<26} -> {pk}")
        sample = ", ".join(skipped[:6]) + (f" (+{len(skipped)-6} more)" if len(skipped) > 6 else "")
        print(f"[transplant] left at PPO default ({len(skipped)} keys): {sample}")

    return {
        "transferred": transferred,
        "skipped": skipped,
        "bc_val_acc": ckpt.get("val_acc"),
        "bc_epoch_at_save": ckpt.get("epoch"),
    }
