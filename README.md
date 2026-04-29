# Super Mario Bros Reinforcement Learning

Final year project — BSc Computer Science, Queen Mary University of London.

Training a reinforcement learning agent to play **Super Mario Bros** through a combination of Proximal Policy Optimisation (PPO) and Behaviour Cloning (BC). The central research question is whether BC-initialised PPO improves training speed, stability, and final performance compared to PPO trained from scratch.

## Project structure

```
mario-rl/
├── envs/
│   ├── wrappers.py         # SkipFrame, EpisodicLifeEnv, MarioRewardShaping,
│   │                       # StuckDetector, LazyFramesToArray
│   └── make_env.py         # make_env() and make_vec_env()
├── agents/
│   ├── bc.py               # Behaviour Cloning (NatureCNN policy, cross-entropy loss)
│   ├── hybrid_ppo.py       # HybridPPO: SB3 PPO + concurrent BC auxiliary loss
│   ├── transplant.py       # BC -> PPO weight transplant
│   ├── value_warmup.py     # Value head warmup after BC transplant
│   ├── callbacks.py        # MarioEvalCallback (flag rate, x_pos, time-to-flag)
│   ├── metrics.py          # Episode metric summarisation
│   └── utils.py            # Config loading, seeding, metadata
├── scripts/
│   ├── train_ppo.py        # PPO training (supports --resume-from, --level)
│   ├── train_bc.py         # Behaviour cloning training
│   ├── train_hybrid_concurrent.py  # Full BC+PPO hybrid pipeline
│   ├── evaluate.py         # Multi-level evaluation -> CSV
│   ├── record_demo.py      # Human gameplay recorder (pygame)
│   ├── record_run.py       # Agent video recorder -> MP4
│   ├── smoke_test_ppo.py   # 10k-step sanity check
│   └── testingEnv.py       # Random agent env verification
├── configs/
│   ├── ppo_default.yaml           # PPO baseline
│   ├── ppo_500k_diag.yaml         # PPO diagnostic (best performer: 100% on 1-1)
│   ├── bc_default.yaml            # BC training for 1-1
│   ├── bc_1-3.yaml                # BC training for 1-3
│   ├── hybrid_concurrent_v2.yaml  # BC+PPO hybrid v2 (bc_coef 0.8, lr 5e-5)
│   ├── hybrid_1-1_v3.yaml         # BC+PPO hybrid v3 (bc_coef 0.5, lr 1e-4, ent 0.02)
│   ├── hybrid_1-1_v4.yaml         # BC+PPO hybrid v4 (PPO hyperparams + BC aux loss)
│   ├── hybrid_1-3.yaml            # BC+PPO hybrid on 1-3
│   ├── finetune_1-3.yaml          # PPO fine-tune from 1-1 weights to 1-3
│   └── _archive/                  # Earlier config iterations
├── data/demos/
│   ├── world1_1/           # Human gameplay recordings for 1-1 (38 episodes)
│   └── world1_3/           # Human gameplay recordings for 1-3 (22 episodes)
├── logs/                   # TensorBoard logs, checkpoints, eval CSVs (gitignored)
├── requirements.txt
└── README.md
```

## Install (WSL Ubuntu + NVIDIA GPU)

These instructions assume WSL2 on Windows with an NVIDIA GPU and working CUDA drivers on the Windows host. You should **not** install CUDA inside WSL — only the CUDA-enabled PyTorch wheels.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify GPU:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Version pins

`gym-super-mario-bros` is built on `nes-py` which requires classic `gym` (not `gymnasium`). The pins in `requirements.txt` (`gym==0.25.2`, `gym-super-mario-bros==7.4.0`, `nes-py==8.2.1`, `stable-baselines3==1.8.0`) are the last mutually compatible versions. Do not upgrade individually.

## Quickstart

```bash
# Verify env works (random agent)
python scripts/testingEnv.py --world 1 --stage 1 --no-gif

# Smoke-test the PPO pipeline (10k steps)
python scripts/smoke_test_ppo.py

# Full PPO training run
python scripts/train_ppo.py --config configs/ppo_500k_diag.yaml --timesteps 2000000

# Watch training live
tensorboard --logdir logs/

# Evaluate trained model across levels
python scripts/evaluate.py --run-dir logs/ppo_500k_diag --levels 1-1 1-2 1-3 --episodes 20

# Record agent videos
python scripts/record_run.py --run-dir logs/ppo_500k_diag --episodes 5
```

## Training pipelines

### PPO from scratch

```bash
python scripts/train_ppo.py --config configs/ppo_500k_diag.yaml --timesteps 2000000
```

Supports `--resume-from <checkpoint.zip>` to fine-tune from an existing model and `--level 1-3` to override the level without editing the config.

### Behaviour Cloning

```bash
# 1. Record human demos (J=jump, K=run, arrows=move, R=restart, ESC=quit)
python scripts/record_demo.py --world 1 --stage 1 --out data/demos/world1_1/

# 2. Train BC model
python scripts/train_bc.py --config configs/bc_default.yaml
```

### BC + PPO Hybrid

Full pipeline: BC weight transplant → value head warmup → PPO with concurrent BC auxiliary loss.

```bash
python scripts/train_hybrid_concurrent.py --config configs/hybrid_1-1_v3.yaml
```

The BC auxiliary loss is annealed linearly from `bc_coef_start` → `bc_coef_end` over training, gradually handing control from demos to PPO.

## Key results

All evaluations use the `best/best_model.zip` checkpoint (best eval return during training), 20 deterministic episodes on 1-1.

| Run | Steps | Flag rate (1-1) | Notes |
|-----|-------|-----------------|-------|
| PPO from scratch (`ppo_500k_diag`) | 2M | **100%** | Best performer. Mean 388 frames to flag. |
| BC+PPO hybrid v2 (`hybrid_v2_nocw_2M`) | 2M | 55% at peak | Collapsed after 500k due to high bc_coef (0.8) anchoring to stuck BC policy |
| BC alone (`bc_no_class_weights`) | — | 0% | Consistent x=1786 every episode — stuck, no exploration |
| Fine-tune 1-3 from 1-1 PPO | 3M | 0% | Plateaued at x≈750, gap-jumping not generalised |

### Generalisation (best PPO model evaluated zero-shot)

| Level | Flag rate | Mean x_pos | Note |
|-------|-----------|------------|------|
| 1-1 | 75% | ~3000 | Trained level |
| 1-3 | 0% | ~300 | Same overworld tileset, poor transfer |
| 1-2 | 0% | ~173 | Underground — complete visual domain shift |

### Key findings so far

- **PPO from scratch beats BC+PPO hybrid on 1-1.** Root cause: the BC model (`bc_no_class_weights`) was deterministically stuck at x=1786. Transplanting it into PPO anchored the policy to a failure mode, and the high BC auxiliary loss coefficient (0.8) prevented PPO from escaping it for over 1M steps.
- **BC+PPO hybrid can work** — the peak 55% completion rate shows the approach isn't fundamentally broken, just sensitive to BC model quality and the bc_coef schedule.
- **Visual transfer fails across level types.** The CNN learns pixel patterns specific to the trained level. Overworld → underground (1-2) is a complete failure; overworld → different overworld layout (1-3) is a partial failure.
- **StuckDetector is necessary.** Without random action injection at stuck x-positions, both BC and hybrid policies loop indefinitely at obstacle points.

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Literature review | Done |
| 2 | Environment setup + preprocessing | Done |
| 3 | PPO baseline on 1-1 | Done — 100% flag completion |
| 4 | Demo recording pipeline | Done — 38 eps (1-1), 22 eps (1-3) |
| 5 | Behaviour cloning | Done |
| 6 | BC weight transplant + value warmup | Done |
| 7 | Hybrid BC+PPO training | Done (v2, v3, v4 variants) |
| 8 | Generalisation evaluation across levels | Done |
| 9 | Transfer learning (1-1 → 1-3) | In progress |
| 10 | Final report writeup | Planned |

## Research questions

1. Does BC initialisation improve PPO's sample efficiency on Super Mario Bros?
2. Does BC initialisation reduce training variance?
3. How does BC model quality (stuck vs completing) affect hybrid training?
4. Does visual transfer across levels improve with a shared BC+PPO foundation?

## License

Academic use only. ROM files are not distributed with this repository — `gym-super-mario-bros` bundles a permissively licensed ROM for research use.
