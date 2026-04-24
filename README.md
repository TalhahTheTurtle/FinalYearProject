# Super Mario Bros Reinforcement Learning

Final year project — BSc Computer Science, Queen Mary University of London.

Training a reinforcement learning agent to play **Super Mario Bros** through a combination of Proximal Policy Optimisation (PPO) and Behaviour Cloning (BC). The central research question is whether BC-initialised PPO improves training speed, stability, and final performance compared to PPO trained from scratch.

## Project structure

```
mario-rl/
├── envs/                  # Mario env wrappers + factory
│   ├── wrappers.py        # SkipFrame, EpisodicLife, RewardShaping, etc.
│   └── make_env.py        # make_env() and make_vec_env()
├── agents/                # PPO, BC, hybrid (built in later phases)
├── scripts/
│   └── sanity_check.py    # Verifies env works end-to-end
├── configs/
│   └── ppo_default.yaml   # Default PPO hyperparameters
├── data/demos/            # Recorded human gameplay (populated later)
├── logs/                  # TensorBoard logs, checkpoints, eval CSVs
├── report/                # Plots and tables for the final report
├── requirements.txt
└── README.md
```

## Install (WSL Ubuntu + NVIDIA GPU)

These instructions assume WSL2 on Windows with an NVIDIA GPU and working CUDA drivers on the Windows host. You should **not** install CUDA inside WSL — only the CUDA-enabled PyTorch wheels.

```bash
# From WSL, in the project root:
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# Install torch with CUDA 12.1 first (don't rely on requirements.txt for this)
pip install torch==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# Then the rest
pip install -r requirements.txt
```

Verify GPU is visible:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### A note on the version pins

`gym-super-mario-bros` is built on top of `nes-py`, which has not been updated to support `gymnasium`. It still needs classic `gym`. The pins in `requirements.txt` (`gym==0.25.2`, `gym-super-mario-bros==7.4.0`, `nes-py==8.2.1`, `stable-baselines3==1.8.0`) are the last mutually compatible versions. Do not upgrade them individually — the whole stack will break.

## Quickstart

Once install is done:

```bash
# 1. Verify the env works end-to-end (random agent)
python scripts/testingEnv.py --world 1 --stage 1 --no-gif

# 2. Smoke-test the PPO pipeline (10k steps, ~1-2 min on a 3060 Ti)
python scripts/smoke_test_ppo.py

# 3. Run the real PPO baseline (200k steps, ~20-30 min)
python scripts/train_ppo.py --config configs/ppo_default.yaml

# 4. Watch training live (in a separate terminal)
tensorboard --logdir logs/

# 5. Evaluate the trained model
python scripts/evaluate.py --run-dir logs/ppo_default

# 6. Record a video of the agent playing
python scripts/record_run.py --run-dir logs/ppo_default --episodes 3
```

## What each script does

| Script | Purpose |
|--------|---------|
| `scripts/testingEnv.py` | Random agent, verifies env stack works. Run this first after any env change. |
| `scripts/smoke_test_ppo.py` | 10k-step PPO run. Verifies the full training pipeline before committing to a real run. |
| `scripts/train_ppo.py` | Main PPO training. Reads a YAML config, logs to TensorBoard, checkpoints, saves best-by-eval-return model. |
| `scripts/evaluate.py` | Loads a trained model, runs deterministic episodes on specified levels, writes a CSV of metrics (flag reach, length, final x-position, score). |
| `scripts/record_run.py` | Renders MP4 videos of the trained agent playing. For the report. |

## Config-driven experiments

Every training run is driven by a YAML config. To make an ablation, copy the default and change what you want:

```bash
cp configs/ppo_default.yaml configs/ppo_gamma99.yaml
# edit gamma in the new file, then:
python scripts/train_ppo.py --config configs/ppo_gamma99.yaml --run-name ppo_gamma99
```

Quick overrides without editing the YAML:

```bash
python scripts/train_ppo.py --config configs/ppo_default.yaml --timesteps 1_000_000 --run-name ppo_1M
python scripts/train_ppo.py --config configs/ppo_default.yaml --seed 7 --run-name ppo_seedB
```

Each run produces:

```
logs/<run_name>/
├── config.yaml          # exact config used (for reproducibility)
├── metadata.json        # git commit, system info, torch version
├── tb/                  # TensorBoard event files
├── ckpts/               # periodic checkpoints
├── best/best_model.zip  # best-by-eval model (use this for evaluation)
├── final_model.zip      # model at end of training
├── eval.csv             # written by evaluate.py
└── videos/              # written by record_run.py
```

## Milestones

Tracking against the interim report plan.

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Literature review | Done (interim report) |
| 2 | Environment setup + preprocessing | Done |
| 3 | PPO baseline (SB3) on World 1-1 | **In progress (Phase 2 code complete, first real run pending)** |
| 4 | Demo recording pipeline | Planned |
| 5 | Behaviour cloning | Planned |
| 6 | Hybrid BC→PPO training | Planned |
| 7 | Evaluation across levels + ablations | Planned |
| 8 | Custom PyTorch PPO (side chapter) | Planned |
| 9 | Final report writeup | Planned |

## Research questions

1. Does BC initialisation improve PPO's sample efficiency on Super Mario Bros?
2. Does BC initialisation reduce training variance across seeds?
3. How does demo quantity (10 / 30 / 60 minutes of gameplay) affect the benefit of BC pretraining?
4. Does reward shaping interact with BC initialisation, or are they independent contributions?

## License

Academic use only. ROM files are not distributed with this repository — `gym-super-mario-bros` bundles a permissively licensed ROM for research use, which is what we rely on.
