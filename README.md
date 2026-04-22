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
# 1. Verify the env works end-to-end
python scripts/sanity_check.py --world 1 --stage 1

# This should run a random agent for up to 2000 steps on World 1-1,
# print the final Mario state, and save a GIF to logs/sanity_check.gif.
```

Expected output roughly:

```
Building env: World 1-1, actions=simple
  observation_space: Box(0, 255, (4, 84, 84), uint8)
  action_space:      Discrete(7)

Episode finished after 187 steps
  total reward: -12.40
  final info:   x_pos=421, flag_get=False, life=2
  GIF saved to: /.../logs/sanity_check.gif
  Sanity check passed.
```

A random agent will almost never reach the flag. That's expected — the point here is just to confirm the pipeline works.

## Milestones

Tracking against the interim report plan.

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Literature review | Done (interim report) |
| 2 | Environment setup + preprocessing | **In progress (this commit)** |
| 3 | PPO baseline (SB3) on World 1-1 | Planned |
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
