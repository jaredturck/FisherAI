# Fisher AI

Fisher AI is a PyTorch implementation of the AlphaZero chess training loop from *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*.

It starts from random weights and learns exclusively through self-play. It does not use human games, opening books, engine labels, endgame tablebases, or supervised pretraining.

## Desktop-optimized architecture

The permanent Fisher AI network is sized for a dual RTX 3090 workstation:

```text
Input: 119 × 8 × 8

Stem:
    3×3 convolution, 119 → 128 channels
    BatchNorm + ReLU

Residual tower:
    10 residual blocks
    128 channels
    squeeze-and-excitation with 32 hidden channels

Policy head:
    1×1 convolution, 128 → 128
    BatchNorm + ReLU
    1×1 convolution, 128 → 73
    flatten to 4,672 action logits

Value head:
    1×1 convolution, 128 → 8
    BatchNorm + ReLU
    128-unit hidden layer
    scalar tanh output
```

The model has **3,310,234 trainable parameters**. This is roughly seven times smaller than the previous 23.3-million-parameter network while retaining enough capacity for strong chess.

## Self-play search

Self-play uses randomized search allocation:

- 25% of positions receive 128 MCTS simulations.
- 75% receive 32 MCTS simulations.
- Full-search positions train both policy and value.
- Fast-search positions train value only.
- Evaluation and UCI play use 256 simulations by default.

The average self-play cost is 56 simulations per move. Search reserves up to eight leaves per game using virtual loss, allowing batches of hundreds of positions to be evaluated together rather than issuing one GPU call per simulation.

Other search behavior remains AlphaZero-style:

- PUCT selection
- root Dirichlet noise
- visit-count policy targets
- final game result value targets
- subtree reuse after each played move
- opening move sampling followed by low-temperature play

## Workstation optimizations

- FP16 inference and mixed-precision training
- channels-last CUDA tensors
- TF32 enabled on supported NVIDIA hardware
- GPU-side extraction of legal policy logits
- inference batches up to 512 positions
- grouped replay sampling to reduce decompression work
- replay capacity measured in positions rather than games
- learner pacing capped at two sampled positions per generated position
- atomic checkpoint writes and checkpoint retention
- throughput reporting for self-play and learning
- dedicated continuous dual-GPU command

The default device assignment is:

```text
cuda:0  learner
cuda:1  self-play inference
```

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Install the PyTorch build appropriate for the local NVIDIA driver when a specific CUDA build is required.

## Migration from Fisher AI 0.1

The network architecture and replay schema changed. Existing checkpoints and replay data are incompatible and must be removed once any required backup has been made:

```bash
rm -rf checkpoints data/replay.lmdb
```

The project remains pure self-play, so old human-game datasets are not used.

## Initialize

```bash
python -m fisher_ai init
```

This creates a random checkpoint and reports the model parameter count.

## Recommended dual-GPU training

```bash
python -m fisher_ai workstation
```

This launches continuous self-play on `cuda:1` and continuous learning on `cuda:0`. Stop both processes with `Ctrl+C`.

Keep `data/replay.lmdb` and `checkpoints/` on the local NVMe rather than inside a synchronized cloud directory. Absolute paths can be set in `fisher_config.json`.

## Individual commands

Generate one self-play cohort:

```bash
python -m fisher_ai self-play --games 64
```

Generate continuously:

```bash
python -m fisher_ai self-play --games 64 --continuous
```

Train one learner burst:

```bash
python -m fisher_ai learn --steps 100
```

Train continuously while respecting replay warmup and learner pacing:

```bash
python -m fisher_ai learn --steps 100 --continuous
```

Run a small smoke-test update before the normal 20,000-position warmup:

```bash
python -m fisher_ai learn --steps 1 --force
```

## Evaluation

```bash
python -m fisher_ai evaluate \
    --checkpoint-a checkpoints/fisher_ai_000010000.pt \
    --checkpoint-b checkpoints/fisher_ai_000005000.pt \
    --games 100
```

Evaluation disables exploration noise, selects the highest-visit move, and uses 256 simulations per move.

## UCI

```bash
python -m fisher_ai uci
```

A UCI `go nodes N` command overrides the default 256 simulations for that move.

## Verification

```bash
ruff check .
pytest
python -m fisher_ai --help
```

## Configuration summary

The committed defaults are intentionally workstation-oriented:

```text
Network:                 10 blocks × 128 channels, SE-32
Parameters:              3,310,234
Full self-play search:   128 simulations
Fast self-play search:   32 simulations
Full-search fraction:    25%
Evaluation search:       256 simulations
Parallel leaves/game:    8
Inference batch:         512
Training batch:          1,024
Training micro-batch:    512
Replay window:           2,000,000 positions
Replay warmup:           20,000 positions
Maximum game length:     320 plies
```

Strength should be measured through fixed checkpoint matches rather than assumed from training age. More self-play normally improves the model, but checkpoints can plateau or regress temporarily, so milestone evaluation remains important.
