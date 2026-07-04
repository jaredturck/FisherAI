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

The model has **3,310,234 trainable parameters**. It is small enough for high-throughput desktop self-play while retaining enough capacity for strong chess.

## Self-play search

Self-play uses randomized search allocation:

- 25% of positions receive 128 MCTS simulations.
- 75% receive 32 MCTS simulations.
- Full-search positions train both policy and value.
- Fast-search positions train value only.
- Evaluation and UCI play use 256 simulations by default.

The average self-play cost is 56 simulations per move. Search reserves up to eight leaves per game using virtual loss, allowing many positions to be evaluated together.

Other search behavior remains AlphaZero-style:

- PUCT selection
- root Dirichlet noise
- visit-count policy targets
- final game-result value targets
- subtree reuse after each played move
- opening move sampling followed by low-temperature play

## Multiprocess workstation pipeline

The workstation command uses all 24 logical CPU threads as independent self-play actors. Each actor is single-threaded and pinned to one logical CPU on Linux.

```text
24 CPU actor processes
    × 6 active games each
    = 144 concurrent self-play games

12 actors → shared inference queue → RTX 3090 cuda:0
12 actors → shared inference queue → RTX 3090 cuda:1
```

Each actor performs chess rules, MCTS traversal, position encoding, and tree backup. When a search reaches neural-network leaves, the actor writes the encoded positions and legal moves into shared memory and submits only a small request descriptor.

Each GPU inference server:

1. collects requests from many actors;
2. combines them into batches of up to roughly 512 positions;
3. performs one mixed-precision network evaluation;
4. gathers only legal policy logits on the GPU;
5. writes policy and value results back to shared memory.

Actors immediately continue their searches when results arrive. Completed games are sent to a dedicated replay-writer process and committed to LMDB without waiting for the other active games to finish.

The learner reads from the same replay database and starts once the configured warmup has been reached. During workstation training, `cuda:0` handles both learner updates and one self-play inference server, while `cuda:1` remains dedicated to self-play inference.

## Runtime optimizations

- 24 independent Python actor processes bypass the GIL
- 144 concurrent games keep inference requests available
- per-actor shared-memory request and response slots
- bounded GPU inference queues with backpressure
- FP16 inference and mixed-precision training
- channels-last CUDA tensors
- TF32 enabled on supported NVIDIA hardware
- GPU-side extraction of legal policy logits
- inference batches targeting 512 positions
- immediate replay writes for every completed game
- grouped replay sampling to reduce decompression work
- replay capacity measured in positions rather than games
- learner pacing capped at two sampled positions per generated position
- atomic checkpoint writes and checkpoint retention
- checkpoint hot-reloading by both inference servers
- live games/hour, positions/second, evaluations/second, and batch-size metrics

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Install the PyTorch build appropriate for the local NVIDIA driver when a specific CUDA build is required.

## Initialize

```bash
python -m fisher_ai init
```

This creates a random checkpoint and reports the model parameter count. Existing Fisher AI 0.2 model checkpoints and replay data remain compatible with this release.

## Start dual-GPU workstation training

```bash
python -m fisher_ai workstation
```

The committed defaults are:

```text
CPU actors:              24
Games per actor:         6
Active games:            144
Self-play GPUs:          cuda:0 and cuda:1
Learner GPU:             cuda:0
Inference batch target:  512
```

Stop the complete process group with `Ctrl+C`.

Override the actor layout when diagnosing or benchmarking:

```bash
python -m fisher_ai workstation \
    --actors 24 \
    --games-per-actor 6 \
    --devices cuda:0 cuda:1
```

## Individual commands

Run distributed self-play without the learner:

```bash
python -m fisher_ai self-play \
    --actors 24 \
    --games-per-actor 6 \
    --devices cuda:0 cuda:1 \
    --games 1000
```

Generate distributed self-play continuously:

```bash
python -m fisher_ai self-play \
    --actors 24 \
    --games-per-actor 6 \
    --devices cuda:0 cuda:1 \
    --continuous
```

Run the original single-process path for debugging:

```bash
python -m fisher_ai self-play --actors 1 --games 8
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

## Progress output

The workstation reports useful work rather than waiting for a whole game cohort:

```text
self-play games=18 replay_games=18 replay_positions=4,892 active=144
plies=4,892 games/hour=... positions/s=... evals/s=... avg_gpu_batch=...
```

The learner prints its warmup status only when the replay count changes or once per minute, avoiding repeated zero-status spam.

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

```text
Network:                   10 blocks × 128 channels, SE-32
Parameters:                3,310,234
Full self-play search:     128 simulations
Fast self-play search:     32 simulations
Full-search fraction:      25%
Evaluation search:         256 simulations
Parallel leaves/game:      8
CPU actors:                24
Games per actor:           6
Active games:              144
GPU inference servers:     2
Inference batch target:    512
Inference batch wait:      2 ms
Training batch:            1,024
Training micro-batch:      512
Replay window:             2,000,000 positions
Replay warmup:             20,000 positions
Maximum game length:       320 plies
```

Strength should be measured through fixed checkpoint matches rather than assumed from training age. More self-play normally improves the model, but checkpoints can plateau or regress temporarily, so milestone evaluation remains important.
