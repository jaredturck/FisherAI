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
- Full-search positions train policy and value.
- Fast-search positions train value only.
- Evaluation and UCI play use 256 simulations by default.

Search remains AlphaZero-style: PUCT selection, root Dirichlet noise, visit-count policy targets, final game-result value targets, subtree reuse, and opening exploration.

## Asynchronous workstation pipeline

The workstation command uses 24 independent Python actor processes. Each actor is pinned to one logical CPU on Linux and manages six concurrent game threads:

```text
24 actor processes
    × 6 active games each
    = 144 concurrent self-play games
```

The game threads are intentionally asynchronous. While one game waits for a neural-network result, another game in the same actor continues CPU-side MCTS work. Each actor may keep eight inference requests outstanding, and each search may reserve up to twenty-four leaves using virtual loss.

All actors feed one shared inference queue:

```text
24 CPU actors
       ↓
shared bounded inference queue
   ↙                    ↘
RTX 3090 cuda:0    RTX 3090 cuda:1
```

The next available inference server consumes work, so actors are not permanently assigned to one GPU. Each server gathers requests until it reaches the target batch, the maximum batch, or the short batching deadline.

Default inference settings:

```text
Target batch:                 512 positions
Maximum batch:                1,024 positions
Batch wait:                   2 ms
Request batch:                up to 32 leaves
Outstanding requests/actor:   8
Inference queue capacity:     4,096 requests
```

Large tensors stay in shared memory. Queues carry only actor, slot, request, and batch identifiers. The GPU gathers only legal policy logits and returns those logits plus the scalar value.

Completed games are sent to a dedicated replay writer and committed to LMDB immediately. Replay compression, metrics, and checkpoint loading remain outside the actor hot path.

## CPU and search optimizations

- 24 independent processes bypass the Python GIL for actor work.
- Multiple game threads per actor keep CPU work available while inference is pending.
- Multiple shared-memory request slots prevent one request from blocking an actor.
- One shared inference queue dynamically balances both GPUs.
- MCTS child statistics use contiguous NumPy arrays for vectorized PUCT selection.
- Search nodes reconstruct a selected leaf from one root-state copy instead of copying every intermediate state.
- Immutable history snapshots cache piece planes and are shared across copied states.
- Root encodings are reused for training samples rather than encoded twice.
- Pinned host buffers, FP16 inference, channels-last tensors, and GPU-side legal-policy gathering reduce transfer overhead.
- Replay writing runs in a dedicated process with larger transactions.

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

This creates a random checkpoint and reports the model parameter count. Existing checkpoints and replay data from the previous desktop-optimized release remain compatible.

## Start dual-GPU workstation training

```bash
python -m fisher_ai workstation
```

Committed defaults:

```text
CPU actors:                    24
Games per actor:               6
Active games:                  144
Parallel leaves per search:    24
Self-play GPUs:                cuda:0 and cuda:1
Learner GPU:                   cuda:0
Inference batch target/max:    512 / 1,024
```

Stop the complete process group with `Ctrl+C`.

Override the actor layout when diagnosing:

```bash
python -m fisher_ai workstation \
    --actors 24 \
    --games-per-actor 6 \
    --devices cuda:0 cuda:1
```

## One-off hardware benchmark

Run the benchmark once before deciding whether to adjust execution parameters:

```bash
python -m fisher_ai benchmark
```

The benchmark runs twenty-six unique sweep configurations covering:

- 4, 6, 8, 10, 12, and 14 games per actor;
- 8, 16, 24, and 32 pending leaves, including the strongest combined settings;
- fair per-actor inference-slot counts so larger game counts are not artificially throttled;
- target GPU batches from 256 to 1,024 and maximum batches from 512 to 2,048;
- 0.5, 1, 2, and 4 ms batching waits;
- 24, 28, and 32 actor processes.

Each sweep configuration uses five seconds of warmup and fifteen seconds of measurement. The top three sweep configurations are then rerun for thirty seconds each. The default run contains about ten and a half minutes of timed work plus process startup and cleanup.

Results are written to a timestamped directory:

```text
benchmarks/YYYYMMDD_HHMMSS/benchmark_results.csv
benchmarks/YYYYMMDD_HHMMSS/benchmark_summary.md
```

The benchmark does **not** modify `fisher_config.json`, checkpoints, or the real replay database. It uses temporary replay storage and the same checkpoint for every configuration. The CSV records the sweep and confirmation stages, actor count, inference-slot count, timings, throughput, queue pressure, and hardware utilization. The Markdown summary reports the full sweep separately from the longer confirmation runs and recommends settings from the confirmed winner.

Useful shorter diagnostic run:

```bash
python -m fisher_ai benchmark \
    --profiles 3 \
    --warmup-seconds 2 \
    --measure-seconds 5 \
    --confirm-top 0
```

## Live metrics

The workstation reports current throughput and queue health:

```text
self-play games=... replay_games=... replay_positions=... active=144
moves/s=... games/hour=... positions/s=... evals/s=...
batch_avg=... batch_p50=... batch_p95=... queue=... inflight=...
```

The most important measures are real moves per second, completed positions per second, evaluations per second, and games per hour. CPU and GPU percentages are supporting diagnostics.

## Individual commands

Distributed self-play without the learner:

```bash
python -m fisher_ai self-play \
    --actors 24 \
    --games-per-actor 6 \
    --devices cuda:0 cuda:1 \
    --games 1000
```

Single-process debugging path:

```bash
python -m fisher_ai self-play --actors 1 --games 8
```

One learner burst:

```bash
python -m fisher_ai learn --steps 100
```

Continuous learner:

```bash
python -m fisher_ai learn --steps 100 --continuous
```

Smoke-test learner update before the normal warmup:

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
python -m fisher_ai benchmark --help
```

Strength should be measured through fixed checkpoint matches rather than inferred from training age. Checkpoints can plateau or regress temporarily, so milestone evaluation remains important.
