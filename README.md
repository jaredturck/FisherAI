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

## Batched workstation pipeline

The workstation command uses 24 independent Python actor processes. Each actor is pinned to one logical CPU on Linux and advances six self-play sessions together in one process-level loop:

```text
24 actor processes
    × 6 active games each
    = 144 concurrent self-play games
```

The actor collects pending leaves across all six games before submitting one combined inference request. This restores the process-level CPU parallelism of the earlier CPU-utilization design while retaining the newer global GPU queue, shared-memory transport, vectorized MCTS selection, cached encodings, replay writer, and detailed metrics.

All actors feed one shared inference queue:

```text
24 CPU actors
       ↓
shared bounded inference queue
   ↙                    ↘
RTX 3090 cuda:0    RTX 3090 cuda:1
```

The next available inference server consumes work, so actors are not permanently assigned to one GPU. Each server gathers complete actor requests until it reaches the target batch, the maximum batch, or the short batching deadline.

Default execution settings:

```text
Scheduler:                     batched
Actor processes:               24
Games per actor:               6
Parallel leaves per game:      8
Actor request capacity:        automatic, currently 64 positions
Shared-memory slots per actor: 1
Target GPU batch:              512 positions
Maximum GPU batch:             1,024 positions
Batch wait:                    2 ms
Inference queue capacity:      4,096 requests
```

`inference_request_batch_size` is a minimum. A value of zero enables automatic sizing from `games_per_actor × parallel_searches`, rounded up to a practical shared-memory boundary. Expected batched actor requests are therefore not split into serial 32-position chunks.

Large tensors stay in shared memory. Queues carry only actor, slot, request, and batch identifiers. The GPU gathers only legal policy logits and returns those logits plus the scalar value.

Completed games are sent to a dedicated replay writer and committed to LMDB immediately. Replay compression, metrics, and checkpoint loading remain outside the actor hot path.

## CPU and search optimizations

- 24 independent processes bypass the Python GIL for actor work.
- Each actor advances all of its sessions together instead of creating blocking per-game Python threads.
- One combined actor inference request carries pending leaves from multiple games.
- One shared inference queue dynamically balances both GPUs.
- Each active game uses one preallocated MCTS arena instead of per-move Python objects,
  dictionaries, lists, and NumPy arrays.
- Actions, packed moves, priors, visits, values, virtual visits, and child ranges live in
  contiguous typed arrays addressed by integer record IDs.
- Selected leaves reconstruct state from one root-state copy and a temporary path of packed moves.
- Immutable history snapshots cache piece planes and are shared across copied states.
- Only the current root encoding is retained for its training sample; leaf encodings are temporary.
- Pinned host buffers, FP16 inference, channels-last tensors, and GPU-side legal-policy gathering reduce transfer overhead.
- Replay writing runs in a dedicated process with larger transactions.
- Live timing separates actor compute, GPU-response waiting, request-queue waiting, and replay waiting.

The previous per-game threaded scheduler remains available only as a benchmark comparison mode. Production training defaults to the batched scheduler.

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

This creates a random checkpoint and reports the model parameter count. Existing checkpoints and replay data remain compatible.

## Start dual-GPU workstation training

```bash
python -m fisher_ai workstation
```

Committed defaults:

```text
Self-play scheduler:            batched
CPU actors:                     24
Games per actor:                6
Active games:                   144
Parallel leaves per search:     8
Self-play GPUs:                 cuda:0 and cuda:1
Learner GPU:                    cuda:0
Inference batch target/max:     512 / 1,024
Actor inference request size:   automatic
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

Run the benchmark once before selecting the final execution parameters:

```bash
python -m fisher_ai benchmark
```

The benchmark runs 15 focused profiles. It directly compares the restored process-level batched scheduler against the regressed per-game threaded scheduler, then explores the plausible batched combinations:

- the exact old CPU-utilization control: 24 actors, 6 games, 8 leaves;
- the current threaded 4-game, 24-leaf configuration;
- 4, 6, and 8 games per actor;
- 8, 16, 24, and 32 pending leaves where useful;
- 1 ms and 2 ms batching waits;
- 512 and 768 target GPU batches;
- 22 and 24 actor processes.

Each sweep configuration uses five seconds of warmup and fifteen seconds of measurement. The top three sweep configurations are then rerun for thirty seconds each.

Results are written to a timestamped directory:

```text
benchmarks/YYYYMMDD_HHMMSS/benchmark_results.csv
benchmarks/YYYYMMDD_HHMMSS/benchmark_summary.md
```

The benchmark does **not** modify `fisher_config.json`, checkpoints, or the real replay database. It uses temporary replay storage and the same checkpoint for every configuration. Reports include scheduler type, resolved actor request capacity, average actor request size, GPU batch size, throughput, system CPU utilization, actor work time, inference wait time, queue pressure, and GPU utilization.

Useful shorter diagnostic run:

```bash
python -m fisher_ai benchmark \
    --profiles 3 \
    --warmup-seconds 2 \
    --measure-seconds 5 \
    --confirm-top 0
```

## Live metrics

The workstation reports current throughput and where actor time is being spent:

```text
self-play games=... replay_games=... replay_positions=... active=144
moves/s=... games/hour=... positions/s=... evals/s=...
request_avg=... batch_avg=... batch_p50=... batch_p95=...
actor_compute=... inference_wait=... queue=... inflight=... replay_queue=...
```

The most important measures are real moves per second, completed positions per second, evaluations per second, and games per hour. System CPU, actor compute, and inference wait are supporting diagnostics that show whether the actors or GPUs are limiting throughput.

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

## Play against FisherAI

The Pygame interface loads the latest checkpoint from the configured checkpoint directory.
You play White and FisherAI uses the configured evaluation simulation count for each move.

```bash
python gui/main.py
```

Click a piece and then a highlighted destination square. Use the **New Game** button or
press `R` to restart. Pawn promotion defaults to a queen.

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
