# Fisher AI

Fisher AI is a compact AlphaZero-style chess trainer built around one repeated
loop:

```text
Generate a fresh in-memory self-play window
→ train for three epochs on fresh positions plus sampled replay
→ append complete fresh games to the RAM replay window
→ save checkpoints/latest.pt
→ repeat
```

The generation phase uses CPU actor processes and one GPU inference server.
The training phase starts only after generation has stopped, then uses the same
configured GPU.

## Training schedule

Fresh self-play windows grow with cumulative generated positions:

```text
0–100,000:       10,000 positions per iteration
100,000–300,000: 20,000 positions per iteration
300,000–750,000: 30,000 positions per iteration
750,000 onward:  50,000 positions per iteration
```

The cumulative fresh-position counter is stored in the atomic model checkpoint.
Replay remains in RAM and warms up again after a process restart.

## Replay window

Replay retains up to 200,000 positions using FIFO eviction by complete game.
Every completed fresh game is inserted after training so current positions are
not sampled twice in the same iteration. Each epoch uses all fresh positions
and independently samples replay without replacement:

```text
10,000-position stage: replay = 0.5 × fresh
Later stages:           replay = 1.0 × fresh
```

## Performance architecture

The self-play hot path uses packed integer moves, dense MCTS state and history
pools, batched state encoding, two inference slots per actor, and shared-array
completed-game transfer. Generated training data is retained in contiguous
structure-of-arrays windows so batches are assembled through indexed gathers.
The trainer, optimizer, and gradient scaler remain resident across outer
iterations. CUDA training materializes one batch ahead and reuses pinned host
buffers for transfers into channels-last GPU tensors.

## Commands

Run continuous training:

```bash
python -m fisher_ai train
```

Run a fixed number of outer iterations:

```bash
python -m fisher_ai train --iterations 1
```

Run the current generation and training benchmark:

```bash
python -m fisher_ai benchmark
```

Benchmark output overwrites:

```text
benchmarks/benchmark_results.csv
benchmarks/benchmark_summary.md
```

The CSV uses a long-format metric schema and contains end-to-end phase timing,
parent-side blocking, training batch timing, workload distributions, process
resource counters, memory usage, configuration metadata, and isolated
component benchmarks for legal move generation, encoding, actor-side MCTS,
batch materialization, transfers, neural inference, network stages, and one
training step.

The normal training path is not instrumented. Detailed component measurements
run only under the benchmark command and are kept separate from the reported
end-to-end generation and training throughput. The benchmark keeps the trainer
resident during generation to match the production pipeline lifecycle.

Launch the chess GUI:

```bash
python -m fisher_ai gui
```

## Configuration

`fisher_config.json` contains only the performance controls intended for
routine tuning:

- CUDA device
- actor and active-game counts
- inference batch sizes and wait time
- MCTS simulations and parallel searches
- training batch size

Network architecture, search constants, replay policy, fresh-window schedule,
the three training epochs, checkpoint location, and benchmark location are
fixed in code.

## Checkpoints

Training stores one atomic checkpoint at:

```text
checkpoints/latest.pt
```

The checkpoint contains model weights, optimizer state, gradient-scaler state,
optimizer step, and cumulative fresh-position count.

## Discord reporting

Set `STATUS_WEBHOOK` in `.env` to receive one synchronous Discord report after
each completed generate-and-train iteration.
