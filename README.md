# Fisher AI

Fisher AI is a compact AlphaZero-style chess trainer built around one repeated
loop:

```text
Generate a fresh in-memory self-play window
→ train on the complete window for three shuffled epochs
→ save checkpoints/latest.pt
→ repeat
```

The generation phase uses CPU actor processes and one GPU inference server.
The training phase starts only after generation has stopped, then uses the same
configured GPU.

## Performance architecture

The self-play hot path uses packed integer moves, dense MCTS state and history
pools, batched state encoding, two inference slots per actor, and shared-array
completed-game transfer. Generated training data is retained in a contiguous
structure-of-arrays window so batches are assembled through indexed gathers.

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
- generated window size
- training batch size

Network architecture, search constants, the three training epochs, checkpoint
location, and benchmark location are fixed in code.

## Checkpoints

Training stores one atomic checkpoint at:

```text
checkpoints/latest.pt
```

The checkpoint contains only model weights, optimizer state, gradient-scaler
state, and optimizer step.

## Discord reporting

Set `STATUS_WEBHOOK` in `.env` to receive one synchronous Discord report after
each completed generate-and-train iteration.
