# Fisher AI

Fisher AI is an AlphaZero-style chess engine built with PyTorch. It learns entirely through self-play using Monte Carlo Tree Search and uses a single GPU for both inference and training in separate phases.

## Installation

```bash
pip install -r requirements.txt
```

## Start training

```bash
python -m fisher_ai workstation
```

Training now runs as a simple repeated cycle:

```text
load checkpoint
→ generate a fixed in-memory window
→ stop generation
→ shuffle every position without replacement
→ train on the complete window
→ save the next checkpoint
```

The default window targets at least 50,000 positions and closes on whole-game boundaries so every completed training position is used. Completed games are retained only in RAM for the current iteration; Fisher AI no longer writes an LMDB replay database. Generation and training both use the single device configured by `runtime.device`.

Run a fixed number of iterations with:

```bash
python -m fisher_ai workstation --iterations 1
```

## Play against Fisher AI

```bash
python -m fisher_ai gui
```

The GUI loads the latest checkpoint from `checkpoints/`.

## Check and benchmark

```bash
ruff check .
python -m pytest -q
python -m fisher_ai benchmark
```

The benchmark measures generation and training separately and saves fresh results under `benchmarks/`. Use `--positions` to choose the benchmark window size.
