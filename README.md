# Fisher AI

Fisher AI is an AlphaZero-style chess engine built with PyTorch. It learns entirely through self-play using Monte Carlo Tree Search and is optimized for a dual RTX 3090 workstation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Start training

```bash
python -m fisher_ai workstation
```

This creates the initial checkpoint when needed and starts self-play and learning. Settings are stored in `fisher_config.json`.

## Play against Fisher AI

```bash
python -m fisher_ai gui
```

The GUI loads the latest checkpoint, so training must have been started at least once.

## Benchmark

```bash
python -m fisher_ai benchmark
```

Results are saved under `benchmarks/`.
