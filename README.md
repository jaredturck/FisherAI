````markdown
# Fisher AI

Fisher AI is an AlphaZero-style chess engine built with PyTorch. It learns entirely through self-play using Monte Carlo Tree Search and is optimized for a dual RTX 3090 workstation.

## Installation

```bash
pip install -r requirements.txt
```

## Start training

```bash
python -m fisher_ai workstation
```

This creates the initial checkpoint when needed and starts self-play and learning. A new checkpoint is saved after the first completed training burst at least 10 minutes after the previous save, and only the newest 10 are kept.

Checkpoint updates are sent to Discord when `STATUS_WEBHOOK` is set in `.env`. Settings are stored in `fisher_config.json`.

## Play against Fisher AI

```bash
python -m fisher_ai gui
```

The GUI loads the latest checkpoint, so training must have been started at least once.

## Check and benchmark

```bash
ruff check .
pytest
python -m fisher_ai benchmark
```

Benchmark results are saved under `benchmarks/`.
````
