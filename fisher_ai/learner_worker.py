import sys
import time

import torch

from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import available_device, load_config
from fisher_ai.network import FisherNetwork
from fisher_ai.replay import ReplayBuffer
from fisher_ai.trainer import AlphaZeroTrainer

STEPS_PER_BURST = 100
LOG_EVERY = 10
POLL_SECONDS = 2.0


def run_learner(config_path):
    config = load_config(config_path)
    device = available_device(config.runtime.learner_device)
    replay = ReplayBuffer(
        config.runtime.replay_path,
        max_positions=config.training.replay_max_positions,
    )
    model = FisherNetwork(config.network)
    manager = CheckpointManager(
        config.runtime.checkpoint_dir,
        keep_recent=config.training.checkpoint_keep_recent,
        milestone_interval=config.training.checkpoint_milestone_interval,
    )
    trainer = AlphaZeroTrainer(
        model,
        replay,
        config,
        device=device,
        checkpoint_manager=manager,
    )
    trainer.load_checkpoint()
    last_wait_status = None
    last_wait_log = 0.0

    try:
        while True:
            completed = 0
            start_time = time.perf_counter()

            while completed < STEPS_PER_BURST:
                replay.refresh()
                if replay.position_count < config.training.warmup_positions:
                    break
                if not trainer.can_train():
                    break

                metrics = trainer.train_step()
                completed += 1
                if completed % LOG_EVERY == 0 or completed == STEPS_PER_BURST:
                    elapsed = time.perf_counter() - start_time
                    positions_per_second = (
                        completed * config.training.batch_size / max(elapsed, 1e-6)
                    )
                    print(
                        f"step={metrics['step']} loss={metrics['loss']:.4f} "
                        f"policy={metrics['policy_loss']:.4f} "
                        f"value={metrics['value_loss']:.4f} "
                        f"lr={metrics['learning_rate']:.6f} "
                        f"positions/s={positions_per_second:.1f}",
                        flush=True,
                    )

            if completed:
                path = trainer.save_checkpoint()
                print(f"Saved {path}", flush=True)
                continue

            wait_status = (replay.position_count, replay.total_positions_added)
            now = time.monotonic()
            if wait_status != last_wait_status or now - last_wait_log >= 60.0:
                print(
                    f"Learner waiting: {replay.position_count:,}/"
                    f"{config.training.warmup_positions:,} warmup positions; "
                    f"{replay.total_positions_added:,} total generated",
                    flush=True,
                )
                last_wait_status = wait_status
                last_wait_log = now
            time.sleep(POLL_SECONDS)
    finally:
        replay.close()


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "fisher_config.json"
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    run_learner(config_path)


if __name__ == "__main__":
    main()
