import sys
import time

import torch

from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import available_device, load_config
from fisher_ai.network import FisherNetwork
from fisher_ai.notifications import DiscordNotifier
from fisher_ai.replay import ReplayBuffer
from fisher_ai.trainer import AlphaZeroTrainer

STEPS_PER_BURST = 100
LOG_EVERY = 10
POLL_SECONDS = 2.0
CHECKPOINT_INTERVAL_SECONDS = 600.0


def format_duration(seconds):
    seconds = max(int(seconds), 0)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def checkpoint_due(last_checkpoint_time, current_time, last_saved_step, current_step):
    return (
        current_step > last_saved_step
        and current_time - last_checkpoint_time >= CHECKPOINT_INTERVAL_SECONDS
    )


def estimate_remaining_time(start_step, current_step, target_step, elapsed_seconds):
    completed_steps = current_step - start_step
    remaining_steps = max(target_step - current_step, 0)
    if completed_steps <= 0 or remaining_steps == 0:
        return 0.0 if remaining_steps == 0 else None
    return elapsed_seconds * remaining_steps / completed_steps


def checkpoint_notification_fields(
    trainer,
    config,
    metrics,
    checkpoint_path,
    session_start_step,
    session_start_time,
    interval_start_time,
    interval_start_positions,
    current_time,
):
    target_step = config.training.training_steps
    progress = min(100.0 * trainer.step / max(target_step, 1), 100.0)
    session_elapsed = current_time - session_start_time
    interval_elapsed = max(current_time - interval_start_time, 1e-6)
    interval_positions = trainer.sampled_positions - interval_start_positions
    positions_per_second = interval_positions / interval_elapsed
    eta = estimate_remaining_time(
        session_start_step,
        trainer.step,
        target_step,
        session_elapsed,
    )
    eta_text = "Target reached" if eta == 0.0 else format_duration(eta) if eta else "Unavailable"

    return [
        (
            "Training progress",
            f"Step **{trainer.step:,} / {target_step:,}** ({progress:.2f}%)\n"
            f"Session: **{format_duration(session_elapsed)}** • ETA: **{eta_text}**",
            False,
        ),
        (
            "Loss",
            f"Total: **{metrics['loss']:.4f}** • Policy: **{metrics['policy_loss']:.4f}** • "
            f"Value: **{metrics['value_loss']:.4f}**",
            False,
        ),
        (
            "Learner",
            f"Learning rate: **{metrics['learning_rate']:.6f}** • "
            f"Positions/s: **{positions_per_second:,.1f}**\n"
            f"Sampled positions: **{trainer.sampled_positions:,}**",
            False,
        ),
        (
            "Replay",
            f"Retained positions: **{trainer.replay.position_count:,}** • "
            f"Generated positions: **{trainer.replay.total_positions_added:,}**",
            False,
        ),
        ("Checkpoint", f"`{checkpoint_path.name}`", False),
    ]


def run_learner(config_path):
    config = load_config(config_path)
    device = available_device(config.runtime.learner_device)
    replay = ReplayBuffer(
        config.runtime.replay_path,
        max_positions=config.training.replay_max_positions,
    )
    model = FisherNetwork(config.network)
    manager = CheckpointManager(config.runtime.checkpoint_dir)
    trainer = AlphaZeroTrainer(
        model,
        replay,
        config,
        device=device,
        checkpoint_manager=manager,
    )
    trainer.load_checkpoint()
    notifier = DiscordNotifier()
    session_start_time = time.monotonic()
    session_start_step = trainer.step
    last_checkpoint_time = session_start_time
    last_checkpoint_step = trainer.step
    last_checkpoint_positions = trainer.sampled_positions
    last_wait_status = None
    last_wait_log = 0.0
    latest_metrics = None

    try:
        while True:
            completed = 0
            burst_start_time = time.perf_counter()

            while completed < STEPS_PER_BURST:
                replay.refresh()
                if replay.position_count < config.training.warmup_positions:
                    break
                if not trainer.can_train():
                    break

                latest_metrics = trainer.train_step()
                completed += 1
                if completed % LOG_EVERY == 0 or completed == STEPS_PER_BURST:
                    elapsed = time.perf_counter() - burst_start_time
                    positions_per_second = (
                        completed * config.training.batch_size / max(elapsed, 1e-6)
                    )
                    print(
                        f"step={latest_metrics['step']} loss={latest_metrics['loss']:.4f} "
                        f"policy={latest_metrics['policy_loss']:.4f} "
                        f"value={latest_metrics['value_loss']:.4f} "
                        f"lr={latest_metrics['learning_rate']:.6f} "
                        f"positions/s={positions_per_second:.1f}",
                        flush=True,
                    )

            if completed:
                current_time = time.monotonic()
                if checkpoint_due(
                    last_checkpoint_time,
                    current_time,
                    last_checkpoint_step,
                    trainer.step,
                ):
                    path = trainer.save_checkpoint()
                    print(f"Saved {path}", flush=True)
                    notifier.send(
                        "Fisher AI checkpoint saved",
                        checkpoint_notification_fields(
                            trainer,
                            config,
                            latest_metrics,
                            path,
                            session_start_step,
                            session_start_time,
                            last_checkpoint_time,
                            last_checkpoint_positions,
                            current_time,
                        ),
                        description="A new recoverable training checkpoint is available.",
                        color="green",
                        include_gpu_stats=True,
                    )
                    last_checkpoint_time = time.monotonic()
                    last_checkpoint_step = trainer.step
                    last_checkpoint_positions = trainer.sampled_positions
                continue

            wait_status = (replay.position_count, replay.total_positions_added)
            current_time = time.monotonic()
            if wait_status != last_wait_status or current_time - last_wait_log >= 60.0:
                print(
                    f"Learner waiting: {replay.position_count:,}/"
                    f"{config.training.warmup_positions:,} warmup positions; "
                    f"{replay.total_positions_added:,} total generated",
                    flush=True,
                )
                last_wait_status = wait_status
                last_wait_log = current_time
            time.sleep(POLL_SECONDS)
    finally:
        notifier.close()
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
