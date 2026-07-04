import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import FisherConfig, available_device, load_config, save_config
from fisher_ai.distributed import DistributedSelfPlayPool
from fisher_ai.evaluate import play_match
from fisher_ai.mcts import MCTS, TorchEvaluator
from fisher_ai.network import FisherNetwork
from fisher_ai.replay import ReplayBuffer
from fisher_ai.self_play import SelfPlayRunner
from fisher_ai.trainer import AlphaZeroTrainer
from fisher_ai.uci import UCIEngine


def build_model(config):
    return FisherNetwork(config.network)


def build_checkpoint_manager(config):
    return CheckpointManager(
        config.runtime.checkpoint_dir,
        keep_recent=config.training.checkpoint_keep_recent,
        milestone_interval=config.training.checkpoint_milestone_interval,
    )


def build_replay(config):
    return ReplayBuffer(
        config.runtime.replay_path,
        max_positions=config.training.replay_max_positions,
    )


def ensure_checkpoint(config, model, manager):
    path = manager.latest_path()
    if path is None:
        path = manager.save(model, config, 0)
    return path


def load_model(config, checkpoint_path=None, device="cpu"):
    model = build_model(config)
    manager = build_checkpoint_manager(config)
    path = Path(checkpoint_path) if checkpoint_path else ensure_checkpoint(config, model, manager)
    step, _ = manager.load(model, path=path, device=device)
    return model, manager, step, path


def command_init(args):
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = FisherConfig()
        save_config(config, config_path)

    model = build_model(config)
    manager = build_checkpoint_manager(config)
    existing_path = manager.latest_path()
    if existing_path is not None:
        print(f"Fisher AI is already initialized at {existing_path}")
        return

    path = manager.save(model, config, 0)
    print(f"Using {args.config}")
    print(f"Created {path}")
    print(f"Parameters: {model.parameter_count():,}")


def command_self_play(args):
    config = load_config(args.config)
    actor_count = args.actors or 1

    if actor_count > 1:
        model = build_model(config)
        manager = build_checkpoint_manager(config)
        ensure_checkpoint(config, model, manager)
        pool = DistributedSelfPlayPool(
            config_path=args.config,
            actor_count=actor_count,
            games_per_actor=args.games_per_actor,
            devices=args.devices,
            checkpoint_path=args.checkpoint,
        )
        target_games = None if args.continuous else args.games
        pool.run(target_games=target_games)
        return

    preferred_device = args.device or config.runtime.self_play_device
    device = available_device(preferred_device)
    model, manager, step, path = load_model(config, args.checkpoint, device=device)
    evaluator = TorchEvaluator(
        model,
        device=device,
        inference_batch_size=config.runtime.inference_batch_size,
        channels_last=config.runtime.channels_last,
    )
    search = MCTS(evaluator, config.search, seed=config.runtime.seed)
    replay = build_replay(config)
    runner = SelfPlayRunner(
        search,
        config.search,
        training_config=config.training,
        seed=config.runtime.seed,
    )
    generated = 0

    try:
        while args.continuous or generated < args.games:
            if args.continuous:
                batch_size = args.games
            else:
                batch_size = min(
                    args.games - generated,
                    config.runtime.self_play_games_per_batch,
                )

            allow_resignation = (
                replay.game_count >= config.training.resignation_enabled_after_games
            )
            batch_path = path
            start_time = time.perf_counter()
            games = runner.play_games(
                batch_size,
                checkpoint_step=step,
                allow_resignation=allow_resignation,
            )
            replay.add_games(games)
            elapsed = time.perf_counter() - start_time
            positions = sum(len(game.samples) for game in games)
            generated += batch_size

            if args.checkpoint is None:
                latest_path = manager.latest_path()
                if latest_path is not None and latest_path != path:
                    step, _ = manager.load(model, path=latest_path, device=device)
                    model.eval()
                    path = latest_path

            games_per_hour = batch_size * 3600 / max(elapsed, 1e-6)
            positions_per_second = positions / max(elapsed, 1e-6)
            progress = "continuous" if args.continuous else f"{generated}/{args.games}"
            print(
                f"Generated {progress} games from {batch_path.name}; "
                f"{games_per_hour:.1f} games/hour, {positions_per_second:.1f} positions/s; "
                f"replay has {replay.game_count:,} games and "
                f"{replay.position_count:,} positions",
                flush=True,
            )
    finally:
        replay.close()


def command_learn(args):
    config = load_config(args.config)
    preferred_device = args.device or config.runtime.learner_device
    device = available_device(preferred_device)
    replay = build_replay(config)
    model = build_model(config)
    manager = build_checkpoint_manager(config)
    trainer = AlphaZeroTrainer(model, replay, config, device=device, checkpoint_manager=manager)
    trainer.load_checkpoint(args.checkpoint)
    last_wait_status = None
    last_wait_log = 0.0

    try:
        while True:
            completed = 0
            start_time = time.perf_counter()

            while completed < args.steps:
                replay.refresh()
                if not args.force and replay.position_count < config.training.warmup_positions:
                    break
                if not args.force and not trainer.can_train():
                    break

                metrics = trainer.train_step()
                completed += 1
                if completed % args.log_every == 0 or completed == args.steps:
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

            if not args.continuous:
                if completed == 0:
                    print(
                        f"Replay has {replay.position_count:,} retained positions and "
                        f"{replay.total_positions_added:,} total generated positions. "
                        "Generate more self-play data or use --force for a smoke test."
                    )
                return

            if completed == 0:
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
                time.sleep(args.poll_seconds)
    finally:
        replay.close()


def command_train(args):
    for cycle in range(args.cycles):
        print(f"Cycle {cycle + 1}/{args.cycles}: self-play")
        self_play_args = argparse.Namespace(
            config=args.config,
            checkpoint=None,
            games=args.games_per_cycle,
            continuous=False,
            device=None,
            actors=1,
            games_per_actor=None,
            devices=None,
        )
        command_self_play(self_play_args)

        print(f"Cycle {cycle + 1}/{args.cycles}: learning")
        learn_args = argparse.Namespace(
            config=args.config,
            checkpoint=None,
            steps=args.steps_per_cycle,
            force=args.force,
            continuous=False,
            poll_seconds=5.0,
            log_every=10,
            device=None,
        )
        command_learn(learn_args)


def command_workstation(args):
    config = load_config(args.config)
    model = build_model(config)
    manager = build_checkpoint_manager(config)
    ensure_checkpoint(config, model, manager)

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"

    learn_command = [
        sys.executable,
        "-m",
        "fisher_ai",
        "learn",
        "--config",
        args.config,
        "--steps",
        str(args.steps_per_burst),
        "--continuous",
        "--poll-seconds",
        "2",
    ]
    pool = DistributedSelfPlayPool(
        config_path=args.config,
        actor_count=args.actors,
        games_per_actor=args.games_per_actor,
        devices=args.devices,
    )
    learn_process = None

    try:
        pool.start()
        print("Starting continuous learning on the configured learner GPU", flush=True)
        learn_process = subprocess.Popen(learn_command)
        pool.monitor(external_processes=[learn_process])
    except KeyboardInterrupt:
        print("Stopping Fisher AI workstation training", flush=True)
    finally:
        pool.stop()
        if learn_process is not None and learn_process.poll() is None:
            learn_process.terminate()
            learn_process.wait()


def command_evaluate(args):
    config = load_config(args.config)
    preferred_device = args.device or config.runtime.self_play_device
    device = available_device(preferred_device)
    model_a, _, _, path_a = load_model(config, args.checkpoint_a, device=device)
    model_b, _, _, path_b = load_model(config, args.checkpoint_b, device=device)
    evaluator_a = TorchEvaluator(
        model_a,
        device=device,
        inference_batch_size=config.runtime.inference_batch_size,
        channels_last=config.runtime.channels_last,
    )
    evaluator_b = TorchEvaluator(
        model_b,
        device=device,
        inference_batch_size=config.runtime.inference_batch_size,
        channels_last=config.runtime.channels_last,
    )
    results = play_match(
        evaluator_a,
        evaluator_b,
        config.search,
        games=args.games,
        seed=config.runtime.seed,
    )
    print(f"A: {path_a}")
    print(f"B: {path_b}")
    print(json.dumps(results, indent=2))


def command_uci(args):
    config = load_config(args.config)
    preferred_device = args.device or config.runtime.self_play_device
    device = available_device(preferred_device)
    model, _, _, _ = load_model(config, args.checkpoint, device=device)
    evaluator = TorchEvaluator(
        model,
        device=device,
        inference_batch_size=config.runtime.inference_batch_size,
        channels_last=config.runtime.channels_last,
    )
    UCIEngine(evaluator, config.search, seed=config.runtime.seed).run()


def build_parser():
    parser = argparse.ArgumentParser(prog="fisher-ai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init",
        help="create the default config and random checkpoint",
    )
    init_parser.add_argument("--config", default="fisher_config.json")
    init_parser.set_defaults(handler=command_init)

    self_play_parser = subparsers.add_parser("self-play", help="generate self-play games")
    self_play_parser.add_argument("--config", default="fisher_config.json")
    self_play_parser.add_argument("--checkpoint")
    self_play_parser.add_argument("--games", type=int, default=64)
    self_play_parser.add_argument("--continuous", action="store_true")
    self_play_parser.add_argument("--device")
    self_play_parser.add_argument("--actors", type=int, default=1)
    self_play_parser.add_argument("--games-per-actor", type=int)
    self_play_parser.add_argument("--devices", nargs="+")
    self_play_parser.set_defaults(handler=command_self_play)

    learn_parser = subparsers.add_parser("learn", help="train from the replay buffer")
    learn_parser.add_argument("--config", default="fisher_config.json")
    learn_parser.add_argument("--checkpoint")
    learn_parser.add_argument("--steps", type=int, default=100)
    learn_parser.add_argument("--force", action="store_true")
    learn_parser.add_argument("--continuous", action="store_true")
    learn_parser.add_argument("--poll-seconds", type=float, default=5.0)
    learn_parser.add_argument("--log-every", type=int, default=10)
    learn_parser.add_argument("--device")
    learn_parser.set_defaults(handler=command_learn)

    train_parser = subparsers.add_parser("train", help="alternate self-play and learning")
    train_parser.add_argument("--config", default="fisher_config.json")
    train_parser.add_argument("--cycles", type=int, default=1)
    train_parser.add_argument("--games-per-cycle", type=int, default=64)
    train_parser.add_argument("--steps-per-cycle", type=int, default=100)
    train_parser.add_argument("--force", action="store_true")
    train_parser.set_defaults(handler=command_train)

    workstation_parser = subparsers.add_parser(
        "workstation",
        help="run continuous dual-GPU self-play and learning",
    )
    workstation_parser.add_argument("--config", default="fisher_config.json")
    workstation_parser.add_argument("--actors", type=int)
    workstation_parser.add_argument("--games-per-actor", type=int)
    workstation_parser.add_argument("--devices", nargs="+")
    workstation_parser.add_argument("--steps-per-burst", type=int, default=100)
    workstation_parser.set_defaults(handler=command_workstation)

    evaluate_parser = subparsers.add_parser("evaluate", help="play a checkpoint match")
    evaluate_parser.add_argument("--config", default="fisher_config.json")
    evaluate_parser.add_argument("--checkpoint-a", required=True)
    evaluate_parser.add_argument("--checkpoint-b", required=True)
    evaluate_parser.add_argument("--games", type=int, default=20)
    evaluate_parser.add_argument("--device")
    evaluate_parser.set_defaults(handler=command_evaluate)

    uci_parser = subparsers.add_parser("uci", help="run Fisher AI as a UCI engine")
    uci_parser.add_argument("--config", default="fisher_config.json")
    uci_parser.add_argument("--checkpoint")
    uci_parser.add_argument("--device")
    uci_parser.set_defaults(handler=command_uci)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.handler(args)
