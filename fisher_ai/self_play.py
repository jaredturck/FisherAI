import numpy as np

from fisher_ai import chess
from fisher_ai.encoding import encode_state
from fisher_ai.game import GameState
from fisher_ai.mcts import MCTSTree
from fisher_ai.replay import GameRecord, TrainingSample


class SelfPlaySession:
    def __init__(
        self,
        max_game_plies,
        tree_capacity,
        checkpoint_step=0,
        audit_resignation=False,
    ):
        self.state = GameState(max_game_plies=max_game_plies)
        self.root = MCTSTree(tree_capacity)
        self.pending_samples = []
        self.moves = []
        self.finished = False
        self.forced_result = None
        self.checkpoint_step = int(checkpoint_step)
        self.audit_resignation = audit_resignation
        self.resignation_streak = 0

    def add_search_sample(self, actions, visit_counts, policy_weight, encoded_state=None):
        self.pending_samples.append(
            {
                "state": encoded_state if encoded_state is not None else encode_state(self.state),
                "legal_actions": actions,
                "visit_counts": visit_counts,
                "policy_weight": policy_weight,
                "player": self.state.board.turn,
            }
        )

    def play_action(self, action):
        move = self.root.advance(action)
        self.moves.append(move.uci())
        self.state.push(move)
        self.finished = self.state.is_terminal()

    def resign(self):
        self.forced_result = -1 if self.state.board.turn == chess.WHITE else 1
        self.finished = True

    def build_record(self, checkpoint_step=None):
        if checkpoint_step is None:
            checkpoint_step = self.checkpoint_step

        result = self.forced_result if self.forced_result is not None else self.state.final_result()
        samples = []

        for pending in self.pending_samples:
            value = result if pending["player"] == chess.WHITE else -result
            samples.append(
                TrainingSample(
                    pending["state"],
                    pending["legal_actions"],
                    pending["visit_counts"],
                    value=value,
                    policy_weight=pending["policy_weight"],
                )
            )

        return GameRecord(
            samples=samples,
            moves=self.moves,
            result=result,
            checkpoint_step=checkpoint_step,
        )


class SelfPlayRunner:
    def __init__(self, mcts, search_config, training_config=None, seed=7):
        self.mcts = mcts
        self.search_config = search_config
        self.rng = np.random.default_rng(seed)
        self.resignation_threshold = -0.9
        self.resignation_consecutive_moves = 3
        self.resignation_audit_fraction = 0.1

        if training_config is not None:
            self.resignation_threshold = training_config.resignation_threshold
            self.resignation_consecutive_moves = training_config.resignation_consecutive_moves
            self.resignation_audit_fraction = training_config.resignation_audit_fraction

    def create_session(self, checkpoint_step=0, allow_resignation=False):
        audit_resignation = (
            allow_resignation and self.rng.random() < self.resignation_audit_fraction
        )
        return SelfPlaySession(
            self.search_config.max_game_plies,
            self.search_config.tree_capacity,
            checkpoint_step=checkpoint_step,
            audit_resignation=audit_resignation,
        )

    def advance_sessions(self, sessions, allow_resignation=False):
        active_indices = [index for index, session in enumerate(sessions) if not session.finished]
        if not active_indices:
            return []

        active = [sessions[index] for index in active_indices]
        full_search_flags = [
            self.rng.random() < self.search_config.full_search_fraction for _ in active
        ]
        simulations = [
            self.search_config.simulations
            if full_search
            else self.search_config.fast_simulations
            for full_search in full_search_flags
        ]
        states = [session.state for session in active]
        roots = [session.root for session in active]
        searched_roots = self.mcts.run(
            states,
            roots=roots,
            add_noise=True,
            simulations=simulations,
        )

        finished_indices = []
        for session_index, session, root, full_search in zip(
            active_indices,
            active,
            searched_roots,
            full_search_flags,
            strict=True,
        ):
            actions, counts = self.mcts.visit_counts(root)
            counts_uint16 = np.clip(counts, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            session.add_search_sample(
                actions.astype(np.uint16),
                counts_uint16,
                policy_weight=1.0 if full_search else 0.0,
                encoded_state=root.encoded_state,
            )

            if full_search and root.mean_value <= self.resignation_threshold:
                session.resignation_streak += 1
            else:
                session.resignation_streak = 0

            if (
                allow_resignation
                and not session.audit_resignation
                and session.resignation_streak >= self.resignation_consecutive_moves
            ):
                session.resign()
                finished_indices.append(session_index)
                continue

            temperature = (
                self.search_config.temperature
                if session.state.board.ply() < self.search_config.temperature_plies
                else self.search_config.late_temperature
            )
            action = self.mcts.choose_action(
                root,
                temperature=temperature,
                greedy=False,
            )
            session.play_action(action)
            if session.finished:
                finished_indices.append(session_index)

        return finished_indices

    def play_games(self, game_count, checkpoint_step=0, allow_resignation=False):
        sessions = [
            self.create_session(
                checkpoint_step=checkpoint_step,
                allow_resignation=allow_resignation,
            )
            for _ in range(game_count)
        ]

        while any(not session.finished for session in sessions):
            self.advance_sessions(sessions, allow_resignation=allow_resignation)

        return [session.build_record() for session in sessions]
