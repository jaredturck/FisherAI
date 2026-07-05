import numpy as np

from fisher_ai import chess
from fisher_ai.dataset import GameRecord, PositionTarget
from fisher_ai.encoding import castling_rights_mask
from fisher_ai.game import GameState

EARLY_TEMPERATURE = 1.0
EARLY_TEMPERATURE_PLIES = 24
LATE_TEMPERATURE = 0.1


class SelfPlaySession:
    def __init__(self, tree):
        self.state = GameState()
        self.root = tree
        self.snapshots = [self.state.history[-1]]
        self.samples = []
        self.finished = False

    def add_search_sample(self, actions, visit_counts):
        self.samples.append(
            PositionTarget(
                self.state.board.turn,
                self.state.board.ply(),
                castling_rights_mask(self.state.board),
                self.state.board.halfmove_clock,
                actions,
                visit_counts,
            )
        )

    def play_action(self, action):
        move = self.root.advance(action)
        self.state.push(move)
        self.snapshots.append(self.state.history[-1])
        self.finished = self.state.is_terminal()

    def build_record(self):
        result = self.state.final_result()
        for sample in self.samples:
            sample.value = (
                result if sample.current_color == chess.WHITE else -result
            )
        return GameRecord(self.snapshots, self.samples)


class SelfPlayRunner:
    def __init__(self, mcts, seed=7):
        self.mcts = mcts
        self.rng = np.random.default_rng(seed)

    def create_session(self):
        return SelfPlaySession(self.mcts.create_tree())

    def advance_sessions(self, sessions):
        active_indices = [
            index
            for index, session in enumerate(sessions)
            if not session.finished
        ]
        if not active_indices:
            return []

        active = [sessions[index] for index in active_indices]
        states = [session.state for session in active]
        roots = [session.root for session in active]
        searched_roots = self.mcts.run(
            states,
            roots=roots,
            add_noise=True,
        )

        finished_indices = []
        for session_index, session, root in zip(
            active_indices,
            active,
            searched_roots,
            strict=True,
        ):
            actions, counts = self.mcts.visit_counts(root)
            counts = np.clip(
                counts,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
            session.add_search_sample(actions.astype(np.uint16), counts)

            temperature = (
                EARLY_TEMPERATURE
                if session.state.board.ply() < EARLY_TEMPERATURE_PLIES
                else LATE_TEMPERATURE
            )
            action = self.mcts.choose_action(root, temperature=temperature)
            session.play_action(action)
            if session.finished:
                finished_indices.append(session_index)

        return finished_indices
