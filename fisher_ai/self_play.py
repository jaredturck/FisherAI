import numpy as np

from fisher_ai import chess
from fisher_ai.dataset import GameRecord
from fisher_ai.encoding import castling_rights_mask
from fisher_ai.game import MAX_GAME_PLIES, GameState
from fisher_ai.mcts import MAX_LEGAL_ACTIONS

EARLY_TEMPERATURE = 1.0
EARLY_TEMPERATURE_PLIES = 24
LATE_TEMPERATURE = 0.1


class SelfPlaySession:
    def __init__(self, tree):
        self.state = GameState()
        self.root = tree
        self.snapshot_bitboards = np.empty(
            (MAX_GAME_PLIES, 12),
            dtype=np.uint64,
        )
        self.snapshot_repetitions = np.empty(
            MAX_GAME_PLIES,
            dtype=np.uint8,
        )
        self.current_colors = np.empty(MAX_GAME_PLIES, dtype=np.bool_)
        self.plies = np.empty(MAX_GAME_PLIES, dtype=np.uint16)
        self.castling_masks = np.empty(MAX_GAME_PLIES, dtype=np.uint8)
        self.halfmove_clocks = np.empty(MAX_GAME_PLIES, dtype=np.uint8)
        self.legal_lengths = np.empty(MAX_GAME_PLIES, dtype=np.uint16)
        self.legal_actions = np.zeros(
            (MAX_GAME_PLIES, MAX_LEGAL_ACTIONS),
            dtype=np.uint16,
        )
        self.visit_counts = np.zeros(
            (MAX_GAME_PLIES, MAX_LEGAL_ACTIONS),
            dtype=np.uint16,
        )
        self.sample_count = 0
        self.finished = False

    def add_search_sample(self, actions, visit_counts):
        index = self.sample_count
        length = len(actions)
        self.state.current_bitboards(self.snapshot_bitboards[index])
        self.snapshot_repetitions[index] = self.state.repetition_count
        self.current_colors[index] = self.state.board.turn
        self.plies[index] = self.state.board.ply()
        self.castling_masks[index] = castling_rights_mask(self.state.board)
        self.halfmove_clocks[index] = self.state.board.halfmove_clock
        self.legal_lengths[index] = length
        self.legal_actions[index, :length] = actions
        self.visit_counts[index, :length] = visit_counts
        self.sample_count += 1

    def play_action(self, action):
        move = self.root.advance(action)
        self.state.push(move)
        self.finished = self.state.is_terminal()

    def build_record(self):
        count = self.sample_count
        result = self.state.final_result()
        values = np.where(
            self.current_colors[:count] == chess.WHITE,
            result,
            -result,
        ).astype(np.float32)
        return GameRecord(
            self.snapshot_bitboards[:count].copy(),
            self.snapshot_repetitions[:count].copy(),
            self.current_colors[:count].copy(),
            self.plies[:count].copy(),
            self.castling_masks[:count].copy(),
            self.halfmove_clocks[:count].copy(),
            self.legal_lengths[:count].copy(),
            self.legal_actions[:count].copy(),
            self.visit_counts[:count].copy(),
            values,
        )


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
