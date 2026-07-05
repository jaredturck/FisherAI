import queue

from fisher_ai import chess
from fisher_ai.game import GameState
from gui.main import (
    BOARD_X,
    BOARD_Y,
    PIECE_RENDER_OFFSETS,
    SQUARE_SIZE,
    ChessGUI,
)


def test_gui_square_mapping():
    gui = object.__new__(ChessGUI)

    assert gui.square_from_screen((BOARD_X + 1, BOARD_Y + 1)) == 56
    bottom_right = (
        BOARD_X + 7 * SQUARE_SIZE + 1,
        BOARD_Y + 7 * SQUARE_SIZE + 1,
    )
    assert gui.square_from_screen(bottom_right) == 7
    assert gui.square_from_screen((0, 0)) is None


def test_gui_defaults_promotion_to_queen():
    gui = object.__new__(ChessGUI)
    gui.state = GameState.from_fen("7k/P7/8/8/8/8/8/7K w - - 0 1")

    move = gui.choose_human_move(48, 56)

    assert move == chess.Move.from_uci("a7a8q")


def test_gui_has_offsets_for_every_piece():
    expected_pieces = {
        (color, piece_type)
        for color in (chess.WHITE, chess.BLACK)
        for piece_type in (
            chess.KING,
            chess.QUEEN,
            chess.BISHOP,
            chess.KNIGHT,
            chess.ROOK,
            chess.PAWN,
        )
    }

    assert set(PIECE_RENDER_OFFSETS) == expected_pieces
    assert PIECE_RENDER_OFFSETS[(chess.WHITE, chess.PAWN)] == (16, -4)
    assert PIECE_RENDER_OFFSETS[(chess.BLACK, chess.QUEEN)] == (-19, 1)


def test_gui_castles_by_clicking_king_destination_or_rook():
    gui = object.__new__(ChessGUI)
    gui.state = GameState.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")

    assert gui.choose_human_move(4, 6) == chess.Move.from_uci("e1g1")
    assert gui.choose_human_move(4, 7) == chess.Move.from_uci("e1g1")
    assert gui.choose_human_move(4, 2) == chess.Move.from_uci("e1c1")
    assert gui.choose_human_move(4, 0) == chess.Move.from_uci("e1c1")


def test_gui_marks_castling_destination_and_rook_as_clickable():
    gui = object.__new__(ChessGUI)
    gui.state = GameState.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    gui.selected_square = None
    gui.legal_destinations = set()

    gui.select_square(4)

    assert {0, 2, 6, 7}.issubset(gui.legal_destinations)


def test_gui_does_not_treat_two_square_rook_move_as_castling():
    gui = object.__new__(ChessGUI)
    gui.state = GameState.from_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")

    rook_move = chess.Move.from_uci("a1c1")

    assert gui.castling_rook_square(rook_move) is None


class FakeTree:
    def __init__(self, move):
        self.move = move
        self.requested_action = None

    def move_for_action(self, action):
        self.requested_action = action
        return self.move


class FakeSearch:
    def __init__(self, tree, action):
        self.tree = tree
        self.action = action
        self.run_arguments = None

    def create_tree(self):
        return self.tree

    def run(self, states, roots, add_noise, simulations):
        self.run_arguments = (states, roots, add_noise, simulations)
        return roots

    def choose_action(self, root, greedy):
        assert root is self.tree
        assert greedy is True
        return self.action


def test_gui_engine_move_uses_current_mcts_tree_api():
    gui = object.__new__(ChessGUI)
    state = GameState()
    move = chess.Move.from_uci("e7e5")
    tree = FakeTree(move)
    gui.search = FakeSearch(tree, action=123)
    gui.engine_simulations = 256
    gui.engine_results = queue.Queue()

    gui.compute_engine_move(state)

    states, roots, add_noise, simulations = gui.search.run_arguments
    assert states == [state]
    assert roots == [tree]
    assert add_noise is False
    assert simulations == 256
    assert tree.requested_action == 123
    assert gui.engine_results.get_nowait() == move
