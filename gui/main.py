import queue
import threading
from pathlib import Path

import numpy as np
import pygame

from fisher_ai import chess
from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import available_device, load_config
from fisher_ai.game import GameState
from fisher_ai.mcts import MCTS, TorchEvaluator
from fisher_ai.network import FisherNetwork

ROOT_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = Path(__file__).resolve().parent / "assets"
CONFIG_PATH = ROOT_DIR / "fisher_config.json"
BACKGROUND_PATH = ASSET_DIR / "marble.jpeg"
SPRITE_PATH = ASSET_DIR / "chess_sprite_sheet.png"

WINDOW_WIDTH = 1060
WINDOW_HEIGHT = 800
BOARD_X = 32
BOARD_Y = 40
SQUARE_SIZE = 90
BOARD_SIZE = SQUARE_SIZE * 8
PANEL_X = BOARD_X + BOARD_SIZE + 28
PANEL_WIDTH = WINDOW_WIDTH - PANEL_X - 28
PANEL_PADDING = 18
PANEL_RADIUS = 14
BUTTON_RADIUS = 9
FOOTER_GAP = 12

HUMAN_COLOR = chess.WHITE
ENGINE_COLOR = chess.BLACK

PIECE_COLUMNS = {
    chess.KING: 0,
    chess.QUEEN: 1,
    chess.BISHOP: 2,
    chess.KNIGHT: 3,
    chess.ROOK: 4,
    chess.PAWN: 5,
}

PIECE_RENDER_OFFSETS = {
    (chess.WHITE, chess.KING): (-18, -4),
    (chess.WHITE, chess.QUEEN): (-19, -4),
    (chess.WHITE, chess.BISHOP): (-12, -4),
    (chess.WHITE, chess.KNIGHT): (1, -4),
    (chess.WHITE, chess.ROOK): (8, -4),
    (chess.WHITE, chess.PAWN): (16, -4),
    (chess.BLACK, chess.KING): (-18, 0),
    (chess.BLACK, chess.QUEEN): (-19, 1),
    (chess.BLACK, chess.BISHOP): (-10, 0),
    (chess.BLACK, chess.KNIGHT): (1, 0),
    (chess.BLACK, chess.ROOK): (8, 0),
    (chess.BLACK, chess.PAWN): (16, 1),
}

LIGHT_SQUARE = (211, 224, 238, 215)
DARK_SQUARE = (30, 79, 126, 225)
SELECTED_SQUARE = (227, 178, 63, 230)
LEGAL_MOVE = (79, 179, 126, 210)
LAST_MOVE = (78, 137, 188, 180)
CHECK_SQUARE = (191, 55, 65, 220)
PANEL_COLOR = (7, 26, 49, 220)
TEXT_COLOR = (238, 242, 247)
MUTED_TEXT = (172, 190, 209)
ACCENT_TEXT = (238, 193, 89)


class ChessGUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("FisherAI Chess")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.Font(None, 42)
        self.body_font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 23)
        self.coordinate_font = pygame.font.Font(None, 21)

        self.background = self.load_background()
        self.piece_sprites = self.load_piece_sprites()
        footer_height = self.small_font.get_height()
        self.footer_y = BOARD_Y + BOARD_SIZE - PANEL_PADDING - footer_height
        self.new_game_button = pygame.Rect(
            PANEL_X + PANEL_PADDING,
            self.footer_y - FOOTER_GAP - 48,
            PANEL_WIDTH - PANEL_PADDING * 2,
            48,
        )

        self.state = None
        self.search = None
        self.checkpoint_step = 0
        self.device = ""
        self.engine_simulations = 0
        self.selected_square = None
        self.legal_destinations = set()
        self.last_move = None
        self.move_history = []
        self.engine_thinking = False
        self.engine_results = queue.Queue()
        self.legal_move_buffer = np.empty(256, dtype=np.uint32)
        self.running = True
        self.status = "Loading FisherAI..."

        self.draw()
        pygame.display.flip()
        self.load_engine()
        self.new_game()

    def load_background(self):
        background = pygame.image.load(BACKGROUND_PATH).convert()
        return pygame.transform.smoothscale(
            background, (WINDOW_WIDTH, WINDOW_HEIGHT)
        )

    def load_piece_sprites(self):
        sheet = pygame.image.load(SPRITE_PATH).convert_alpha()
        sprites = {}

        sheet_width, sheet_height = sheet.get_size()
        for color, row in ((chess.WHITE, 0), (chess.BLACK, 1)):
            for piece_type, column in PIECE_COLUMNS.items():
                left = round(column * sheet_width / 6)
                right = round((column + 1) * sheet_width / 6)
                top = round(row * sheet_height / 2)
                bottom = round((row + 1) * sheet_height / 2)
                rect = pygame.Rect(left, top, right - left, bottom - top)
                sprite = sheet.subsurface(rect).copy()
                sprite = pygame.transform.smoothscale(
                    sprite, (SQUARE_SIZE, SQUARE_SIZE)
                )
                sprites[(color, piece_type)] = sprite

        return sprites

    def load_engine(self):
        config = load_config(CONFIG_PATH)
        manager = CheckpointManager(ROOT_DIR / "checkpoints")
        checkpoint_path = manager.latest_path()
        if checkpoint_path is None:
            self.status = "No checkpoint found in checkpoints/"
            self.draw()
            pygame.display.flip()
            return

        preferred_device = config.device
        self.device = available_device(preferred_device)
        model = FisherNetwork()
        self.checkpoint_step = manager.load(
            model,
            path=checkpoint_path,
            device="cpu",
        )

        evaluator = TorchEvaluator(
            model,
            device=self.device,
            inference_batch_size=config.inference_batch_size,
        )
        self.search = MCTS(
            evaluator,
            simulations=config.simulations,
            parallel_searches=config.parallel_searches,
            seed=1007,
        )
        self.engine_simulations = config.simulations
        self.status = "Your move"

    def new_game(self):
        if self.engine_thinking:
            return

        self.state = GameState()
        self.selected_square = None
        self.legal_destinations.clear()
        self.last_move = None
        self.move_history = []
        self.status = "Your move" if self.search else "No checkpoint loaded"

    def square_from_screen(self, position):
        mouse_x, mouse_y = position
        column = (mouse_x - BOARD_X) // SQUARE_SIZE
        row = (mouse_y - BOARD_Y) // SQUARE_SIZE
        if not 0 <= column < 8 or not 0 <= row < 8:
            return None
        return column + (7 - row) * 8

    def screen_from_square(self, square):
        column = chess.square_file(square)
        row = 7 - chess.square_rank(square)
        return BOARD_X + column * SQUARE_SIZE, BOARD_Y + row * SQUARE_SIZE

    def piece_at(self, square):
        piece_type = self.state.board.piece_type_at(square)
        if piece_type is None:
            return None

        square_mask = 1 << square
        color = (
            chess.WHITE
            if self.state.board.occupied_co[chess.WHITE] & square_mask
            else chess.BLACK
        )
        return color, piece_type

    def current_legal_moves(self):
        if not hasattr(self, "legal_move_buffer"):
            self.legal_move_buffer = np.empty(256, dtype=np.uint32)
        count, _ = self.state.board.fill_legal_moves(self.legal_move_buffer)
        return self.legal_move_buffer[:count]

    def legal_moves_from(self, square):
        return [
            int(move)
            for move in self.current_legal_moves()
            if chess.move_from_square(move) == square
        ]

    def castling_rook_square(self, move):
        if self.piece_at(chess.move_from_square(move)) != (
            self.state.board.turn,
            chess.KING,
        ):
            return None
        if abs(chess.move_to_square(move) - chess.move_from_square(move)) != 2:
            return None

        rank_start = chess.move_from_square(move) - chess.square_file(
            chess.move_from_square(move)
        )
        return rank_start + (
            7
            if chess.move_to_square(move) > chess.move_from_square(move)
            else 0
        )

    def choose_human_move(self, from_square, to_square):
        legal_moves = self.current_legal_moves()
        candidates = [
            move
            for move in legal_moves
            if chess.move_from_square(move) == from_square
            and chess.move_to_square(move) == to_square
        ]

        if not candidates:
            candidates = [
                move
                for move in legal_moves
                if chess.move_from_square(move) == from_square
                and self.castling_rook_square(move) == to_square
            ]

        if not candidates:
            return None

        for move in candidates:
            if chess.move_promotion(move) == chess.QUEEN:
                return move

        return candidates[0]

    def handle_board_click(self, position):
        if (
            self.search is None
            or self.engine_thinking
            or self.state.is_terminal()
        ):
            return
        if self.state.board.turn != HUMAN_COLOR:
            return

        square = self.square_from_screen(position)
        if square is None:
            self.clear_selection()
            return

        piece = self.piece_at(square)
        if self.selected_square is None:
            if piece and piece[0] == HUMAN_COLOR:
                self.select_square(square)
            return

        move = self.choose_human_move(self.selected_square, square)
        if move is not None:
            self.clear_selection()
            self.play_move(move)
            if not self.state.is_terminal():
                self.start_engine_move()
            return

        if piece and piece[0] == HUMAN_COLOR:
            self.select_square(square)
        else:
            self.clear_selection()

    def select_square(self, square):
        moves = self.legal_moves_from(square)
        self.selected_square = square
        self.legal_destinations = {
            chess.move_to_square(move) for move in moves
        }
        self.legal_destinations.update(
            rook_square
            for move in moves
            if (rook_square := self.castling_rook_square(move)) is not None
        )

    def clear_selection(self):
        self.selected_square = None
        self.legal_destinations.clear()

    def play_move(self, move):
        self.state.push(move)
        self.last_move = move
        self.move_history.append(chess.move_to_uci(move))
        self.update_status()

    def start_engine_move(self):
        self.engine_thinking = True
        self.status = "FisherAI is thinking..."
        state = self.state.copy()
        thread = threading.Thread(
            target=self.compute_engine_move, args=(state,), daemon=True
        )
        thread.start()

    def compute_engine_move(self, state):
        root = self.search.create_tree()
        root = self.search.run(
            [state],
            roots=[root],
            add_noise=False,
        )[0]
        action = self.search.choose_action(root, greedy=True)
        self.engine_results.put(root.move_for_action(action))

    def process_engine_result(self):
        if self.engine_results.empty():
            return

        move = self.engine_results.get_nowait()
        self.engine_thinking = False
        self.play_move(move)

    def update_status(self):
        terminal_status = self.state.terminal_status()
        if terminal_status == chess.CHECKMATE:
            winner = (
                "You win"
                if self.state.board.turn == ENGINE_COLOR
                else "FisherAI wins"
            )
            self.status = f"Checkmate — {winner}"
        elif terminal_status != chess.ONGOING:
            self.status = "Draw"
        elif self.state.board.is_check():
            self.status = (
                "Check — your move"
                if self.state.board.turn == HUMAN_COLOR
                else "Check"
            )
        elif self.state.board.turn == HUMAN_COLOR:
            self.status = "Your move"
        else:
            self.status = "FisherAI is thinking..."

    def draw(self):
        self.screen.blit(self.background, (0, 0))
        self.draw_board()
        self.draw_panel()

    def draw_board(self):
        check_square = None
        if self.state and self.state.board.is_check():
            check_square = self.state.board.king(self.state.board.turn)

        for row in range(8):
            for column in range(8):
                square = column + (7 - row) * 8
                x = BOARD_X + column * SQUARE_SIZE
                y = BOARD_Y + row * SQUARE_SIZE
                color = (
                    LIGHT_SQUARE if (row + column) % 2 == 0 else DARK_SQUARE
                )

                if self.last_move and square in (
                    chess.move_from_square(self.last_move),
                    chess.move_to_square(self.last_move),
                ):
                    color = LAST_MOVE
                if square == self.selected_square:
                    color = SELECTED_SQUARE
                if square == check_square:
                    color = CHECK_SQUARE

                surface = pygame.Surface(
                    (SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA
                )
                surface.fill(color)
                self.screen.blit(surface, (x, y))

                if square in self.legal_destinations:
                    self.draw_legal_marker(square)

                if self.state:
                    piece = self.piece_at(square)
                    if piece:
                        offset_x, offset_y = PIECE_RENDER_OFFSETS[piece]
                        self.screen.blit(
                            self.piece_sprites[piece],
                            (x + offset_x, y + offset_y),
                        )

        self.draw_coordinates()
        pygame.draw.rect(
            self.screen,
            (214, 176, 92),
            pygame.Rect(
                BOARD_X - 3, BOARD_Y - 3, BOARD_SIZE + 6, BOARD_SIZE + 6
            ),
            width=3,
            border_radius=3,
        )

    def draw_legal_marker(self, square):
        x, y = self.screen_from_square(square)
        occupied = self.piece_at(square) is not None
        center = (x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2)

        if occupied:
            pygame.draw.circle(
                self.screen, LEGAL_MOVE, center, SQUARE_SIZE // 2 - 8, width=6
            )
        else:
            pygame.draw.circle(self.screen, LEGAL_MOVE, center, 11)

    def draw_coordinates(self):
        for column, file_name in enumerate("abcdefgh"):
            text = self.coordinate_font.render(file_name, True, TEXT_COLOR)
            x = BOARD_X + column * SQUARE_SIZE + SQUARE_SIZE - 16
            y = BOARD_Y + BOARD_SIZE - 22
            self.screen.blit(text, (x, y))

        for row in range(8):
            rank = str(8 - row)
            text = self.coordinate_font.render(rank, True, TEXT_COLOR)
            self.screen.blit(
                text, (BOARD_X + 5, BOARD_Y + row * SQUARE_SIZE + 4)
            )

    def draw_panel(self):
        panel = pygame.Surface((PANEL_WIDTH, BOARD_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(
            panel, PANEL_COLOR, panel.get_rect(), border_radius=PANEL_RADIUS
        )
        self.screen.blit(panel, (PANEL_X, BOARD_Y))

        content_x = PANEL_X + PANEL_PADDING
        self.draw_text("FisherAI", self.title_font, ACCENT_TEXT, content_x, 62)
        self.draw_text(self.status, self.body_font, TEXT_COLOR, content_x, 116)

        self.draw_text("Engine", self.body_font, ACCENT_TEXT, content_x, 172)
        self.draw_text(
            f"Checkpoint: {self.checkpoint_step:,}",
            self.small_font,
            TEXT_COLOR,
            content_x,
            205,
        )
        self.draw_text(
            f"Device: {self.device or 'not loaded'}",
            self.small_font,
            TEXT_COLOR,
            content_x,
            231,
        )
        self.draw_text(
            f"Search: {self.engine_simulations:,} simulations",
            self.small_font,
            TEXT_COLOR,
            content_x,
            257,
        )

        self.draw_text("Moves", self.body_font, ACCENT_TEXT, content_x, 312)
        self.draw_move_history()

        pygame.draw.rect(
            self.screen,
            (32, 82, 128),
            self.new_game_button,
            border_radius=BUTTON_RADIUS,
        )
        pygame.draw.rect(
            self.screen,
            (214, 176, 92),
            self.new_game_button,
            width=2,
            border_radius=BUTTON_RADIUS,
        )
        button_text = self.body_font.render("New Game", True, TEXT_COLOR)
        button_x = self.new_game_button.centerx - button_text.get_width() // 2
        button_y = self.new_game_button.centery - button_text.get_height() // 2
        self.screen.blit(button_text, (button_x, button_y))

        self.draw_text(
            "You play White",
            self.small_font,
            MUTED_TEXT,
            content_x,
            self.footer_y,
        )

    def draw_move_history(self):
        visible_moves = self.move_history[-18:]
        start_index = len(self.move_history) - len(visible_moves)

        for offset in range(0, len(visible_moves), 2):
            move_number = (start_index + offset) // 2 + 1
            white_move = visible_moves[offset]
            black_move = (
                visible_moves[offset + 1]
                if offset + 1 < len(visible_moves)
                else ""
            )
            line = f"{move_number:>3}.  {white_move:<6} {black_move}"
            self.draw_text(
                line,
                self.small_font,
                TEXT_COLOR,
                PANEL_X + PANEL_PADDING,
                346 + (offset // 2) * 31,
            )

    def draw_text(self, value, font, color, x, y):
        text = font.render(value, True, color)
        self.screen.blit(text, (x, y))

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            self.new_game()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.new_game_button.collidepoint(event.pos):
                self.new_game()
            else:
                self.handle_board_click(event.pos)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)

            self.process_engine_result()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main():
    gui = ChessGUI()
    gui.run()


if __name__ == "__main__":
    main()
