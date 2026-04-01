import pygame, chess
from ai_model import MCTSEngine

class ChessGUI:
    def __init__(self):
        pygame.init()
        self.width = 600
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.screen.fill((255, 255, 255))
        pygame.display.set_caption('Chess GUI')
        self.cell_size = 60
        self.border_offset = 60
        self.grid = [[[] for i in range(8)] for i in range(8)]
        self.piece_sheet = pygame.image.load('icons/chess_pieces.png').convert_alpha()
        self.board_colours = (0, 128, 128, 230), (0, 0, 0, 230)
        self.select_colour = (255, 105, 180, 230)
        self.move_colour = (255, 255, 255, 90)

        self.font_size = 28
        self.font = pygame.font.Font(None, self.font_size)
        self.font_offset = self.cell_size - (self.font_size // 2) - 5

        self.board = chess.Board()
        self.engine = MCTSEngine()
        self.selected_square = None
        self.legal_targets = set()

        self.piece_offsets = {
            'p': (3, 59, 47, 98), 'k': (72, 10, 53, 146), 'q': (143, 22, 54, 134), 'b': (218, 33, 48, 124),
            'r': (284, 51, 59, 106), 'n': (356, 48, 58, 109), 'P': (3, 205, 47, 97), 'K': (72, 156, 53, 146),
            'Q': (143, 168, 54, 134), 'B': (218, 179, 48, 123), 'R': (284, 197, 59, 105), 'N': (356, 194, 58, 108)
        }

        self.pieces = {}
        for name, coords in self.piece_offsets.items():
            x, y, w, h = coords
            frame = self.piece_sheet.subsurface(pygame.Rect(x, y, w, h)).copy()
            scale = min((self.cell_size - 10) / w, (self.cell_size - 10) / h)
            self.pieces[name] = pygame.transform.smoothscale(frame, (int(w * scale), int(h * scale)))

        self.add_background_img()
        self.draw_board()

    def add_background_img(self):
        self.bg_img = pygame.image.load('icons/blue-marble.jpg')
        self.bg_img = pygame.transform.scale(self.bg_img, (self.width, self.height))
        self.screen.blit(self.bg_img, (0, 0))

    def screen_to_square(self, mx, my):
        if mx < self.border_offset or my < self.border_offset:
            return None
        if mx >= self.width - self.border_offset or my >= self.height - self.border_offset:
            return None
        col = (mx - self.border_offset) // self.cell_size
        row = (my - self.border_offset) // self.cell_size
        rank = 7 - row
        return chess.square(col, rank)

    def square_to_screen(self, square):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        x = (file_index * self.cell_size) + self.border_offset
        y = ((7 - rank_index) * self.cell_size) + self.border_offset
        return x, y

    def get_move(self, from_sq, to_sq):
        move = chess.Move(from_sq, to_sq)
        if move in self.board.legal_moves:
            return move

        piece = self.board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            target_rank = chess.square_rank(to_sq)
            if target_rank == 0 or target_rank == 7:
                move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                if move in self.board.legal_moves:
                    return move
        return None

    def update_selection(self, square):
        piece = self.board.piece_at(square)
        if piece and piece.color == self.board.turn:
            self.selected_square = square
            self.legal_targets = {move.to_square for move in self.board.legal_moves if move.from_square == square}
        else:
            self.selected_square = None
            self.legal_targets = set()

    def draw_labels(self):
        for row in range(8):
            colour = self.board_colours[row % 2]
            text = self.font.render(str(8 - row), True, colour)
            x = self.border_offset + 4
            y = self.border_offset + (row * self.cell_size)
            self.screen.blit(text, (x, y))

        for col in range(8):
            colour = self.board_colours[(col + 1) % 2]
            text = self.font.render(chr(ord('a') + col), True, colour)
            x = self.border_offset + (col * self.cell_size) + self.font_offset
            y = self.border_offset + (7 * self.cell_size) + self.font_offset + 5
            self.screen.blit(text, (x, y))

    def draw_board(self):
        self.screen.blit(self.bg_img, (0, 0))

        for file_index in range(8):
            for rank_index in range(8):
                x = (file_index * self.cell_size) + self.border_offset
                y = (rank_index * self.cell_size) + self.border_offset
                s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                color = self.board_colours[(file_index + rank_index) % 2 == 1]
                s.fill(color)
                self.screen.blit(s, (x, y))
                self.grid[file_index][rank_index] = [s, color]

        if self.selected_square is not None:
            x, y = self.square_to_screen(self.selected_square)
            s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            s.fill(self.select_colour)
            self.screen.blit(s, (x, y))

        for square in self.legal_targets:
            x, y = self.square_to_screen(square)
            s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            s.fill(self.move_colour)
            self.screen.blit(s, (x, y))

        self.draw_labels()

        for square, piece in self.board.piece_map().items():
            x, y = self.square_to_screen(square)
            image = self.pieces[piece.symbol()]
            w, h = image.get_size()
            self.screen.blit(image, (x + (self.cell_size - w) // 2, y + (self.cell_size - h) // 2))

    def make_engine_move(self):
        if self.board.is_game_over() or self.board.turn != chess.BLACK:
            return
        move = self.engine.search(self.board)
        if move:
            self.board.push(move)

    def select_square(self):
        if self.board.is_game_over() or self.board.turn != chess.WHITE:
            return

        square = self.screen_to_square(*pygame.mouse.get_pos())
        if square is None:
            self.selected_square = None
            self.legal_targets = set()
            return

        if self.selected_square is None:
            self.update_selection(square)
            return

        if square == self.selected_square:
            self.selected_square = None
            self.legal_targets = set()
            return

        move = self.get_move(self.selected_square, square)
        if move:
            self.board.push(move)
            self.selected_square = None
            self.legal_targets = set()
            self.make_engine_move()
            return

        self.update_selection(square)

    def main_loop(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.select_square()

            self.draw_board()
            pygame.display.flip()

if __name__ == '__main__':
    gui = ChessGUI()
    gui.main_loop()
