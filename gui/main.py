import pygame, time, chess
from hf_model import ChessEngine, ChessTree

class ChessGUI:
    def __init__(self):
        pygame.init()
        self.width = 600
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.screen.fill((255, 255, 255))
        pygame.display.set_caption('Chess GUI')
        self.clock = pygame.time.Clock()

        self.cell_size = 60
        self.border_offset = 60
        self.grid = [[[] for i in range(8)] for i in range(8)]
        self.piece_sheet = pygame.image.load('icons/chess_pieces.png').convert_alpha()
        self.board_colours = (0, 128, 128, 230), (0, 0, 0, 230)

        # font
        self.font_size = 28
        font_offset = self.cell_size - (self.font_size//2)-5
        self.font = pygame.font.Font(None, 28)

        # draw ranks on left (omit rank 1 to avoid collision), files on bottom
        self.board_text = {}
        for row in range(7):
            rank = str(8 - row)
            self.board_text[(0, row)] = [self.font.render(rank, True, self.board_colours[row % 2]), 0, 0]
        for col in range(8):
            file = 'abcdefgh'[col]
            self.board_text[(col, 7)] = [self.font.render(file, True, self.board_colours[col % 2]), font_offset, font_offset+5]

        self.selected_square = None

        # Setup piece images
        self.piece_offsets = {
            'p':(3, 59, 47, 98), 'k':(72, 10, 53, 146), 'q':(143, 22, 54, 134), 'b':(218, 33, 48, 124),
            'r':(284, 51, 59, 106), 'n':(356, 48, 58, 109), 'P':(3, 205, 47, 97), 'K':(72, 156, 53, 146),
            'Q':(143, 168, 54, 134), 'B':(218, 179, 48, 123), 'R':(284, 197, 59, 105), 'N':(356, 194, 58, 108)
        }

        self.pieces = {}
        for name, coords in self.piece_offsets.items():
            x,y,w,h = coords
            frame = self.piece_sheet.subsurface(pygame.Rect(x, y, w, h)).copy()
            scale = min((self.cell_size - 10) / w, (self.cell_size - 10) / h)
            self.pieces[name] = pygame.transform.smoothscale(frame, (int(w * scale), int(h * scale)))

        # engine + truth board
        self.engine = ChessEngine(player='black')
        if not hasattr(self.engine, 'game_board'):
            self.engine.game_board = chess.Board()
        self.engine.set_fen(self.engine.game_board.fen())

        self.human_color = chess.WHITE

        # bot knobs
        self.bot_max_time = 2.0
        self.bot_max_plies = 4
        self.bot_batch_size = 32

        # Draw board
        self.add_background_img()
        self.redraw()

    def add_background_img(self):
        self.bg_img = pygame.image.load('icons/blue-marble.jpg')
        self.bg_img = pygame.transform.scale(self.bg_img, (self.width, self.height))
        self.screen.blit(self.bg_img, (0, 0))

    def square_name(self, col, row):
        file = 'abcdefgh'[col]
        rank = str(8 - row)
        return file + rank

    def piece_symbol_at(self, col, row):
        sq = chess.parse_square(self.square_name(col, row))
        p = self.engine.game_board.piece_at(sq)
        if not p:
            return '.'
        return p.symbol()

    def redraw(self):
        self.screen.blit(self.bg_img, (0, 0))
        self.draw_board()

    def draw_board(self):
        for i in range(8):
            for j in range(8):
                x = (i * self.cell_size) + self.border_offset
                y = (j * self.cell_size) + self.border_offset

                s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                color = self.board_colours[(i + j) % 2 == 0]
                s.fill(color)
                self.screen.blit(s, (x, y))
                self.grid[i][j] = [s, color]

                if (i,j) in self.board_text:
                    txt,offset_y,offset_x = self.board_text[(i,j)]
                    self.screen.blit(txt, (x + offset_x, y + offset_y))

                piece_name = self.piece_symbol_at(i, j)
                if piece_name != '.':
                    w,h = self.pieces[piece_name].get_size()
                    self.screen.blit(self.pieces[piece_name], (x + (self.cell_size - w) // 2, y + (self.cell_size - h) // 2))

    def highlight_square(self, col, row):
        x = (col * self.cell_size) + self.border_offset
        y = (row * self.cell_size) + self.border_offset
        s = self.grid[col][row][0]
        s.fill((255, 105, 180, 230))
        self.screen.blit(s, (x, y))
        self.grid[col][row][1] = (255, 105, 180, 230)

        piece_name = self.piece_symbol_at(col, row)
        if piece_name != '.':
            w,h = self.pieces[piece_name].get_size()
            self.screen.blit(self.pieces[piece_name], (x + (self.cell_size - w) // 2, y + (self.cell_size - h) // 2))

    def bot_move(self):
        if self.engine.game_board.is_game_over():
            return
        if self.engine.game_board.turn == self.human_color:
            return

        self.engine.set_fen(self.engine.game_board.fen())

        uci = None

        if hasattr(self.engine, 'search_bfs') and hasattr(self.engine, 'best_tree_move'):
            self.engine.tree = ChessTree(self.engine.game_board.fen())
            self.engine.search_bfs(max_plies=self.bot_max_plies, batch_size=self.bot_batch_size, max_time=self.bot_max_time, play_best=False)

            info = self.engine.best_tree_move(
                max_depth=self.bot_max_plies,
                agg='pessimistic',
                depth_bonus=0.10,
                depth_penalty=0.10,
                use_path_min=True
            )

            per_move = info.get('per_move', {}) if info else {}
            if per_move:
                if self.engine.game_board.turn == chess.WHITE:
                    uci = max(per_move, key=lambda m: per_move[m])
                else:
                    uci = min(per_move, key=lambda m: per_move[m])

        if not uci and hasattr(self.engine, 'get_move_output'):
            player = 'white' if self.engine.game_board.turn == chess.WHITE else 'black'
            output, _ = self.engine.get_move_output(player=player)
            parsed = self.engine.parse_output(output, board=self.engine.board)
            if parsed:
                uci = parsed['candidates'][0]['move']

        if not uci:
            return

        move = chess.Move.from_uci(uci)
        if move in self.engine.game_board.legal_moves:
            self.engine.game_board.push(move)
            self.engine.set_fen(self.engine.game_board.fen())

    def select_square(self):
        mx, my = pygame.mouse.get_pos()
        col = (mx - self.border_offset) // self.cell_size
        row = (my - self.border_offset) // self.cell_size

        if not (0 <= col <= 7 and 0 <= row <= 7):
            return

        if self.engine.game_board.is_game_over():
            return

        if self.engine.game_board.turn != self.human_color:
            return

        moved = False

        if self.selected_square:
            x,y = self.selected_square
            from_sq = self.square_name(x, y)
            to_sq = self.square_name(col, row)

            uci = from_sq + to_sq
            move = chess.Move.from_uci(uci)

            from_piece = self.engine.game_board.piece_at(chess.parse_square(from_sq))
            if from_piece and from_piece.piece_type == chess.PAWN and to_sq[1] in ['1', '8']:
                move_q = chess.Move.from_uci(uci + 'q')
                if move_q in self.engine.game_board.legal_moves:
                    move = move_q

            if move in self.engine.game_board.legal_moves:
                self.engine.game_board.push(move)
                self.engine.set_fen(self.engine.game_board.fen())
                moved = True

            self.selected_square = None

        self.redraw()

        if moved:
            self.bot_move()
            self.redraw()
            return

        sq = chess.parse_square(self.square_name(col, row))
        piece = self.engine.game_board.piece_at(sq)
        if not piece:
            return
        if piece.color != self.human_color:
            return

        self.highlight_square(col, row)
        self.selected_square = (col, row)

    def main_loop(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.select_square()

            pygame.display.flip()
            self.clock.tick(60)

if __name__ == '__main__':
    gui = ChessGUI()
    gui.main_loop()
