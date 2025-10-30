import pygame, torch
from ai_model import FisherAI, DEVICE, piece_lookup

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

        self.ai = FisherAI().to(DEVICE)
        self.ai.load_weights()

        self.board_array = torch.tensor([[
            11.,  9.,  10., 12., 13., 10.,  9.,  11.,  
            8.,   8.,  8.,  8.,  8.,  8.,   8.,  8.,  
            1.,   1.,  1.,  1.,  1.,  1.,   1.,  1.,  
            1.,   1.,  1.,  1.,  1.,  1.,   1.,  1.,  
            1.,   1.,  1.,  1.,  1.,  1.,   1.,  1.,  
            1.,   1.,  1.,  1.,  1.,  1.,   1.,  1., 
            2.,   2.,  2.,  2.,  2.,  2.,   2.,  2.,
            5.,   3.,  4.,  6.,  7.,  4.,   3.,  5.
        ]], dtype=torch.long)
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

        # Draw board
        self.add_background_img()
        self.draw_board()
    
    def draw_board(self):
        ''' Change all cells to default colors '''
        board_2d = self.board_array.reshape(8,8)
        for i in range(8):
            for j in range(8):
                x = (i * self.cell_size) + self.border_offset
                y = (j * self.cell_size) + self.border_offset
                s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                color = self.board_colours[(i + j) % 2 == 0]
                s.fill(color)
                self.screen.blit(s, (x, y))
                self.grid[i][j] = [s, color]

                # Add piece
                piece_name = piece_lookup[int(board_2d[j][i])]
                if piece_name != '.':
                    w,h = self.pieces[piece_name].get_size()
                    self.screen.blit(self.pieces[piece_name], (x + (self.cell_size - w) // 2, y + (self.cell_size - h) // 2))

    def add_background_img(self):
        ''' Adds a background image to the chess board '''
        self.bg_img = pygame.image.load('icons/blue-marble.jpg')
        self.bg_img = pygame.transform.scale(self.bg_img, (self.width, self.height))
        self.screen.blit(self.bg_img, (0, 0))
    
    def select_square(self):
        ''' Update selected square'''
        mx, my = pygame.mouse.get_pos()
        col = (mx - self.border_offset) // self.cell_size
        row = (my - self.border_offset) // self.cell_size
        board_2d = self.board_array.reshape(8,8)

        # Update move
        moved = False
        if self.selected_square:
            x,y = self.selected_square
            piece = board_2d[y, x].item()
            board_2d[y, x] = 1
            board_2d[row, col] = piece
            self.board_array = board_2d.reshape(1, -1)
            self.selected_square = None
            moved = True
            self.board_array = self.ai.predict(self.board_array)
            board_2d = self.board_array.reshape(8,8)

        # update board cell colours on selection
        self.screen.blit(self.bg_img, (0, 0))
        for i in range(8):
            for j in range(8):
                s = self.grid[i][j][0]
                color = self.board_colours[(i + j) % 2 == 0]
                s.fill(color)
                self.screen.blit(s, ((i * self.cell_size) + self.border_offset, (j * self.cell_size) + self.border_offset))
                self.grid[i][j][1] = color

                # Add piece
                x = (i * self.cell_size) + self.border_offset
                y = (j * self.cell_size) + self.border_offset
                piece_name = piece_lookup[int(board_2d[j][i])]
                if piece_name != '.':
                    w,h = self.pieces[piece_name].get_size()
                    self.screen.blit(self.pieces[piece_name], (x + (self.cell_size - w) // 2, y + (self.cell_size - h) // 2))

        if not moved and (mx > self.border_offset and my > self.border_offset) and \
            (mx < self.width - self.border_offset and my < self.height - self.border_offset):

            # Update selected square
            x = (col * self.cell_size) + self.border_offset
            y = (row * self.cell_size) + self.border_offset
            s = self.grid[col][row][0]
            s.fill((255, 105, 180, 230))
            self.screen.blit(s, (x, y))
            self.grid[col][row][1] = (255, 105, 180, 230)

            piece_name = piece_lookup[int(board_2d[row][col])]
            if piece_name != '.':
                w,h = self.pieces[piece_name].get_size()
                self.screen.blit(self.pieces[piece_name], (x + (self.cell_size - w) // 2, y + (self.cell_size - h) // 2))
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

if __name__ == '__main__':
    gui = ChessGUI()
    gui.main_loop()
