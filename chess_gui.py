import pygame, torch
from ai_model import FisherAI, DEVICE

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
            8.,   8.,  8.,  8.,  1.,  8.,   8.,  8.,  
            1.,   1.,  1.,  1.,  1.,  1.,   1.,  1.,  
            1.,   1.,  1.,  1.,  8.,  1.,   1.,  1.,  
            1.,   1.,  1.,  1.,  1.,  1.,   1.,  1.,  
            1.,   1.,  1.,  1.,  1.,  1.,   1.,  1., 
            2.,   2.,  2.,  2.,  2.,  2.,   2.,  2.,
            5.,   3.,  4.,  6.,  7.,  4.,   3.,  5.
        ]], dtype=torch.long)

        # Setup piece images
        piece_offsets = {
            'w_pawn':(3, 59, 47, 98), 'w_king':(72, 10, 53, 146), 'w_queen':(143, 22, 54, 134), 'w_bishop':(218, 33, 48, 124), 
            'w_rook':(284, 51, 59, 106), 'w_knight':(356, 48, 58, 109), 'b_pawn':(3, 205, 47, 97), 'b_king':(72, 156, 53, 146), 
            'b_queen':(143, 168, 54, 134), 'b_bishop':(218, 179, 48, 123), 'b_rook':(284, 197, 59, 105), 'b_knight':(356, 194, 58, 108)
        }

        self.pieces = {}
        for name, coords in piece_offsets.items():
            x,y,w,h = coords
            frame = self.piece_sheet.subsurface(pygame.Rect(x, y, w, h)).copy()
            scale = min((self.cell_size - 10) / w, (self.cell_size - 10) / h)
            self.pieces[name] = pygame.transform.smoothscale(frame, (int(w * scale), int(h * scale)))

        # Draw board
        self.add_background_img()
        self.draw_board()

        self.ai.predict(self.board_array)
    
    def draw_board(self):
        ''' Change all cells to default colors '''
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
                w,h = self.pieces['w_pawn'].get_size()
                self.screen.blit(self.pieces['w_pawn'], (x + (self.cell_size - w) // 2, y + (self.cell_size - h) // 2))

    def add_background_img(self):
        ''' Adds a background image to the chess board '''
        self.bg_img = pygame.image.load('icons/blue-marble.jpg')
        self.bg_img = pygame.transform.scale(self.bg_img, (self.width, self.height))
        self.screen.blit(self.bg_img, (0, 0))
    
    def select_square(self):
        ''' Update selected square'''
        mx, my = pygame.mouse.get_pos()

        # update board cell colours on selection
        self.screen.blit(self.bg_img, (0, 0))
        for i in range(8):
            for j in range(8):
                s = self.grid[i][j][0]
                color = self.board_colours[(i + j) % 2 == 0]
                s.fill(color)
                self.screen.blit(s, ((i * self.cell_size) + self.border_offset, (j * self.cell_size) + self.border_offset))
                self.grid[i][j][1] = color

        if (mx > self.border_offset and my > self.border_offset) and \
            (mx < self.width - self.border_offset and my < self.height - self.border_offset):
            col = (mx - self.border_offset) // self.cell_size
            row = (my - self.border_offset) // self.cell_size
            print(f'Clicked on cell: ({col}, {row})')

            # Update selected square
            x = (col * self.cell_size) + self.border_offset
            y = (row * self.cell_size) + self.border_offset
            s = self.grid[col][row][0]
            s.fill((255, 105, 180, 230))
            self.screen.blit(s, (x, y))
            self.grid[col][row][1] = (255, 105, 180, 230)

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
