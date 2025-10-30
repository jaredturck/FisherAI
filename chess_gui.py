import pygame

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

        self.add_background_img()
        self.draw_board()
    
    def draw_board(self):
        ''' Change all cells to default colors '''
        for i in range(8):
            for j in range(8):
                x = (i * self.cell_size) + self.border_offset
                y = (j * self.cell_size) + self.border_offset
                s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                color = (0,0,0,230) if (i + j) % 2 == 0 else (255,255,255,230)
                s.fill(color)
                self.screen.blit(s, (x, y))
                self.grid[i][j] = [s, color]

    def add_background_img(self):
        ''' Adds a background image to the chess board '''
        self.bg_img = pygame.image.load('icons/blue-marble.jpg')
        self.bg_img = pygame.transform.scale(self.bg_img, (self.width, self.height))
        self.screen.blit(self.bg_img, (0, 0))
    
    def select_square(self):
        ''' Update selected square'''
        mx, my = pygame.mouse.get_pos()
        if (mx > self.border_offset and my > self.border_offset) and \
            (mx < self.width - self.border_offset and my < self.height - self.border_offset):
            col = (mx - self.border_offset) // self.cell_size
            row = (my - self.border_offset) // self.cell_size
            print(f'Clicked on cell: ({col}, {row})')

            # update board cell colours on selection
            self.screen.blit(self.bg_img, (0, 0))
            for i in range(8):
                for j in range(8):
                    s = self.grid[i][j][0]
                    color = (0,0,0,230) if (i + j) % 2 == 0 else (255,255,255,230)
                    s.fill(color)
                    self.screen.blit(s, ((i * self.cell_size) + self.border_offset, (j * self.cell_size) + self.border_offset))
                    self.grid[i][j][1] = color

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
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.select_square()

            pygame.display.flip()

if __name__ == '__main__':
    gui = ChessGUI()
    gui.main_loop()
