import math
import chess
import torch

class MCTSEngine:
    def __init__(self, simulations=800, c_puct=1.4):
        self.simulations = simulations
        self.c_puct = c_puct
        self.policy_size = 4672
        self.piece_values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
        self.direction_lookup = {(0, 1): 0, (0, -1): 1, (1, 0): 2, (-1, 0): 3, (1, 1): 4, (-1, 1): 5, (1, -1): 6, (-1, -1): 7}
        self.knight_lookup = {(1, 2): 0, (2, 1): 1, (2, -1): 2, (1, -2): 3, (-1, -2): 4, (-2, -1): 5, (-2, 1): 6, (-1, 2): 7}
        self.piece_planes = {
            (chess.WHITE, chess.PAWN): 0, (chess.WHITE, chess.KNIGHT): 1, (chess.WHITE, chess.BISHOP): 2,
            (chess.WHITE, chess.ROOK): 3, (chess.WHITE, chess.QUEEN): 4, (chess.WHITE, chess.KING): 5,
            (chess.BLACK, chess.PAWN): 6, (chess.BLACK, chess.KNIGHT): 7, (chess.BLACK, chess.BISHOP): 8,
            (chess.BLACK, chess.ROOK): 9, (chess.BLACK, chess.QUEEN): 10, (chess.BLACK, chess.KING): 11,
        }

    def search(self, board, simulations=None):
        root = {'move': None, 'prior': 1.0, 'visit_count': 0, 'value_sum': 0.0, 'mean_value': 0.0, 'children': []}
        simulations = simulations or self.simulations
        if not board.is_game_over():
            policy, _ = self.evaluate(board)
            self.expand(root, board, policy)
        for _ in range(simulations):
            self.simulate(board.copy(stack=False), root)

        self.last_simulations = simulations
        self.last_root_children = len(root['children'])

        if not root['children']:
            self.last_best_move = None
            return None

        best_move = max(root['children'], key=lambda child: child['visit_count'])['move']
        self.last_best_move = best_move
        return best_move

    def simulate(self, board, node):
        path = [node]
        while node['children'] and not board.is_game_over():
            child = max(node['children'], key=lambda child: self.puct(node, child))
            board.push(child['move'])
            node = child
            path.append(node)
        if board.is_game_over():
            outcome = board.outcome()
            value = 0.0 if outcome.winner is None else (1.0 if outcome.winner == board.turn else -1.0)
        else:
            policy, value = self.evaluate(board)
            self.expand(node, board, policy)
        for node in reversed(path):
            node['visit_count'] += 1
            node['value_sum'] += value
            node['mean_value'] = node['value_sum'] / node['visit_count']
            value = -value

    def evaluate(self, board):
        tensor = self.board_to_tensor(board)
        counts = tensor[:12].sum(dim=(1, 2))
        score = 0.0
        score += 100 * (counts[0] - counts[6]).item()
        score += 320 * (counts[1] - counts[7]).item()
        score += 330 * (counts[2] - counts[8]).item()
        score += 500 * (counts[3] - counts[9]).item()
        score += 900 * (counts[4] - counts[10]).item()
        for square in (chess.D4, chess.E4, chess.D5, chess.E5):
            piece = board.piece_at(square)
            if piece:
                score += 15 if piece.color == chess.WHITE else -15
        if board.has_kingside_castling_rights(chess.WHITE):
            score += 10
        if board.has_queenside_castling_rights(chess.WHITE):
            score += 10
        if board.has_kingside_castling_rights(chess.BLACK):
            score -= 10
        if board.has_queenside_castling_rights(chess.BLACK):
            score -= 10
        if board.turn == chess.BLACK:
            score = -score
        value = math.tanh(score / 1000.0)
        policy = torch.zeros(self.policy_size, dtype=torch.float32)
        for move in board.legal_moves:
            idx = self.move_to_index(board, move)
            if idx is None:
                continue
            piece = board.piece_at(move.from_square)
            prior = 1.0
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                captured_value = self.piece_values[chess.PAWN] if board.is_en_passant(move) else self.piece_values.get(captured.piece_type, 0)
                prior += 1.0 + captured_value / 1000.0
                prior -= self.piece_values.get(piece.piece_type, 0) / 10000.0
            if move.promotion == chess.QUEEN:
                prior += 5.0
            elif move.promotion:
                prior += 3.0
            if board.gives_check(move):
                prior += 1.5
            if board.is_castling(move):
                prior += 1.0
            file_index = chess.square_file(move.to_square)
            rank_index = chess.square_rank(move.to_square)
            if 2 <= file_index <= 5 and 2 <= rank_index <= 5:
                prior += 0.25
            policy[idx] = prior
        return policy, value

    def board_to_tensor(self, board):
        tensor = torch.zeros(18, 8, 8, dtype=torch.float32)
        for square, piece in board.piece_map().items():
            plane = self.piece_planes[(piece.color, piece.piece_type)]
            tensor[plane, chess.square_rank(square), chess.square_file(square)] = 1.0
        if board.turn == chess.WHITE:
            tensor[12].fill_(1.0)
        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[13].fill_(1.0)
        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[14].fill_(1.0)
        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[15].fill_(1.0)
        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[16].fill_(1.0)
        if board.ep_square is not None:
            tensor[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1.0
        return tensor

    def puct(self, parent, child):
        return child['mean_value'] + self.c_puct * child['prior'] * math.sqrt(parent['visit_count'] + 1) / (child['visit_count'] + 1)

    def expand(self, node, board, policy):
        if node['children']:
            return
        children = []
        total = 0.0
        for move in board.legal_moves:
            idx = self.move_to_index(board, move)
            prior = policy[idx].item() if idx is not None else 0.0
            children.append({'move': move, 'prior': prior, 'visit_count': 0, 'value_sum': 0.0, 'mean_value': 0.0, 'children': []})
            total += prior
        if total <= 0:
            total = float(len(children))
            for child in children:
                child['prior'] = 1.0 / total
        else:
            for child in children:
                child['prior'] /= total
        node['children'] = children

    def move_to_index(self, board, move):
        from_file = chess.square_file(move.from_square)
        from_rank = chess.square_rank(move.from_square)
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        dx = to_file - from_file
        dy = to_rank - from_rank
        piece = board.piece_at(move.from_square)
        if piece is None:
            return None
        base = move.from_square * 73
        if piece.piece_type == chess.PAWN and move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
            return base + 64 + (move.promotion - 2) * 3 + dx + 1
        step_x = 0 if dx == 0 else dx // abs(dx)
        step_y = 0 if dy == 0 else dy // abs(dy)
        if (dx == 0 or dy == 0 or abs(dx) == abs(dy)) and (step_x, step_y) in self.direction_lookup:
            return base + self.direction_lookup[(step_x, step_y)] * 7 + max(abs(dx), abs(dy)) - 1
        if (dx, dy) in self.knight_lookup:
            return base + 56 + self.knight_lookup[(dx, dy)]
        return None

    def main(self, fen):
        board = chess.Board(fen)
        return self.search(board)

if __name__ == '__main__':
    fen = chess.STARTING_FEN
    engine = MCTSEngine(simulations=800, c_puct=1.4)
    best_move = engine.main(fen)
    print(best_move.uci() if best_move else None)
