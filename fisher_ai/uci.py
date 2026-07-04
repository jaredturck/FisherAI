import sys

import chess

from fisher_ai.game import GameState
from fisher_ai.mcts import MCTS, MCTSNode


class UCIEngine:
    def __init__(self, evaluator, search_config, seed=7):
        self.evaluator = evaluator
        self.search_config = search_config
        self.search = MCTS(evaluator, search_config, seed=seed)
        self.state = GameState(max_game_plies=search_config.max_game_plies)

    def set_position(self, command):
        parts = command.split()
        move_index = parts.index("moves") if "moves" in parts else len(parts)

        if len(parts) > 1 and parts[1] == "startpos":
            self.state = GameState(max_game_plies=self.search_config.max_game_plies)
        elif len(parts) > 1 and parts[1] == "fen":
            fen = " ".join(parts[2:move_index])
            self.state = GameState.from_fen(fen, max_game_plies=self.search_config.max_game_plies)

        if move_index < len(parts):
            for uci in parts[move_index + 1 :]:
                self.state.push(chess.Move.from_uci(uci))

    def best_move(self, simulations=None):
        if self.state.is_terminal():
            return None

        simulations = simulations or self.search_config.evaluation_simulations
        root = MCTSNode(state=self.state)
        root = self.search.run(
            [self.state],
            roots=[root],
            add_noise=False,
            simulations=simulations,
        )[0]
        action = self.search.choose_action(root, greedy=True)
        return root.children[action].move

    def run(self):
        for raw_line in sys.stdin:
            line = raw_line.strip()

            if line == "uci":
                print("id name Fisher AI AlphaZero")
                print("id author Jared Turck and OpenAI")
                print("uciok", flush=True)
            elif line == "isready":
                print("readyok", flush=True)
            elif line == "ucinewgame":
                self.state = GameState(max_game_plies=self.search_config.max_game_plies)
            elif line.startswith("position "):
                self.set_position(line)
            elif line.startswith("go"):
                parts = line.split()
                simulations = None
                if "nodes" in parts:
                    simulations = int(parts[parts.index("nodes") + 1])
                move = self.best_move(simulations=simulations)
                best_move = move.uci() if move is not None else "0000"
                print(f"bestmove {best_move}", flush=True)
            elif line == "quit":
                return
