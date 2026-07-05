from fisher_ai import chess
from fisher_ai.game import GameState
from fisher_ai.mcts import MCTS, MCTSNode


def play_game(white_evaluator, black_evaluator, search_config, seed=7):
    state = GameState(max_game_plies=search_config.max_game_plies)
    white_search = MCTS(white_evaluator, search_config, seed=seed)
    black_search = MCTS(black_evaluator, search_config, seed=seed + 1)
    moves = []

    while not state.is_terminal():
        search = white_search if state.board.turn == chess.WHITE else black_search
        root = MCTSNode(state=state)
        root = search.run(
            [state],
            roots=[root],
            add_noise=False,
            simulations=search_config.evaluation_simulations,
        )[0]
        action = search.choose_action(root, greedy=True)
        child = root.children[action]
        child.ensure_state()
        moves.append(child.move.uci())
        state = child.state

    return state.final_result(), moves


def play_match(evaluator_a, evaluator_b, search_config, games=20, seed=7):
    results = {"a_wins": 0, "b_wins": 0, "draws": 0, "games": []}

    for game_index in range(games):
        a_is_white = game_index % 2 == 0
        white = evaluator_a if a_is_white else evaluator_b
        black = evaluator_b if a_is_white else evaluator_a
        result, moves = play_game(white, black, search_config, seed=seed + game_index * 2)

        if result == 0:
            results["draws"] += 1
            winner = "draw"
        elif (result == 1 and a_is_white) or (result == -1 and not a_is_white):
            results["a_wins"] += 1
            winner = "a"
        else:
            results["b_wins"] += 1
            winner = "b"

        results["games"].append(
            {
                "game": game_index + 1,
                "a_is_white": a_is_white,
                "winner": winner,
                "moves": moves,
            }
        )

    return results
