from transformers import pipeline
import torch, time, chess
import re
import textwrap
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from collections import deque

class BatchPrefixAllowedTokens:
    def __init__(self, fns):
        self.fns = fns

    def __call__(self, batch_id, input_ids):
        return self.fns[batch_id](batch_id, input_ids)

class ChessTree:
    def __init__(self, root_key):
        self.root_key = root_key
        self.current_key = root_key
        self.nodes = {}
        self.ensure_node(root_key)

    def ensure_node(self, key):
        if key not in self.nodes:
            self.nodes[key] = {'key': key, 'pos': None, 'moves': []}
        return self.nodes[key]

    def set_node_data(self, key, parsed):
        node = self.ensure_node(key)
        node['pos'] = parsed['pos']

        existing_children = {}
        for m in node['moves']:
            existing_children[m['move']] = m.get('child')

        moves = []
        for c in parsed['candidates']:
            moves.append({
                'move': c['move'],
                'score': c['score'],
                'child': existing_children.get(c['move'])
            })

        node['moves'] = moves
        return node

    def link_child(self, parent_key, move, child_key):
        parent = self.ensure_node(parent_key)
        self.ensure_node(child_key)

        for m in parent['moves']:
            if m['move'] == move:
                m['child'] = child_key
                return

        parent['moves'].append({'move': move, 'score': None, 'child': child_key})

    def set_current(self, key):
        self.current_key = key

class ChessEngine:
    def __init__(self, player='white'):
        self.player = player
        self.game_board = chess.Board()
        self.board = chess.Board(self.game_board.fen())

        self.legal_moves = "\n".join([i.uci() for i in self.board.legal_moves])
        self.tree = ChessTree(self.game_board.fen())

        self.chess_rules = textwrap.dedent('''
        Opening:
        - fight for centre control, develop knights and bishops early, castle early, avoid repeated moves, open lines for your pieces, connect the rocks,
        - put pieces on active squares, fish development, if attacked, develop with tempo, use known openings, don't block bishops, avoid edge-knight moves

        tactical:
        - look for forcing moves, check first then capture, always check what's opponent threatening? count attackers and defenders, 
        - watch out for forks, pins, skewers and discovered checks, consider king safety, win material if you can do so safely, 
        - if unsure pick solid safe well known moves

        material:
        - pawn = 1, knight/bishop = 3, rook = 5, queen = 9, consider material value

        middle game:
        - improve your worst placed pieces, activate rooks on open files, target weakness in enemy position, create weakness to win
        - trade when it improves position, simplify safely, control the centre, coordinate pieces, maintain tempo, increase pressure before attacking

        pawns:
        - maintain strong pawn positions, don't create isolated pawns, avoid double stacked pawns, use pawn breaks to open lines, passed pawns are strong assets
        - bishops love open diagonals, protect the king with pawn cover

        piece rules:
        - knights belong near the centre, avoid knights on edges of the board, rooks belong on open files, don't overexpose queen, don't trap your own pieces

        end game:
        - push passed pawns, know the rule of the squares put rooks behind passed pawns, cut off enemy king, trade into winning pawn endgame

        important: These are general chess suggestions, and best practises not absolute rules. You can deviate, use this as good advice only.
        ''')

        self.format_prompt = textwrap.dedent('''
        Reasoning: low

        1. position score must be an integer between -5 and 5
          -5 means you are very likely losing in the CURRENT position.
          5 means you are very likely winning in the CURRENT position.
          0 means equal, not winning or losing in the CURRENT position.

        2. output EXACTLY 4 LINES and NOTHING ELSE (no explanations, no repeated legal moves list, no analysis)

        3. lines 2-4 must be a CSV table with 2 columns: move,score
          - move must be EXACTLY one of the moves in the legal moves list
          - score must be an integer 1 to 100 (how confident you are that move is best)

        TEMPLATE (copy this structure exactly):
        pos,0
        e2e4,87
        d2d4,81
        g1f3,62
        ''').strip()

        self.prompt = textwrap.dedent(f'''
        PLAYER: {self.player}

        FEN: {self.board.fen()}

        LEGAL MOVES (authoritative; do not question or analyze this list; one per line):
        {self.legal_moves}

        Output the 4 lines now.
        ''').strip()

        self.pipe = pipeline(
            'text-generation',
            model='microsoft/Phi-3.5-mini-instruct',
            torch_dtype=torch.bfloat16,
            device=0
        )
    
    def sync_search_board(self):
        self.board = chess.Board(self.game_board.fen())
    
    def expand_fens_batch(self, fens):
        items = []
        messages_batch = []
        prefix_fns = []

        for fen in fens:
            board = chess.Board(fen)
            player = 'white' if board.turn == chess.WHITE else 'black'

            legal_moves = "\n".join([i.uci() for i in board.legal_moves])

            prompt = textwrap.dedent(f'''
            PLAYER: {player}

            FEN: {board.fen()}

            LEGAL MOVES (authoritative; do not question or analyze this list; one per line):
            {legal_moves}

            Output the 4 lines now.
            ''').strip()

            moves = sorted(set([m.uci() for m in board.legal_moves]))
            move_pattern = "(?:" + "|".join([re.escape(m) for m in moves]) + ")"

            pos_pattern = r"(?:-5|-?[1-4]|0|[1-4]|5)"
            score_pattern = r"(?:100|[1-9]\d?)"

            line1 = r"pos," + pos_pattern
            line_move = move_pattern + r"," + score_pattern

            regex = line1 + r"\n" + line_move + r"\n" + line_move + r"\n" + line_move + r"\n?"
            parser = RegexParser(regex)
            prefix_fn = build_transformers_prefix_allowed_tokens_fn(self.pipe.tokenizer, parser)

            messages = [
                {'role' : 'system', 'content' : self.format_prompt},
                {'role' : 'system', 'content' : self.chess_rules},
                {'role' : 'system', 'content' : f'You are playing as {player}. pos is from {player} perspective: +5 means {player} winning, -5 means {player} losing.'},
                {'role': 'user', 'content': prompt},
            ]

            items.append({'fen': fen, 'board': board, 'player': player})
            messages_batch.append(messages)
            prefix_fns.append(prefix_fn)

        prefix_router = BatchPrefixAllowedTokens(prefix_fns)

        start = time.time()
        with torch.inference_mode():
            outputs = self.pipe(
                messages_batch,
                batch_size=len(messages_batch),
                max_new_tokens=256,
                prefix_allowed_tokens_fn=prefix_router
            )
        end = time.time() - start

        results = []
        for i in range(len(items)):
            item = items[i]
            board = item['board']
            player = item['player']
            fen = item['fen']

            gen = outputs[i]
            if isinstance(gen, list):
                gen = gen[0]

            gt = gen['generated_text']

            if isinstance(gt, list):
                output = gt[-1].get('content', '')
            else:
                output = gt

            output = output.rsplit('assistantfinal')[-1].strip()

            parsed = self.parse_output(output, board=board)
            if not parsed:
                results.append(None)
                continue

            pos = parsed['pos']
            pos_white = pos if player == 'white' else -pos
            parsed['pos'] = pos_white

            self.tree.set_node_data(fen, parsed)

            children = []
            for c in parsed['candidates']:
                move = c['move']
                board.push_uci(move)
                child_key = board.fen()
                board.pop()

                self.tree.link_child(fen, move, child_key)
                children.append(child_key)

            results.append({'fen': fen, 'pos_white': pos_white, 'children': children, 'time': end})

        return results
    
    def parse_output(self, output, board=None):
        if board is None:
            board = self.board

        moves = sorted(set([m.uci() for m in board.legal_moves]))
        move_pattern = "(?:" + "|".join([re.escape(m) for m in moves]) + ")"

        pos_pattern = r"(?:-5|-?[1-4]|0|[1-4]|5)"
        score_pattern = r"(?:100|[1-9]\d?)"

        output = output.strip()

        m = re.match(r"^pos,(" + pos_pattern + r")\n", output)
        if not m:
            return None

        pos = int(m.group(1))

        line_re = re.compile(r"^(" + move_pattern + r"),(" + score_pattern + r")$")

        candidates = []
        for line in output.splitlines()[1:]:
            mm = line_re.match(line.strip())
            if not mm:
                return None
            candidates.append({'move': mm.group(1), 'score': int(mm.group(2))})

        if len(candidates) != 3:
            return None

        return {'pos': pos, 'candidates': candidates}
    
    def set_fen(self, fen):
        self.board = chess.Board(fen)

    def expand_fen(self, fen):
        self.set_fen(fen)

        player = 'white' if self.board.turn == chess.WHITE else 'black'

        output, end = self.get_move_output(player=player)
        parsed = self.parse_output(output)
        if not parsed:
            return None

        # model's pos is from "player" perspective; convert to white perspective
        pos = parsed['pos']
        pos_white = pos if player == 'white' else -pos
        parsed['pos'] = pos_white

        parent_key = fen

        self.tree.set_node_data(parent_key, parsed)

        children = []
        for c in parsed['candidates']:
            move = c['move']

            self.board.push_uci(move)
            child_key = self.board.fen()
            self.board.pop()

            self.tree.link_child(parent_key, move, child_key)
            children.append(child_key)

        return {'fen': parent_key, 'pos_white': pos_white, 'children': children, 'time': end}

    def search_bfs(self, max_plies=3, batch_size=8, max_time=None, play_best=False, agg='pessimistic', depth_bonus=0.10, depth_penalty=0.10, use_path_min=True):
        start_time = time.time()

        root_fen = self.game_board.fen()
        self.set_fen(root_fen)
        self.tree.ensure_node(root_fen)
        self.tree.set_current(root_fen)

        q = deque()
        q.append((root_fen, 0))

        while q:
            if max_time is not None and (time.time() - start_time) >= max_time:
                break

            batch = []
            while q and len(batch) < batch_size:
                batch.append(q.popleft())

            to_expand_fens = []
            to_expand_plys = []

            for fen, ply in batch:
                if ply >= max_plies:
                    continue

                node = self.tree.ensure_node(fen)

                if node['pos'] is not None and node['moves']:
                    if node['pos'] < 0:
                        continue

                    for m in node['moves']:
                        child = m.get('child')
                        if not child:
                            continue
                        child_node = self.tree.ensure_node(child)
                        if child_node['pos'] is None:
                            q.append((child, ply + 1))
                    continue

                to_expand_fens.append(fen)
                to_expand_plys.append(ply)

            if not to_expand_fens:
                continue

            if max_time is not None and (time.time() - start_time) >= max_time:
                break

            infos = self.expand_fens_batch(to_expand_fens)

            if max_time is not None and (time.time() - start_time) >= max_time:
                break

            for info, ply in zip(infos, to_expand_plys):
                if not info:
                    continue

                if info['pos_white'] < 0:
                    continue

                for child_fen in info['children']:
                    child_node = self.tree.ensure_node(child_fen)
                    if child_node['pos'] is None:
                        q.append((child_fen, ply + 1))

        self.set_fen(root_fen)
        self.tree.set_current(root_fen)

        best = None
        if play_best:
            best = self.best_tree_move(
                max_depth=max_plies,
                agg=agg,
                depth_bonus=depth_bonus,
                depth_penalty=depth_penalty,
                use_path_min=use_path_min
            )

            if best and best.get('move'):
                self.game_board.push_uci(best['move'])
                self.sync_search_board()
                self.tree.set_current(self.game_board.fen())

        print(f'Finished BFS search in {time.time() - start_time} seconds')
        return best
    
    def best_tree_move(self, max_depth=4, agg='pessimistic', depth_bonus=0.10, depth_penalty=0.10, use_path_min=True, unknown_penalty=0.25):
        root_fen = self.board.fen()
        root_node = self.tree.ensure_node(root_fen)

        if not root_node.get('moves'):
            return None

        move_scores = {}
        move_paths = {}

        root_pos = root_node.get('pos')
        if root_pos is None:
            root_pos = -unknown_penalty

        stack = []

        for m in root_node['moves']:
            move = m.get('move')
            child = m.get('child')
            if not move or not child:
                continue

            stack.append((child, 1, move, [move], root_pos))

        while stack:
            fen, depth, first_move, path_moves, path_min = stack.pop()

            node = self.tree.ensure_node(fen)
            pos = node.get('pos')
            if pos is None:
                pos = -unknown_penalty

            if use_path_min:
                if pos < path_min:
                    path_min_now = pos
                else:
                    path_min_now = path_min
                base = path_min_now
            else:
                path_min_now = path_min
                base = pos

            children = []
            for mm in node.get('moves', []):
                child_fen = mm.get('child')
                if child_fen:
                    children.append((mm.get('move'), child_fen))

            is_cutoff = depth >= max_depth
            is_leaf = len(children) == 0

            if is_cutoff or is_leaf:
                if base >= 0:
                    leaf_score = base + (depth_bonus * depth)
                else:
                    leaf_score = base - (depth_penalty * depth)

                if first_move not in move_scores:
                    move_scores[first_move] = [leaf_score]
                    move_paths[first_move] = [path_moves]
                else:
                    move_scores[first_move].append(leaf_score)
                    move_paths[first_move].append(path_moves)

                continue

            for move_uci, child_fen in children:
                if not move_uci:
                    continue
                stack.append((child_fen, depth + 1, first_move, path_moves + [move_uci], path_min_now))

        if not move_scores:
            return None

        best_move = None
        best_value = None
        best_line = None

        for move, scores in move_scores.items():
            if agg in ['pessimistic', 'min']:
                value = min(scores)
                idx = scores.index(value)
            elif agg in ['optimistic', 'max']:
                value = max(scores)
                idx = scores.index(value)
            else:
                value = sum(scores) / len(scores)
                idx = scores.index(max(scores))

            if best_value is None or value > best_value:
                best_value = value
                best_move = move
                best_line = move_paths[move][idx]

        return {
            'move': best_move,
            'value': best_value,
            'line': best_line,
            'per_move': {m: (min(s) if agg in ['pessimistic', 'min'] else (max(s) if agg in ['optimistic', 'max'] else (sum(s) / len(s)))) for m, s in move_scores.items()},
            'counts': {m: len(s) for m, s in move_scores.items()},
        }
    
    def play_top_move(self, player=None):
        parent_key = self.board.fen()

        output, end = self.get_move_output(player=player)
        parsed = self.parse_output(output)
        if not parsed:
            return None

        self.tree.set_node_data(parent_key, parsed)

        top_move = parsed['candidates'][0]['move']
        self.board.push_uci(top_move)

        child_key = self.board.fen()
        self.tree.link_child(parent_key, top_move, child_key)
        self.tree.set_current(child_key)

        print(f'{player} played: {top_move}, pos: {parsed["pos"]} in {end} seconds')
        print(self.board)

        return {'move': top_move, 'pos': parsed['pos'], 'parsed': parsed, 'time': end, 'from': parent_key, 'to': child_key}

    def get_move_output(self, player=None):
        if player is None:
            player = self.player

        if player in ['w', 'W', 'white', 'White', 'WHITE']:
            player = 'white'
        else:
            player = 'black'

        self.legal_moves = "\n".join([i.uci() for i in self.board.legal_moves])

        self.prompt = textwrap.dedent(f'''
        PLAYER: {player}

        FEN: {self.board.fen()}

        LEGAL MOVES (authoritative; do not question or analyze this list; one per line):
        {self.legal_moves}

        Output the 4 lines now.
        ''').strip()

        moves = sorted(set([m.uci() for m in self.board.legal_moves]))
        move_pattern = "(?:" + "|".join([re.escape(m) for m in moves]) + ")"

        pos_pattern = r"(?:-5|-?[1-4]|0|[1-4]|5)"
        score_pattern = r"(?:100|[1-9]\d?)"

        line1 = r"pos," + pos_pattern
        line_move = move_pattern + r"," + score_pattern

        regex = line1 + r"\n" + line_move + r"\n" + line_move + r"\n" + line_move + r"\n?"
        parser = RegexParser(regex)
        prefix_function = build_transformers_prefix_allowed_tokens_fn(self.pipe.tokenizer, parser)

        messages = [
            {'role' : 'system', 'content' : self.format_prompt},
            {'role' : 'system', 'content' : self.chess_rules},
            {'role' : 'system', 'content' : f'You are playing as {player}. pos is from {player} perspective: +5 means {player} winning, -5 means {player} losing.'},
            {'role': 'user', 'content': self.prompt},
        ]

        start = time.time()
        outputs = self.pipe(messages, max_new_tokens=256, prefix_allowed_tokens_fn=prefix_function)
        output = outputs[0]['generated_text'][-1]['content']
        end = time.time() - start

        return output.rsplit('assistantfinal')[-1].strip(), end
