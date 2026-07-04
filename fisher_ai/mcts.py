import math

import numpy as np
import torch

from fisher_ai.encoding import encode_state, legal_action_map


class TorchEvaluator:
    def __init__(self, model, device="cpu", inference_batch_size=512, channels_last=True):
        self.model = model
        self.device = torch.device(device)
        self.inference_batch_size = inference_batch_size
        self.channels_last = channels_last and self.device.type == "cuda"
        self.model.to(self.device)
        if self.channels_last:
            self.model.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = True
        self.model.eval()

    def evaluate(self, states, legal_actions=None):
        policy_parts = []
        value_parts = []

        for start in range(0, len(states), self.inference_batch_size):
            end = min(start + self.inference_batch_size, len(states))
            batch_states = states[start:end]
            array = np.stack([encode_state(state) for state in batch_states])
            tensor = torch.from_numpy(array)

            if self.device.type == "cuda":
                tensor = tensor.pin_memory()
            tensor = tensor.to(self.device, non_blocking=True)
            if self.channels_last:
                tensor = tensor.contiguous(memory_format=torch.channels_last)

            with torch.inference_mode():
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.device.type == "cuda",
                ):
                    policy, value = self.model(tensor)

            if legal_actions is None:
                policy_parts.append(policy.float().cpu().numpy())
            else:
                chunk_actions = legal_actions[start:end]
                max_actions = max(len(actions) for actions in chunk_actions)
                padded_actions = np.zeros((len(chunk_actions), max_actions), dtype=np.int64)

                for index, actions in enumerate(chunk_actions):
                    padded_actions[index, : len(actions)] = actions

                action_tensor = torch.from_numpy(padded_actions)
                if self.device.type == "cuda":
                    action_tensor = action_tensor.pin_memory()
                action_tensor = action_tensor.to(self.device, non_blocking=True)
                gathered = policy.gather(1, action_tensor).float().cpu().numpy()
                policy_parts.extend(
                    gathered[index, : len(actions)] for index, actions in enumerate(chunk_actions)
                )

            value_parts.append(value.float().cpu().numpy())

        policies = np.concatenate(policy_parts) if legal_actions is None else policy_parts
        return policies, np.concatenate(value_parts)


class MCTSNode:
    def __init__(self, state=None, parent=None, move=None, prior=1.0):
        self.state = state
        self.parent = parent
        self.move = move
        self.prior = prior
        self.base_prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_visit_count = 0
        self.children = {}
        self.child_actions = ()
        self.expanded = False
        self.pending = False

    @property
    def mean_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ensure_state(self):
        if self.state is None:
            self.state = self.parent.state.child(self.move)
        return self.state

    def detach(self):
        self.parent = None
        self.move = None
        return self


class MCTS:
    def __init__(self, evaluator, config, seed=7):
        self.evaluator = evaluator
        self.config = config
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        states,
        roots=None,
        add_noise=False,
        simulations=None,
        parallel_searches=None,
    ):
        if roots is None:
            roots = [MCTSNode(state=state) for state in states]

        simulation_counts = self.normalize_simulations(simulations, len(roots))
        parallel_searches = parallel_searches or self.config.parallel_searches
        target_visits = [
            root.visit_count + count
            for root, count in zip(roots, simulation_counts, strict=True)
        ]

        unexpanded_roots = [
            root for root in roots if not root.expanded and not root.state.is_terminal()
        ]
        if unexpanded_roots:
            self.evaluate_and_expand(unexpanded_roots)

        noise_flags = self.normalize_flags(add_noise, len(roots))
        for root, noise_flag in zip(roots, noise_flags, strict=True):
            if root.expanded:
                self.set_root_priors(root, noise_flag)

        while any(
            root.visit_count < target
            for root, target in zip(roots, target_visits, strict=True)
            if not root.state.is_terminal()
        ):
            pending_leaves = []
            made_progress = False

            for root, target in zip(roots, target_visits, strict=True):
                if root.state.is_terminal():
                    continue

                reservations = 0
                while (
                    root.visit_count + root.virtual_visit_count < target
                    and reservations < parallel_searches
                ):
                    leaf = self.select_leaf(root)
                    if leaf is None:
                        break

                    self.reserve(leaf)
                    made_progress = True
                    reservations += 1

                    if leaf.ensure_state().is_terminal():
                        self.release(leaf)
                        self.backup(leaf, leaf.state.terminal_value())
                    else:
                        leaf.pending = True
                        pending_leaves.append(leaf)

            if pending_leaves:
                values = self.evaluate_and_expand(pending_leaves)
                for leaf, value in zip(pending_leaves, values, strict=True):
                    leaf.pending = False
                    self.release(leaf)
                    self.backup(leaf, float(value))

            if not made_progress:
                break

        return roots

    def normalize_simulations(self, simulations, count):
        if simulations is None:
            return [self.config.simulations] * count
        if np.isscalar(simulations):
            return [int(simulations)] * count
        assert len(simulations) == count
        return [int(value) for value in simulations]

    @staticmethod
    def normalize_flags(flags, count):
        if isinstance(flags, (bool, np.bool_)):
            return [bool(flags)] * count
        assert len(flags) == count
        return [bool(value) for value in flags]

    def select_leaf(self, root):
        node = root
        while node.expanded and node.children:
            node = self.select_child(node)
            if node is None:
                return None
            node.ensure_state()
            if node.state.is_terminal():
                break
        return node

    def select_child(self, node):
        parent_visits = max(1, node.visit_count + node.virtual_visit_count)
        best_score = -float("inf")
        best_child = None

        for action in node.child_actions:
            child = node.children[action]
            if child.pending and child.visit_count == 0:
                continue

            exploitation = -child.mean_value
            exploration = (
                self.config.c_puct
                * child.prior
                * math.sqrt(parent_visits)
                / (1 + child.visit_count + child.virtual_visit_count)
            )
            score = (
                exploitation
                + exploration
                - self.config.virtual_loss * child.virtual_visit_count
            )

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def evaluate_and_expand(self, nodes):
        states = [node.ensure_state() for node in nodes]
        mappings = [legal_action_map(node.state) for node in nodes]
        action_lists = [np.asarray(sorted(mapping), dtype=np.int64) for mapping in mappings]
        policy_output, values = self.evaluator.evaluate(states, legal_actions=action_lists)

        for index, node in enumerate(nodes):
            actions = action_lists[index]
            logits = policy_output[index]
            if isinstance(policy_output, np.ndarray):
                logits = logits[actions]

            logits = logits - logits.max()
            priors = np.exp(logits)
            priors /= priors.sum()

            mapping = mappings[index]
            for action, prior in zip(actions, priors, strict=True):
                move = mapping[int(action)]
                node.children[int(action)] = MCTSNode(
                    parent=node,
                    move=move,
                    prior=float(prior),
                )

            node.child_actions = tuple(int(action) for action in actions)
            node.expanded = True

        return values

    def set_root_priors(self, root, add_noise):
        children = [root.children[action] for action in root.child_actions]
        for child in children:
            child.prior = child.base_prior

        if not add_noise or not children:
            return

        noise = self.rng.dirichlet([self.config.dirichlet_alpha for _ in children])
        fraction = self.config.dirichlet_fraction

        for child, noise_value in zip(children, noise, strict=True):
            child.prior = (1.0 - fraction) * child.base_prior + fraction * float(noise_value)

    @staticmethod
    def reserve(leaf):
        node = leaf
        while node is not None:
            node.virtual_visit_count += 1
            node = node.parent

    @staticmethod
    def release(leaf):
        node = leaf
        while node is not None:
            node.virtual_visit_count -= 1
            node = node.parent

    @staticmethod
    def backup(leaf, value):
        node = leaf
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent

    @staticmethod
    def visit_counts(root):
        actions = np.asarray(root.child_actions, dtype=np.int64)
        counts = np.asarray(
            [root.children[int(action)].visit_count for action in actions],
            dtype=np.float32,
        )
        return actions, counts

    def choose_action(self, root, temperature=1.0, greedy=False):
        actions, counts = self.visit_counts(root)
        assert counts.sum() > 0

        if greedy or temperature <= 0:
            return int(actions[np.argmax(counts)])

        weights = counts ** (1.0 / temperature)
        probabilities = weights / weights.sum()
        return int(self.rng.choice(actions, p=probabilities))
