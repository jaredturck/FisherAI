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
        encoded_states = [encode_state(state) for state in states]
        return self.evaluate_encoded(encoded_states, legal_actions=legal_actions)

    def evaluate_encoded(self, encoded_states, legal_actions=None):
        policy_parts = []
        value_parts = []

        for start in range(0, len(encoded_states), self.inference_batch_size):
            end = min(start + self.inference_batch_size, len(encoded_states))
            array = np.stack(encoded_states[start:end]).astype(np.float32, copy=False)
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
                    gathered[index, : len(actions)]
                    for index, actions in enumerate(chunk_actions)
                )

            value_parts.append(value.float().cpu().numpy())

        policies = np.concatenate(policy_parts) if legal_actions is None else policy_parts
        return policies, np.concatenate(value_parts)


class MCTSNode:
    def __init__(self, state=None, parent=None, move=None, prior=1.0, parent_index=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.parent_index = parent_index
        self.prior = prior
        self.base_prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_visit_count = 0
        self.children = {}
        self.child_actions = np.asarray([], dtype=np.int64)
        self.child_nodes = []
        self.child_priors = np.asarray([], dtype=np.float32)
        self.child_base_priors = np.asarray([], dtype=np.float32)
        self.child_visits = np.asarray([], dtype=np.int32)
        self.child_value_sums = np.asarray([], dtype=np.float32)
        self.child_virtual_visits = np.asarray([], dtype=np.int16)
        self.child_pending = np.asarray([], dtype=np.bool_)
        self.expanded = False
        self.pending = False
        self.encoded_state = None

    @property
    def mean_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ensure_state(self):
        if self.state is not None:
            return self.state

        moves = []
        node = self
        while node.state is None:
            moves.append(node.move)
            node = node.parent

        state = node.state.copy()
        state.push_moves(reversed(moves))
        self.state = state
        return self.state

    def set_pending(self, pending):
        self.pending = pending
        if self.parent is not None:
            self.parent.child_pending[self.parent_index] = pending

    def detach(self):
        self.parent = None
        self.move = None
        self.parent_index = None
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
            root for root in roots if not root.expanded and not root.ensure_state().is_terminal()
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
            if not root.ensure_state().is_terminal()
        ):
            pending_leaves = []
            made_progress = False

            for root, target in zip(roots, target_visits, strict=True):
                if root.ensure_state().is_terminal():
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
                    leaf_state = leaf.ensure_state()

                    if leaf_state.is_terminal():
                        self.release(leaf)
                        self.backup(leaf, leaf_state.terminal_value())
                    else:
                        leaf.set_pending(True)
                        pending_leaves.append(leaf)

            if pending_leaves:
                values = self.evaluate_and_expand(pending_leaves)
                for leaf, value in zip(pending_leaves, values, strict=True):
                    leaf.set_pending(False)
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
        while node.expanded and len(node.child_nodes):
            node = self.select_child(node)
            if node is None:
                return None
        return node

    def select_child(self, node):
        parent_visits = max(1, node.visit_count + node.virtual_visit_count)
        denominators = 1 + node.child_visits + node.child_virtual_visits
        mean_values = np.divide(
            node.child_value_sums,
            node.child_visits,
            out=np.zeros_like(node.child_value_sums),
            where=node.child_visits > 0,
        )
        exploitation = -mean_values
        exploration = (
            self.config.c_puct
            * node.child_priors
            * np.sqrt(parent_visits)
            / denominators
        )
        scores = (
            exploitation
            + exploration
            - self.config.virtual_loss * node.child_virtual_visits
        )
        unavailable = node.child_pending & (node.child_visits == 0)
        scores[unavailable] = -np.inf

        index = int(np.argmax(scores))
        if not np.isfinite(scores[index]):
            return None
        return node.child_nodes[index]

    def evaluate_and_expand(self, nodes):
        states = [node.ensure_state() for node in nodes]
        for node, state in zip(nodes, states, strict=True):
            if node.encoded_state is None:
                node.encoded_state = encode_state(state)

        mappings = [legal_action_map(state) for state in states]
        action_lists = [np.asarray(sorted(mapping), dtype=np.int64) for mapping in mappings]
        encoded_states = [node.encoded_state for node in nodes]
        if hasattr(self.evaluator, "evaluate_encoded"):
            policy_output, values = self.evaluator.evaluate_encoded(
                encoded_states,
                legal_actions=action_lists,
            )
        else:
            policy_output, values = self.evaluator.evaluate(
                states,
                legal_actions=action_lists,
            )

        for index, node in enumerate(nodes):
            actions = action_lists[index]
            logits = policy_output[index]
            if isinstance(policy_output, np.ndarray):
                logits = logits[actions]

            logits = logits - logits.max()
            priors = np.exp(logits)
            priors /= priors.sum()

            mapping = mappings[index]
            child_nodes = []
            for child_index, (action, prior) in enumerate(
                zip(actions, priors, strict=True)
            ):
                move = mapping[int(action)]
                child = MCTSNode(
                    parent=node,
                    move=move,
                    prior=float(prior),
                    parent_index=child_index,
                )
                node.children[int(action)] = child
                child_nodes.append(child)

            node.child_actions = actions.astype(np.int64, copy=False)
            node.child_nodes = child_nodes
            node.child_priors = priors.astype(np.float32, copy=True)
            node.child_base_priors = priors.astype(np.float32, copy=True)
            child_count = len(actions)
            node.child_visits = np.zeros(child_count, dtype=np.int32)
            node.child_value_sums = np.zeros(child_count, dtype=np.float32)
            node.child_virtual_visits = np.zeros(child_count, dtype=np.int16)
            node.child_pending = np.zeros(child_count, dtype=np.bool_)
            node.expanded = True

        return values

    def set_root_priors(self, root, add_noise):
        root.child_priors[:] = root.child_base_priors
        for child, prior in zip(root.child_nodes, root.child_priors, strict=True):
            child.prior = float(prior)
            child.base_prior = float(prior)

        if not add_noise or not len(root.child_nodes):
            return

        noise = self.rng.dirichlet(
            np.full(len(root.child_nodes), self.config.dirichlet_alpha)
        )
        fraction = self.config.dirichlet_fraction
        root.child_priors[:] = (
            (1.0 - fraction) * root.child_base_priors + fraction * noise
        )
        for child, prior in zip(root.child_nodes, root.child_priors, strict=True):
            child.prior = float(prior)

    @staticmethod
    def reserve(leaf):
        node = leaf
        while node is not None:
            node.virtual_visit_count += 1
            if node.parent is not None:
                node.parent.child_virtual_visits[node.parent_index] += 1
            node = node.parent

    @staticmethod
    def release(leaf):
        node = leaf
        while node is not None:
            node.virtual_visit_count -= 1
            if node.parent is not None:
                node.parent.child_virtual_visits[node.parent_index] -= 1
            node = node.parent

    @staticmethod
    def backup(leaf, value):
        node = leaf
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            if node.parent is not None:
                node.parent.child_visits[node.parent_index] += 1
                node.parent.child_value_sums[node.parent_index] += value
            value = -value
            node = node.parent

    @staticmethod
    def visit_counts(root):
        return root.child_actions.copy(), root.child_visits.astype(np.float32, copy=True)

    def choose_action(self, root, temperature=1.0, greedy=False):
        actions, counts = self.visit_counts(root)
        assert counts.sum() > 0

        if greedy or temperature <= 0:
            return int(actions[np.argmax(counts)])

        weights = counts ** (1.0 / temperature)
        probabilities = weights / weights.sum()
        return int(self.rng.choice(actions, p=probabilities))
