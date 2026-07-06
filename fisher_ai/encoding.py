"""Encode chess histories into neural-network input planes."""

import numpy as np

from fisher_ai import chess
from fisher_ai.game import HISTORY_LENGTH, MAX_GAME_PLIES

INPUT_PLANES = 119
ACTION_PLANES = 73
ACTION_SIZE = chess.ACTION_SIZE
move_to_action = chess.move_to_action


class StateEncodingWorkspace:
    """Reuse temporary arrays needed for batched state encoding."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.history_bitboards = np.empty(
            (self.capacity, HISTORY_LENGTH, 12),
            dtype=np.uint64,
        )
        self.history_repetitions = np.empty(
            (self.capacity, HISTORY_LENGTH),
            dtype=np.uint8,
        )
        self.history_valid = np.empty(
            (self.capacity, HISTORY_LENGTH),
            dtype=np.bool_,
        )
        self.current_colors = np.empty(self.capacity, dtype=np.bool_)
        self.plies = np.empty(self.capacity, dtype=np.uint16)
        self.castling_masks = np.empty(self.capacity, dtype=np.uint8)
        self.halfmove_clocks = np.empty(self.capacity, dtype=np.uint8)


def bitboards_to_planes(bitboards):
    """Expand packed bitboards into binary board planes."""
    bitboards = np.ascontiguousarray(bitboards, dtype=np.uint64)
    byte_shape = (*bitboards.shape, 8)
    byte_view = bitboards.view(np.uint8).reshape(byte_shape)
    return np.unpackbits(
        byte_view,
        axis=-1,
        bitorder="little",
    ).reshape(*bitboards.shape, 8, 8)


def encode_history_batch(
    history_bitboards,
    history_repetitions,
    history_valid,
    current_colors,
    plies,
    castling_masks,
    halfmove_clocks,
    output=None,
):
    """Encode aligned position histories into network input planes."""
    history_bitboards = np.asarray(history_bitboards, dtype=np.uint64)
    batch_size = len(history_bitboards)
    if output is None:
        output = np.zeros(
            (batch_size, INPUT_PLANES, 8, 8),
            dtype=np.float16,
        )
    else:
        output[:batch_size].fill(0)
        output = output[:batch_size]

    history_output = output[:, :112].reshape(
        batch_size,
        HISTORY_LENGTH,
        14,
        8,
        8,
    )
    piece_planes = bitboards_to_planes(history_bitboards)
    piece_planes *= np.asarray(history_valid)[..., None, None, None]

    current_colors = np.asarray(current_colors, dtype=np.bool_)
    white_rows = np.flatnonzero(current_colors)
    black_rows = np.flatnonzero(~current_colors)
    if len(white_rows):
        history_output[white_rows, :, :12] = piece_planes[white_rows]
    if len(black_rows):
        history_output[black_rows, :, :6] = piece_planes[
            black_rows,
            :,
            6:12,
            ::-1,
            ::-1,
        ]
        history_output[black_rows, :, 6:12] = piece_planes[
            black_rows,
            :,
            0:6,
            ::-1,
            ::-1,
        ]

    valid = np.asarray(history_valid, dtype=np.bool_)
    repetitions = np.asarray(history_repetitions)
    history_output[:, :, 12] = ((repetitions >= 2) & valid)[..., None, None]
    history_output[:, :, 13] = ((repetitions >= 3) & valid)[..., None, None]

    output[:, 112] = current_colors[:, None, None]
    output[:, 113] = (
        np.minimum(np.asarray(plies), MAX_GAME_PLIES) / MAX_GAME_PLIES
    )[:, None, None]

    castling_masks = np.asarray(castling_masks, dtype=np.uint8)
    own_kingside = np.where(
        current_colors,
        castling_masks & 1,
        castling_masks >> 2 & 1,
    )
    own_queenside = np.where(
        current_colors,
        castling_masks >> 1 & 1,
        castling_masks >> 3 & 1,
    )
    opponent_kingside = np.where(
        current_colors,
        castling_masks >> 2 & 1,
        castling_masks & 1,
    )
    opponent_queenside = np.where(
        current_colors,
        castling_masks >> 3 & 1,
        castling_masks >> 1 & 1,
    )
    output[:, 114] = own_kingside[:, None, None]
    output[:, 115] = own_queenside[:, None, None]
    output[:, 116] = opponent_kingside[:, None, None]
    output[:, 117] = opponent_queenside[:, None, None]
    output[:, 118] = (np.minimum(np.asarray(halfmove_clocks), 100) / 100.0)[
        :, None, None
    ]
    return output


def encode_states(states, output=None, workspace=None):
    """Encode a batch of live game states into one output tensor."""
    batch_size = len(states)
    if workspace is None:
        workspace = StateEncodingWorkspace(batch_size)
    elif batch_size > workspace.capacity:
        raise ValueError(
            f"encoding batch {batch_size} exceeds workspace "
            f"capacity {workspace.capacity}"
        )

    history_bitboards = workspace.history_bitboards[:batch_size]
    history_repetitions = workspace.history_repetitions[:batch_size]
    history_valid = workspace.history_valid[:batch_size]
    current_colors = workspace.current_colors[:batch_size]
    plies = workspace.plies[:batch_size]
    castling_masks = workspace.castling_masks[:batch_size]
    halfmove_clocks = workspace.halfmove_clocks[:batch_size]
    history_bitboards.fill(0)
    history_repetitions.fill(0)
    history_valid.fill(False)

    for index, state in enumerate(states):
        length = state.history_length
        start = HISTORY_LENGTH - length
        history_bitboards[index, start:] = state.history_bitboards[:length]
        history_repetitions[index, start:] = state.history_repetitions[:length]
        history_valid[index, start:] = True
        current_colors[index] = state.board.turn
        plies[index] = state.board.ply_count
        castling_masks[index] = state.board.castling_rights
        halfmove_clocks[index] = state.board.halfmove_clock

    return encode_history_batch(
        history_bitboards,
        history_repetitions,
        history_valid,
        current_colors,
        plies,
        castling_masks,
        halfmove_clocks,
        output=output,
    )


def encode_state(state, output=None):
    """Encode one live game state into network input planes."""
    if output is None:
        batch_output = None
    else:
        batch_output = output[None]
    return encode_states([state], output=batch_output)[0]


def encode_window_batch(
    snapshot_bitboards,
    snapshot_repetitions,
    game_starts,
    indices,
    current_colors,
    plies,
    castling_masks,
    halfmove_clocks,
    output=None,
):
    """Encode indexed positions directly from window arrays."""
    indices = np.asarray(indices, dtype=np.int64)
    starts = np.asarray(game_starts, dtype=np.int64)[indices]
    offsets = np.arange(-7, 1, dtype=np.int64)
    history_indices = indices[:, None] + offsets
    history_valid = history_indices >= starts[:, None]
    history_indices = np.maximum(history_indices, starts[:, None])

    return encode_history_batch(
        np.asarray(snapshot_bitboards)[history_indices],
        np.asarray(snapshot_repetitions)[history_indices],
        history_valid,
        np.asarray(current_colors)[indices],
        np.asarray(plies)[indices],
        np.asarray(castling_masks)[indices],
        np.asarray(halfmove_clocks)[indices],
        output=output,
    )
