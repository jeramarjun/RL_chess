"""
Strategic Feature Extraction Module for Reinforcement Learning Chess Evaluation.
Extracts normalized feature vectors directly aligned with the components of AIPlayer's evaluation function.
"""

import math
from config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    EMPTY_SQUARE,
    PIECE_VALUES,
    PAWN_PST,
    KNIGHT_PST,
    BISHOP_PST,
    ROOK_PST,
    KING_PST_LATE_GAME,
)

FEATURE_NAMES = [
    "material",
    "piece_square_tables",
    "minor_activity",
    "rook_activity",
    "pawn_structure",
    "mobility",
    "king_safety",
    "bias",
]

FEATURE_DIM = len(FEATURE_NAMES)


def clip(x, low=-1.0, high=1.0):
    """Clip a float within bounds [low, high]."""
    return max(low, min(high, float(x)))


def extract_features(engine):
    """
    Extract normalized feature dictionary from White's perspective.
    Corresponds directly to the strategic components evaluated in AIPlayer.
    """
    board = engine.board
    material = 0.0
    pst = 0.0
    white_minor_activity = 0.0
    black_minor_activity = 0.0
    white_rook_activity = 0.0
    black_rook_activity = 0.0
    king_safety = 0.0

    white_pawns = []
    black_pawns = []

    nonking_material = 0

    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            piece = board[r][c]
            if piece == EMPTY_SQUARE:
                continue

            side = piece[0]
            kind = piece[1]
            sign = 1.0 if side == "w" else -1.0
            val = abs(PIECE_VALUES.get(piece, 0))

            if kind != "K":
                nonking_material += val
            material += sign * val

            # Piece-Square Table
            table_row = r if side == "w" else BOARD_HEIGHT - 1 - r
            table_val = 0
            if kind == "P":
                table_val = PAWN_PST[table_row][c]
            elif kind == "N":
                table_val = KNIGHT_PST[table_row][c]
            elif kind == "B":
                table_val = BISHOP_PST[table_row][c]
            elif kind == "R":
                table_val = ROOK_PST[table_row][c]
            elif kind == "K" and nonking_material <= 260:
                table_val = KING_PST_LATE_GAME[table_row][c]

            pst += sign * table_val

            # Centrality for minors
            center = 3.5 - (abs(c - 3.5) + abs(r - 3.5)) * 0.5
            if kind == "N":
                if side == "w":
                    white_minor_activity += max(0.0, center)
                else:
                    black_minor_activity += max(0.0, center)
            elif kind == "B":
                if side == "w":
                    white_minor_activity += max(0.0, center) * 0.8
                else:
                    black_minor_activity += max(0.0, center) * 0.8
            elif kind == "R":
                if side == "w":
                    white_rook_activity += max(0, 7 - r) * 0.7
                else:
                    black_rook_activity += max(0, r) * 0.7
            elif kind == "P":
                if side == "w":
                    white_pawns.append((r, c))
                else:
                    black_pawns.append((r, c))
            elif kind == "K":
                if side == "w" and r < 5:
                    king_safety -= 12.0
                elif side == "b" and r > 2:
                    king_safety += 12.0

    # Pawn structure
    pawn_structure = 0.0
    for r, c in white_pawns:
        if c in (3, 4):
            pawn_structure += 7.0
        if r <= 4:
            pawn_structure += (4 - r) * 1.2

    for r, c in black_pawns:
        if c in (3, 4):
            pawn_structure -= 7.0
        if r >= 3:
            pawn_structure -= (r - 3) * 1.2

    white_pawn_cols = {c for _, c in white_pawns}
    black_pawn_cols = {c for _, c in black_pawns}
    for c in white_pawn_cols:
        if c - 1 in white_pawn_cols or c + 1 in white_pawn_cols:
            pawn_structure += 2.5
    for c in black_pawn_cols:
        if c - 1 in black_pawn_cols or c + 1 in black_pawn_cols:
            pawn_structure -= 2.5

    # Check status
    try:
        if engine.is_in_check():
            king_safety += -90.0 if engine.white_to_move else 90.0
    except Exception:
        pass

    # Mobility
    try:
        raw_mob = len(engine._get_all_possible_moves())
    except Exception:
        raw_mob = 0
    mobility = min(raw_mob, 30) * (0.45 if engine.white_to_move else -0.45)

    minor_activity = white_minor_activity - black_minor_activity
    rook_activity = white_rook_activity - black_rook_activity

    return {
        "material": clip(material / 4000.0),
        "piece_square_tables": clip(pst / 1000.0),
        "minor_activity": clip(minor_activity / 20.0),
        "rook_activity": clip(rook_activity / 40.0),
        "pawn_structure": clip(pawn_structure / 50.0),
        "mobility": clip(mobility / 15.0),
        "king_safety": clip(king_safety / 100.0),
        "bias": 1.0,
    }


def features_to_vector(features_dict):
    """Convert feature dict to a list of floats ordered by FEATURE_NAMES."""
    return [features_dict[name] for name in FEATURE_NAMES]
