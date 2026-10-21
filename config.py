"""
Configuration and constants for 8x8 Standard Chess RL Engine.
Defines board dimensions, piece representations, piece-square tables,
and hyperparameters for Reinforcement Learning and Alpha-Beta Search.
"""

BOARD_WIDTH = 8
BOARD_HEIGHT = 8
EMPTY_SQUARE = "--"

# Piece definitions: color + piece type
WHITE_PAWN = "wP"
WHITE_KNIGHT = "wN"
WHITE_BISHOP = "wB"
WHITE_ROOK = "wR"
WHITE_QUEEN = "wQ"
WHITE_KING = "wK"

BLACK_PAWN = "bP"
BLACK_KNIGHT = "bN"
BLACK_BISHOP = "bB"
BLACK_ROOK = "bR"
BLACK_QUEEN = "bQ"
BLACK_KING = "bK"

ALL_PIECES = [
    WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING,
    BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING
]

# Baseline standard material values (used for baseline comparison and initial prior)
PIECE_VALUES = {
    WHITE_PAWN: 100,
    BLACK_PAWN: -100,
    WHITE_KNIGHT: 320,
    BLACK_KNIGHT: -320,
    WHITE_BISHOP: 330,
    BLACK_BISHOP: -330,
    WHITE_ROOK: 500,
    BLACK_ROOK: -500,
    WHITE_QUEEN: 900,
    BLACK_QUEEN: -900,
    WHITE_KING: 0,
    BLACK_KING: 0,
}

# Piece-Square Tables (White perspective; flipped vertically for Black)
PAWN_PST = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

KNIGHT_PST = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]
]

BISHOP_PST = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
]

ROOK_PST = [
    [0, 0, 0, 5, 5, 0, 0, 0],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [0, 0, 0, 5, 5, 0, 0, 0]
]

QUEEN_PST = [
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20]
]

KING_PST_MIDDLE_GAME = [
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [20, 30, 10, 0, 0, 10, 30, 20]
]

KING_PST_LATE_GAME = [
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10, 0, 0, -10, -20, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -30, 0, 0, 0, 0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50]
]

# Unicode piece characters for terminal and logging display
UNICODE_PIECES = {
    WHITE_PAWN: "♙",
    WHITE_KNIGHT: "♘",
    WHITE_BISHOP: "♗",
    WHITE_ROOK: "♖",
    WHITE_QUEEN: "♕",
    WHITE_KING: "♔",
    BLACK_PAWN: "♟",
    BLACK_KNIGHT: "♞",
    BLACK_BISHOP: "♝",
    BLACK_ROOK: "♜",
    BLACK_QUEEN: "♛",
    BLACK_KING: "♚",
    EMPTY_SQUARE: "."
}

PIECE_SYMBOLS = UNICODE_PIECES

# Reinforcement Learning Default Hyperparameters
DEFAULT_GAMMA = 0.99
DEFAULT_LAMBDA = 0.70
DEFAULT_ALPHA = 0.02
DEFAULT_EPSILON_START = 0.20
DEFAULT_EPSILON_MIN = 0.02
DEFAULT_EPSILON_DECAY = 0.995
DEFAULT_EPISODES = 1000
DEFAULT_SEARCH_DEPTH = 4
DEFAULT_QUIESCENCE_DEPTH = 3
DEFAULT_BLITZ_TIME = 60.0
