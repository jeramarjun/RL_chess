"""
Standard 8x8 Chess rules engine wrapper using python-chess.
Maintains full compatibility with course assignment tournament interface
and provides the exact GameEngine and Move APIs expected by AIPlayer.
"""

import chess
from config import (
    BOARD_WIDTH,
    BOARD_HEIGHT,
    EMPTY_SQUARE,
    WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING,
    BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING,
)

FROM_CHESS = {
    (True, chess.PAWN): WHITE_PAWN,
    (True, chess.KNIGHT): WHITE_KNIGHT,
    (True, chess.BISHOP): WHITE_BISHOP,
    (True, chess.ROOK): WHITE_ROOK,
    (True, chess.QUEEN): WHITE_QUEEN,
    (True, chess.KING): WHITE_KING,
    (False, chess.PAWN): BLACK_PAWN,
    (False, chess.KNIGHT): BLACK_KNIGHT,
    (False, chess.BISHOP): BLACK_BISHOP,
    (False, chess.ROOK): BLACK_ROOK,
    (False, chess.QUEEN): BLACK_QUEEN,
    (False, chess.KING): BLACK_KING,
}

PROMOTION_TYPES = {
    "Q": chess.QUEEN,
    "R": chess.ROOK,
    "B": chess.BISHOP,
    "N": chess.KNIGHT,
}

INVERSE_PROMOTION_TYPES = {
    chess.QUEEN: "Q",
    chess.ROOK: "R",
    chess.BISHOP: "B",
    chess.KNIGHT: "N",
}


def sq_to_rc(sq):
    """Convert python-chess Square (0..63) to (row, col) where row 0 is rank 8."""
    return 7 - chess.square_rank(sq), chess.square_file(sq)


def rc_to_sq(r, c):
    """Convert (row, col) where row 0 is rank 8 to python-chess Square."""
    return chess.square(c, 7 - r)


class _PseudoCount:
    def __init__(self, count):
        self._count = count

    def __len__(self):
        return self._count


class Move:
    """Represents a chess move compatible with the course engine interface."""

    def __init__(self, start_square, end_square, board=None, promotion=None, _chess_move=None):
        self.start_row, self.start_col = start_square
        self.end_row, self.end_col = end_square
        self.promotion = promotion.upper() if promotion else None
        self._chess_move = _chess_move
        self.piece_moved = (
            board[self.start_row][self.start_col] if board is not None else EMPTY_SQUARE
        )
        self.piece_captured = (
            board[self.end_row][self.end_col] if board is not None else EMPTY_SQUARE
        )

    @classmethod
    def from_chess(cls, m, engine):
        sr, sc = sq_to_rc(m.from_square)
        er, ec = sq_to_rc(m.to_square)
        promo = chess.piece_symbol(m.promotion).upper() if m.promotion else None
        obj = cls((sr, sc), (er, ec), engine.board, promo, m)
        if engine.chess_board.is_en_passant(m):
            obj.piece_captured = BLACK_PAWN if engine.chess_board.turn else WHITE_PAWN
        return obj

    def to_chess(self):
        if self._chess_move is not None:
            return self._chess_move
        promo_piece = PROMOTION_TYPES.get(self.promotion) if self.promotion else None
        return chess.Move(
            rc_to_sq(self.start_row, self.start_col),
            rc_to_sq(self.end_row, self.end_col),
            promotion=promo_piece,
        )

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return self.to_chess() == other.to_chess()

    def __hash__(self):
        return hash(self.to_chess())

    def __repr__(self):
        promo_str = f", promotion={self.promotion}" if self.promotion else ""
        return f"Move(({self.start_row},{self.start_col})->({self.end_row},{self.end_col}){promo_str})"


class GameEngine:
    """
    Standard 8x8 Chess GameEngine implementation.
    Wraps python-chess with a 2D board matrix and course tournament interface.
    """

    def __init__(self):
        self.chess_board = chess.Board()
        self.move_log = []
        self.position_history = {}
        self._sync()
        self.update_position_history()

    def reset(self):
        self.chess_board.reset()
        self.move_log.clear()
        self.position_history.clear()
        self._sync()
        self.update_position_history()

    def _sync(self):
        self.board = []
        for r in range(BOARD_HEIGHT):
            row = []
            for c in range(BOARD_WIDTH):
                p = self.chess_board.piece_at(rc_to_sq(r, c))
                row.append(EMPTY_SQUARE if p is None else FROM_CHESS[(p.color, p.piece_type)])
            self.board.append(row)
        self.white_to_move = self.chess_board.turn

    def get_initial_board(self):
        initial = chess.Board()
        result = []
        for r in range(BOARD_HEIGHT):
            row = []
            for c in range(BOARD_WIDTH):
                piece = initial.piece_at(rc_to_sq(r, c))
                row.append(EMPTY_SQUARE if piece is None else FROM_CHESS[(piece.color, piece.piece_type)])
            result.append(row)
        return result

    def _position_key(self):
        return (
            self.chess_board.board_fen(),
            self.chess_board.turn,
            self.chess_board.castling_rights,
            self.chess_board.ep_square,
        )

    def update_position_history(self):
        k = self._position_key()
        self.position_history[k] = self.position_history.get(k, 0) + 1

    def get_repetition_count(self):
        if self.chess_board.is_repetition(3):
            return 3
        if self.chess_board.is_repetition(2):
            return 2
        return 1

    def get_legal_moves(self):
        return [Move.from_chess(m, self) for m in list(self.chess_board.legal_moves)]

    def _get_all_possible_moves(self):
        """Returns pseudo-legal mobility count helper for efficient evaluation."""
        return _PseudoCount(self.chess_board.legal_moves.count())

    def make_move(self, move):
        m = move.to_chess()
        if m not in self.chess_board.legal_moves:
            raise ValueError(f"Illegal move: {m.uci()}")
        if self.chess_board.is_en_passant(m):
            move.piece_captured = BLACK_PAWN if self.chess_board.turn else WHITE_PAWN
        self.move_log.append(move)
        self.chess_board.push(m)
        self._sync()
        self.update_position_history()

    def undo_move(self):
        if not self.chess_board.move_stack:
            return
        current_key = self._position_key()
        if current_key in self.position_history:
            self.position_history[current_key] -= 1
            if self.position_history[current_key] <= 0:
                del self.position_history[current_key]
        self.chess_board.pop()
        if self.move_log:
            self.move_log.pop()
        self._sync()

    def is_in_check(self):
        return self.chess_board.is_check()

    def _is_king_in_check(self, check_current_player=False):
        color = self.chess_board.turn if check_current_player else not self.chess_board.turn
        king = self.chess_board.king(color)
        return king is None or self.chess_board.is_attacked_by(not color, king)

    def _find_king(self, color):
        sq = self.chess_board.king(color == "w")
        return sq_to_rc(sq) if sq is not None else None

    def _is_square_attacked(self, square, friendly_color):
        r, c = square
        return self.chess_board.is_attacked_by(friendly_color != "w", rc_to_sq(r, c))

    def get_game_state(self):
        if self.chess_board.is_checkmate():
            return "checkmate"
        if self.chess_board.is_stalemate():
            return "stalemate"
        if (
            self.chess_board.is_insufficient_material()
            or self.chess_board.is_fifty_moves()
            or self.chess_board.is_repetition(3)
        ):
            return "draw"
        return "ongoing"
