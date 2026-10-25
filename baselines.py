"""
Baseline Agents for Systematic Performance Comparisons and Tournament Benchmarking.
Implements:
1. RandomPlayer: Baseline stochastic agent selecting legal moves uniformly.
2. GreedyMaterialPlayer: 1-ply greedy agent maximizing immediate material balance.
3. UntrainedRLPlayer: Alpha-Beta search agent with zero-weight unlearned prior (w = 0).
4. HeuristicPlayer: Baseline AIPlayer from ai_player_adaptive.py with static heuristic weights.
"""

import random
from config import EMPTY_SQUARE, PIECE_VALUES
from ai_player_adaptive import AIPlayer
from rl_player import RLAIPlayer
from td_learning import TDModel


class RandomPlayer:
    """Baseline player that chooses a uniformly random legal move."""

    def __init__(self, board):
        self.engine = board

    def get_best_move(self):
        legal = self.engine.get_legal_moves()
        return random.choice(legal) if legal else None


class GreedyMaterialPlayer:
    """Baseline player choosing the move with maximum immediate material gain."""

    def __init__(self, board):
        self.engine = board

    def get_best_move(self):
        legal = self.engine.get_legal_moves()
        if not legal:
            return None

        best_score = -999999
        best_move = legal[0]
        is_white = self.engine.white_to_move

        for m in legal:
            self.engine.make_move(m)
            try:
                # Material sum from our perspective
                mat = 0
                for row in self.engine.board:
                    for piece in row:
                        if piece != EMPTY_SQUARE:
                            v = PIECE_VALUES.get(piece, 0)
                            mat += v if is_white else -v
                if mat > best_score:
                    best_score = mat
                    best_move = m
            finally:
                self.engine.undo_move()

        return best_move


class UntrainedRLPlayer(RLAIPlayer):
    """RL Player initialized with strictly zero weights (w = 0) to evaluate pure learning gains."""

    def __init__(self, board):
        model = TDModel()  # All weights are 0.0
        super().__init__(board, model=model, training_mode=False)


class HeuristicPlayer(AIPlayer):
    """The original unaugmented baseline AIPlayer from ai_player_adaptive.py."""

    def __init__(self, board):
        super().__init__(board)
