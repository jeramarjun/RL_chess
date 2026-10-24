"""
Reinforcement Learning Enhanced AI Player for 8x8 Chess.
Directly subclasses AIPlayer from ai_player_adaptive.py and replaces static evaluation
with the learned TD(lambda) value function while preserving the full Alpha-Beta Negamax search.
"""

import math
import random
import time

from config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    EMPTY_SQUARE,
    PIECE_VALUES,
)
from ai_player_adaptive import AIPlayer
from td_learning import TDModel


class RLAIPlayer(AIPlayer):
    """
    Reinforcement Learning Chess Agent.
    Combines the iterative-deepening Alpha-Beta Negamax search from ai_player_adaptive.py
    with the learned TD(lambda) evaluation weights.
    """

    def __init__(self, board, model=None, training_mode=False):
        super().__init__(board)
        self.td_model = model if model is not None else TDModel()
        self.training_mode = training_mode

        if self.training_mode:
            # Training parameters
            self.total_time_budget = 10.0
            self.safety_reserve = 0.5
            self.max_moves = 60
            self.max_depth = 2
            self.quiescence_depth = 1
            self.mcts_simulations = 0
        else:
            # Full tournament configuration
            self.total_time_budget = 58.0
            self.safety_reserve = 2.0
            self.max_moves = 75
            self.max_depth = 8
            self.quiescence_depth = 5
            self.mcts_simulations = 8

    def baseline_score(self):
        """Compute the unlearned static baseline evaluation from White's perspective."""
        return super().evaluate_board()

    def evaluate_board(self):
        """
        Learned evaluation function from White's perspective.
        Combines baseline heuristic score with the learned TD feature correction.
        """
        base = self.baseline_score()
        features = self.td_model.features(self.engine)
        return self.td_model.evaluate_board_score(base, features)

    def get_training_move(self, epsilon=0.10, temperature=0.0):
        """
        Select a move with injected exploration stochasticity for training self-play.

        Stochasticity mechanisms:
        1. Epsilon-greedy exploration: choose a uniformly random legal move with probability epsilon.
        2. Softmax / Boltzmann sampling: when temperature > 0, sample moves according to exp(Q / tau).
        3. Random tie-breaking: break ties among equally optimal moves uniformly at random.
        """
        legal = self.engine.get_legal_moves()
        if not legal:
            return None

        # 1. Epsilon-greedy exploration
        if random.random() < epsilon:
            return random.choice(legal)

        # 2. Fast 1-ply evaluation under the learned TD evaluation function
        is_white = self.engine.white_to_move
        scores = []

        for m in legal:
            self.engine.make_move(m)
            try:
                # Evaluation from the side to move perspective
                white_eval = self.evaluate_board()
                score = white_eval if is_white else -white_eval
            finally:
                self.engine.undo_move()
            scores.append(score)

        if temperature > 0.05 and len(legal) > 1:
            max_s = max(scores)
            exp_scores = [
                math.exp(min(20.0, max(-20.0, (s - max_s) / (temperature * 50.0))))
                for s in scores
            ]
            total_exp = sum(exp_scores)
            if total_exp > 0:
                probs = [e / total_exp for e in exp_scores]
                return random.choices(legal, weights=probs, k=1)[0]

        # Greedy choice with random tie-breaking
        max_score = max(scores)
        best_moves = [m for m, s in zip(legal, scores) if abs(s - max_score) < 1e-4]
        return random.choice(best_moves)
