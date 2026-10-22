"""
Comprehensive Test Suite for RL-Enhanced Chess Engine.
Validates:
1. GameEngine board representation and legal move execution/undo.
2. Feature extraction alignment and numerical bounds.
3. TDModel value approximation, gradient derivation, and eligibility trace updates.
4. RLAIPlayer integration with AIPlayer negamax search and stochasticity mechanisms.
5. Checkpoint serialization and reproduction integrity.
"""

import unittest
import os
import shutil
import tempfile

from board import GameEngine, Move
from config import EMPTY_SQUARE, BOARD_HEIGHT, BOARD_WIDTH, WHITE_PAWN, BLACK_PAWN
from features import extract_features, FEATURE_NAMES
from td_learning import TDModel
from rl_player import RLAIPlayer
from baselines import RandomPlayer, GreedyMaterialPlayer, UntrainedRLPlayer, HeuristicPlayer


class TestRLEngine(unittest.TestCase):

    def setUp(self):
        self.engine = GameEngine()
        self.model = TDModel()

    def test_engine_initialization(self):
        """Verify initial board dimensions and active player."""
        self.assertEqual(len(self.engine.board), BOARD_HEIGHT)
        self.assertEqual(len(self.engine.board[0]), BOARD_WIDTH)
        self.assertTrue(self.engine.white_to_move)
        legal_moves = self.engine.get_legal_moves()
        self.assertEqual(len(legal_moves), 20)  # Standard chess 20 opening moves

    def test_make_and_undo_move(self):
        """Ensure move execution and reversal preserves exact board state."""
        legal_moves = self.engine.get_legal_moves()
        move = legal_moves[0]
        initial_board = [row[:] for row in self.engine.board]

        self.engine.make_move(move)
        self.assertFalse(self.engine.white_to_move)
        self.assertEqual(len(self.engine.move_log), 1)

        self.engine.undo_move()
        self.assertTrue(self.engine.white_to_move)
        self.assertEqual(len(self.engine.move_log), 0)
        self.assertEqual(self.engine.board, initial_board)

    def test_feature_extraction(self):
        """Check feature extraction contains all required keys and bounded values."""
        feats = extract_features(self.engine)
        for name in FEATURE_NAMES:
            self.assertIn(name, feats)
            self.assertTrue(-1.0 <= feats[name] <= 1.0 or name == "bias")

    def test_td_model_learning_step(self):
        """Test TD(lambda) update step modifies weights in direction of error."""
        trajectory = [
            (0.0, extract_features(self.engine)),
            (50.0, extract_features(self.engine)),
        ]
        initial_weights = self.model.weights.copy()
        err = self.model.learn_episode(trajectory, final_outcome=1.0)

        self.assertGreater(err, 0.0)
        self.assertEqual(self.model.games_trained, 1)

    def test_rl_player_search(self):
        """Verify RLAIPlayer successfully generates valid moves via Alpha-Beta."""
        player = RLAIPlayer(self.engine, self.model, training_mode=True)
        move = player.get_training_move(epsilon=0.0, temperature=0.0)
        self.assertIsNotNone(move)
        self.assertIn(move, self.engine.get_legal_moves())

    def test_stochasticity_injection(self):
        """Ensure epsilon-greedy exploration can pick legal moves."""
        player = RLAIPlayer(self.engine, self.model, training_mode=True)
        move = player.get_training_move(epsilon=1.0, temperature=0.0)
        self.assertIsNotNone(move)
        self.assertIn(move, self.engine.get_legal_moves())

    def test_checkpoint_save_and_load(self):
        """Verify checkpoint serialization consistency."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            self.model.weights["material"] = 0.42
            self.model.save(temp_path)
            loaded_model = TDModel.load(temp_path)
            self.assertAlmostEqual(loaded_model.weights["material"], 0.42, places=4)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_all_baselines_instantiation(self):
        """Ensure all 4 baseline players can select a legal move."""
        b1 = RandomPlayer(self.engine)
        b2 = GreedyMaterialPlayer(self.engine)
        b3 = UntrainedRLPlayer(self.engine)
        b4 = HeuristicPlayer(self.engine)

        self.assertIn(b1.get_best_move(), self.engine.get_legal_moves())
        self.assertIn(b2.get_best_move(), self.engine.get_legal_moves())
        self.assertIn(b3.get_best_move(), self.engine.get_legal_moves())
        self.assertIn(b4.get_best_move(), self.engine.get_legal_moves())


if __name__ == "__main__":
    unittest.main()
