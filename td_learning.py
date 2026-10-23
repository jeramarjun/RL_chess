"""
Temporal Difference Learning (TD-Learning) Engine for Chess Evaluation.
Implements TD(lambda) and TD(0) with linear-residual value approximation,
eligibility traces, potential-based reward shaping, and parameter persistence.
"""

import json
import math
import os
from config import (
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA,
    DEFAULT_ALPHA,
)
from features import FEATURE_NAMES, FEATURE_DIM, extract_features


class TDModel:
    """
    Temporal Difference Value Estimator V(s; w).
    Learns strategic weights for chess position evaluation:
        Score(s) = Baseline(s) + Correction_scale * sum(w_i * phi_i(s))
        V(s) = tanh(Score(s) / value_scale) in [-1.0, +1.0]
    """

    def __init__(
        self,
        alpha=DEFAULT_ALPHA,
        gamma=DEFAULT_GAMMA,
        lambda_param=DEFAULT_LAMBDA,
        value_scale=800.0,
        correction_scale=300.0,
        weights=None,
    ):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.lambda_param = float(lambda_param)
        self.value_scale = float(value_scale)
        self.correction_scale = float(correction_scale)

        # Initialize weights (material is anchored non-negative)
        self.weights = {name: 0.0 for name in FEATURE_NAMES}
        if weights:
            for name in FEATURE_NAMES:
                self.weights[name] = float(weights.get(name, 0.0))

        self.games_trained = 0
        self.total_updates = 0
        self.last_mean_td_error = 0.0

    def features(self, engine):
        """Extract strategic features from GameEngine."""
        return extract_features(engine)

    def correction(self, features_dict):
        """Compute the learned centipawn evaluation correction."""
        return self.correction_scale * sum(
            self.weights[name] * features_dict[name] for name in FEATURE_NAMES
        )

    def evaluate_board_score(self, baseline_score, features_dict):
        """Return combined centipawn evaluation score from White's perspective."""
        return baseline_score + self.correction(features_dict)

    def value(self, baseline_score, features_dict):
        """
        Normalized state value in [-1.0, +1.0] from White's perspective.
        +1.0 indicates White is winning; -1.0 indicates Black is winning.
        """
        score = self.evaluate_board_score(baseline_score, features_dict)
        return math.tanh(score / self.value_scale)

    def gradient(self, value, features_dict):
        """
        Compute gradient dV / dw_i = (1 - V^2) * (correction_scale / value_scale) * phi_i.
        """
        scale = (1.0 - value * value) * (self.correction_scale / self.value_scale)
        return {name: scale * features_dict[name] for name in FEATURE_NAMES}

    def learn_episode(self, trajectory, final_outcome, shaped_rewards=None):
        """
        Perform episodic TD(lambda) update across the played trajectory.

        Parameters:
            trajectory: List of tuples (baseline_score, features_dict) for each ply.
            final_outcome: Terminal result from White's perspective (+1.0 White win, -1.0 Black win, 0.0 Draw,
                           or None for truncated ply-limit games bootstrapping from final state).
            shaped_rewards: Optional list of intermediate rewards per ply.

        Returns:
            mean_abs_td_error: Mean absolute TD error over the episode.
        """
        if not trajectory:
            return 0.0

        n = len(trajectory)
        values = [self.value(b_score, feat) for b_score, feat in trajectory]
        td_errors = []

        eligibility = {name: 0.0 for name in FEATURE_NAMES}
        effective_alpha = max(0.002, self.alpha / (1.0 + 0.0008 * self.games_trained))

        for t in range(n):
            v_t = values[t]
            r_t = shaped_rewards[t] if shaped_rewards and t < len(shaped_rewards) else 0.0

            if t + 1 < n:
                v_target = r_t + self.gamma * values[t + 1]
            else:
                v_target = float(final_outcome) if final_outcome is not None else values[t]

            delta = v_target - v_t
            td_errors.append(abs(delta))

            grad_t = self.gradient(v_t, trajectory[t][1])

            decay = self.gamma * self.lambda_param
            for name in FEATURE_NAMES:
                eligibility[name] = decay * eligibility[name] + grad_t[name]
                update = effective_alpha * delta * eligibility[name]
                self.weights[name] += update

                # Constrain weights: material is bounded non-negative to preserve chess rules
                if name == "material":
                    self.weights[name] = max(0.0, min(1.5, self.weights[name]))
                else:
                    self.weights[name] = max(-1.0, min(1.0, self.weights[name]))

        self.games_trained += 1
        self.total_updates += n
        self.last_mean_td_error = sum(td_errors) / len(td_errors)
        return self.last_mean_td_error

    def save(self, filepath):
        """Save model hyperparameters and learned weights to JSON."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {
            "algorithm": "TD(lambda)_residual_evaluator",
            "feature_names": FEATURE_NAMES,
            "feature_dim": FEATURE_DIM,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "lambda_param": self.lambda_param,
            "value_scale": self.value_scale,
            "correction_scale": self.correction_scale,
            "weights": self.weights,
            "games_trained": self.games_trained,
            "total_updates": self.total_updates,
            "last_mean_td_error": self.last_mean_td_error,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath):
        """Load trained model weights from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = cls(
            alpha=data.get("alpha", DEFAULT_ALPHA),
            gamma=data.get("gamma", DEFAULT_GAMMA),
            lambda_param=data.get("lambda_param", DEFAULT_LAMBDA),
            value_scale=data.get("value_scale", 800.0),
            correction_scale=data.get("correction_scale", 300.0),
            weights=data.get("weights", {}),
        )
        model.games_trained = int(data.get("games_trained", 0))
        model.total_updates = int(data.get("total_updates", 0))
        model.last_mean_td_error = float(data.get("last_mean_td_error", 0.0))
        return model
