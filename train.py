"""
Self-Play Reinforcement Learning Training Pipeline for 8x8 Chess.
Trains the TD-Leaf(lambda) evaluation function through autonomous self-play,
integrating exploratory stochasticity (randomized opening plies, decaying epsilon-greedy,
and temperature sampling) with eligibility traces and potential-based reward shaping.
"""

import argparse
import csv
import json
import math
import os
import random
import time

from board import GameEngine
from config import (
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA,
    DEFAULT_ALPHA,
    DEFAULT_EPSILON_START,
    DEFAULT_EPSILON_MIN,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_EPISODES,
)
from td_learning import TDModel
from rl_player import RLAIPlayer


def outcome_for_white(state, engine, is_ply_limit=False):
    """Determine outcome from White's perspective (+1.0 White win, -1.0 Black win, 0.0 Draw, None for ply limit)."""
    if state == "checkmate":
        return -1.0 if engine.white_to_move else 1.0
    if is_ply_limit:
        return None
    return 0.0


def play_self_play_episode(
    model,
    epsilon,
    temperature,
    max_plies=80,
    random_opening_plies=6,
):
    """
    Play one autonomous self-play episode with exploratory stochasticity.

    Returns:
        outcome: float in {+1.0, -1.0, 0.0} or None
        trajectory: list of (baseline_score, features_dict)
        shaped_rewards: list of intermediate transition rewards
        state: terminal state string
        plies: number of plies played
    """
    engine = GameEngine()
    white_player = RLAIPlayer(engine, model, training_mode=True)
    black_player = RLAIPlayer(engine, model, training_mode=True)

    trajectory = []
    shaped_rewards = []
    prev_material = 0.0

    hit_ply_limit = False

    for ply in range(max_plies):
        state = engine.get_game_state()
        if state != "ongoing":
            break

        base_score = white_player.baseline_score()
        features = model.features(engine)
        trajectory.append((base_score, features))

        # Reward shaping: potential-based material differential
        curr_material = features["material"]
        delta_mat = curr_material - prev_material
        shaped_reward = 0.10 * delta_mat
        shaped_rewards.append(shaped_reward)
        prev_material = curr_material

        legal = engine.get_legal_moves()
        if not legal:
            break

        # Exploratory stochasticity
        if ply < random_opening_plies:
            move = random.choice(legal)
        else:
            player = white_player if engine.white_to_move else black_player
            move = player.get_training_move(epsilon=epsilon, temperature=temperature)

        if move is None:
            break

        engine.make_move(move)
    else:
        hit_ply_limit = True

    final_state = engine.get_game_state()
    outcome = outcome_for_white(final_state, engine, is_ply_limit=hit_ply_limit)

    return outcome, trajectory, shaped_rewards, final_state, len(trajectory)


def main():
    parser = argparse.ArgumentParser(description="Train RL Chess Evaluator via Self-Play TD-Learning")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Total training episodes")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Checkpoint frequency")
    parser.add_argument("--max-plies", type=int, default=80, help="Maximum plies per episode")
    parser.add_argument("--random-opening-plies", type=int, default=6, help="Stochastic opening half-moves")
    parser.add_argument("--epsilon-start", type=float, default=DEFAULT_EPSILON_START, help="Initial exploration rate")
    parser.add_argument("--epsilon-min", type=float, default=DEFAULT_EPSILON_MIN, help="Minimum exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=DEFAULT_EPSILON_DECAY, help="Exploration decay factor")
    parser.add_argument("--temperature", type=float, default=0.20, help="Softmax exploration temperature")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="TD learning rate")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Discount factor")
    parser.add_argument("--lambda-param", type=float, default=DEFAULT_LAMBDA, help="Eligibility trace decay")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
    parser.add_argument("--out", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save logs and metrics")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    csv_log_path = os.path.join(args.results_dir, "training_log.csv")
    metrics_json_path = os.path.join(args.results_dir, "training_metrics.json")
    final_weights_path = "trained_weights.json"

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        model = TDModel.load(args.resume)
    else:
        model = TDModel(
            alpha=args.alpha,
            gamma=args.gamma,
            lambda_param=args.lambda_param,
        )

    print("=" * 70)
    print("  REINFORCEMENT LEARNING CHESS ENGINE: SELF-PLAY TD-LEARNING")
    print(f"  Target Episodes:       {args.episodes}")
    print(f"  Random Opening Plies:  {args.random_opening_plies} (Exploration Stochasticity)")
    print(f"  Epsilon Exploration:   {args.epsilon_start:.2f} -> {args.epsilon_min:.2f} (decay: {args.epsilon_decay})")
    print(f"  TD Hyperparameters:    alpha={model.alpha}, gamma={model.gamma}, lambda={model.lambda_param}")
    print("=" * 70)

    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "global_games",
            "outcome",
            "result_state",
            "plies",
            "epsilon",
            "td_error",
            "elapsed_sec",
            *model.weights.keys(),
        ])

    metrics_history = []
    start_total_time = time.perf_counter()

    current_epsilon = args.epsilon_start
    current_temp = args.temperature

    for episode in range(1, args.episodes + 1):
        t_start = time.perf_counter()

        outcome, trajectory, shaped_rewards, state, plies = play_self_play_episode(
            model=model,
            epsilon=current_epsilon,
            temperature=current_temp,
            max_plies=args.max_plies,
            random_opening_plies=args.random_opening_plies,
        )

        td_error = model.learn_episode(trajectory, outcome, shaped_rewards)

        current_epsilon = max(args.epsilon_min, current_epsilon * args.epsilon_decay)
        current_temp = max(0.02, current_temp * args.epsilon_decay)

        elapsed = time.perf_counter() - t_start

        out_val = outcome if outcome is not None else 0.0
        rec = {
            "episode": episode,
            "global_games": model.games_trained,
            "outcome": out_val,
            "result_state": state,
            "plies": plies,
            "epsilon": round(current_epsilon, 4),
            "td_error": round(td_error, 5),
            "weights": {k: round(v, 5) for k, v in model.weights.items()},
            "elapsed": round(elapsed, 3),
        }
        metrics_history.append(rec)

        with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                model.games_trained,
                out_val,
                state,
                plies,
                round(current_epsilon, 4),
                round(td_error, 5),
                round(elapsed, 3),
                *[round(model.weights[k], 5) for k in model.weights],
            ])

        if episode % 25 == 0 or episode == 1 or episode == args.episodes:
            total_elapsed = time.perf_counter() - start_total_time
            avg_plies = sum(m["plies"] for m in metrics_history[-25:]) / min(25, len(metrics_history))
            out_str = f"{outcome:+4.1f}" if outcome is not None else "boot"
            print(
                f"Episode {episode:4d}/{args.episodes:4d} | "
                f"State: {state:13s} | "
                f"Outcome: {out_str} | "
                f"Plies: {plies:2d} (avg {avg_plies:4.1f}) | "
                f"TD Error: {td_error:.4f} | "
                f"Eps: {current_epsilon:.3f} | "
                f"Total Time: {total_elapsed:6.1f}s",
                flush=True,
            )

        if episode % args.checkpoint_every == 0 or episode == args.episodes:
            ckpt_path = os.path.join(args.out, f"checkpoint_episode_{episode:05d}.json")
            model.save(ckpt_path)
            model.save(os.path.join(args.out, "best_model.json"))
            model.save(final_weights_path)

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "hyperparameters": {
                "episodes": args.episodes,
                "alpha": model.alpha,
                "gamma": model.gamma,
                "lambda_param": model.lambda_param,
                "epsilon_start": args.epsilon_start,
                "epsilon_min": args.epsilon_min,
                "random_opening_plies": args.random_opening_plies,
            },
            "final_weights": model.weights,
            "training_history": metrics_history,
        }, f, indent=2)

    total_time = time.perf_counter() - start_total_time
    print("=" * 70)
    print(f"Training successfully completed in {total_time:.2f} seconds ({total_time/args.episodes:.3f}s/episode)!")
    print(f"Final Learned Weights:")
    for name, w in model.weights.items():
        print(f"  {name:22s}: {w:+.5f}")
    print(f"Saved weights to: {final_weights_path}")
    print(f"Saved metrics to: {metrics_json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
