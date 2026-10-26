"""
Systematic Performance Comparison and Tournament Evaluation Module.
Evaluates the trained RL Chess Engine against 4 baseline opponents:
1. Random Baseline
2. Greedy Material Player
3. Untrained RL Player (theta_0 prior)
4. Heuristic Baseline (AIPlayer from ai_player_adaptive.py)

Uses paired matches (alternating White and Black) under equal search budgets
to compute win/draw/loss rates, score percentages, and Elo differentials.
"""

import argparse
import json
import math
import os
import random
import time

from board import GameEngine
from config import (
    DEFAULT_SEARCH_DEPTH,
    DEFAULT_QUIESCENCE_DEPTH,
)
from td_learning import TDModel
from rl_player import RLAIPlayer
from baselines import (
    RandomPlayer,
    GreedyMaterialPlayer,
    UntrainedRLPlayer,
    HeuristicPlayer,
)


def play_tournament_game(player_white, player_black, max_plies=120):
    """
    Play a single tournament game between White and Black.
    Returns:
        winner: "White", "Black", or "Draw"
        plies: int
        state: final state string
    """
    engine = player_white.engine
    # Ensure both players reference the exact same engine
    player_black.engine = engine

    for ply in range(max_plies):
        state = engine.get_game_state()
        if state != "ongoing":
            break

        player = player_white if engine.white_to_move else player_black
        move = player.get_best_move()

        if move is None:
            break

        engine.make_move(move)

    final_state = engine.get_game_state()
    plies_played = len(engine.move_log)

    if final_state == "checkmate":
        # If White to move is checkmated, Black won; if Black to move is checkmated, White won
        winner = "Black" if engine.white_to_move else "White"
    else:
        winner = "Draw"

    return winner, plies_played, final_state


def compute_elo_diff(score_pct):
    """
    Compute Elo rating difference from score percentage using the logistic formula:
    Delta Elo = -400 * log10(1 / Score - 1)
    """
    s = max(0.01, min(0.99, score_pct / 100.0))
    return round(-400.0 * math.log10(1.0 / s - 1.0), 1)


def run_benchmark(model_path, games_per_opponent=10, max_plies=120, search_depth=3):
    """Run systematic tournament benchmark across all baseline opponents."""
    model = TDModel.load(model_path) if model_path and os.path.exists(model_path) else TDModel()

    opponents = [
        ("Random Baseline", RandomPlayer),
        ("Greedy Material", GreedyMaterialPlayer),
        ("Untrained Baseline (w=0)", UntrainedRLPlayer),
        ("Heuristic AIPlayer", HeuristicPlayer),
    ]

    results = {}
    print("=" * 80)
    print("  SYSTEMATIC PERFORMANCE EVALUATION: RL AGENT VS BASELINES")
    print(f"  Model: {model_path} | Games per Opponent: {games_per_opponent} (Paired White/Black)")
    print(f"  Search Depth: {search_depth} | Quiescence: 2 | Max Plies: {max_plies}")
    print("=" * 80)

    for opp_name, opp_cls in opponents:
        print(f"\nEvaluating vs. {opp_name} ({games_per_opponent} games)...")
        wins = 0
        draws = 0
        losses = 0
        total_plies = 0

        for game_idx in range(games_per_opponent):
            engine = GameEngine()
            rl_agent = RLAIPlayer(engine, model=model, training_mode=False)
            rl_agent.max_depth = search_depth
            rl_agent.quiescence_depth = 2

            opponent = opp_cls(engine)
            if hasattr(opponent, "max_depth"):
                opponent.max_depth = search_depth
                opponent.quiescence_depth = 2

            rl_is_white = (game_idx % 2 == 0)
            white_player = rl_agent if rl_is_white else opponent
            black_player = opponent if rl_is_white else rl_agent

            winner, plies, state = play_tournament_game(white_player, black_player, max_plies)
            total_plies += plies

            if (winner == "White" and rl_is_white) or (winner == "Black" and not rl_is_white):
                wins += 1
                outcome_char = "WIN "
            elif winner == "Draw":
                draws += 1
                outcome_char = "DRAW"
            else:
                losses += 1
                outcome_char = "LOSS"

            print(f"  Game {game_idx+1:2d}/{games_per_opponent}: {outcome_char} | State: {state:12s} | Plies: {plies:2d} | RL: {'White' if rl_is_white else 'Black'}")

        total_games = wins + draws + losses
        win_rate = round(100.0 * wins / total_games, 1)
        draw_rate = round(100.0 * draws / total_games, 1)
        loss_rate = round(100.0 * losses / total_games, 1)
        score_pct = round(100.0 * (wins + 0.5 * draws) / total_games, 1)
        elo_diff = compute_elo_diff(score_pct)
        avg_plies = round(total_plies / total_games, 1)

        results[opp_name] = {
            "games": total_games,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "loss_rate": loss_rate,
            "score_pct": score_pct,
            "elo_diff": elo_diff,
            "avg_plies": avg_plies,
        }

        print(f"--> Result vs {opp_name}: {wins}W - {draws}D - {losses}L (Score: {score_pct}%, Delta Elo: {elo_diff:+5.1f})")

    # Save results to JSON
    os.makedirs("results", exist_ok=True)
    out_file = "results/evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("  FINAL TOURNAMENT PERFORMANCE SUMMARY")
    print("=" * 80)
    header = f"{'Opponent Baseline':<28} | {'Games':<5} | {'W':<3} | {'D':<3} | {'L':<3} | {'Score %':<7} | {'Win %':<6} | {'Delta Elo':<9}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<28} | {r['games']:<5} | {r['wins']:<3} | {r['draws']:<3} | {r['losses']:<3} | {r['score_pct']:<7.1f} | {r['win_rate']:<6.1f} | {r['elo_diff']:+9.1f}")
    print("=" * 80)
    print(f"Results saved to: {out_file}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL Chess Engine against Baselines")
    parser.add_argument("--model", type=str, default="trained_weights.json", help="Path to trained model JSON")
    parser.add_argument("--games", type=int, default=10, help="Games per opponent baseline")
    parser.add_argument("--max-plies", type=int, default=100, help="Maximum plies per match")
    parser.add_argument("--depth", type=int, default=3, help="Search depth for match play")
    parser.add_argument("--seed", type=int, default=2027, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    run_benchmark(
        model_path=args.model,
        games_per_opponent=args.games,
        max_plies=args.max_plies,
        search_depth=args.depth,
    )


if __name__ == "__main__":
    main()
