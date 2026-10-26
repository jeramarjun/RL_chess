"""
Plotting and Publication-Quality Figure Generation Module.
Generates all graphs required by the assignment specification:
1. figure1_td_error_convergence.png: TD Error (delta) and loss convergence over episodes.
2. figure2_weight_trajectories.png: Reinforcement Learning feature weights evolution.
3. figure3_game_lengths_and_outcomes.png: Episode length (plies) and rolling win/draw rate.
4. figure4_baseline_tournament.png: Systematic tournament benchmark comparison vs baselines.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def plot_td_error(metrics_data, out_path="figures/figure1_td_error_convergence.png"):
    """Plot TD error curve with moving average."""
    history = metrics_data.get("training_history", [])
    if not history:
        print("No training history to plot TD error.")
        return

    episodes = [m["episode"] for m in history]
    errors = [m["td_error"] for m in history]

    # Exponential moving average
    window = min(25, len(errors))
    ema = []
    current = errors[0]
    alpha = 2.0 / (window + 1.0)
    for e in errors:
        current = alpha * e + (1.0 - alpha) * current
        ema.append(current)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(episodes, errors, color="#90CAF9", alpha=0.45, linewidth=1.0, label="Raw TD Error |δ|")
    ax.plot(episodes, ema, color="#0D47A1", linewidth=2.2, label=f"EMA (window={window})")

    ax.set_title("Temporal-Difference Error Convergence over Self-Play Training", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Training Episode (Self-Play Games)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Absolute TD Error (|δ|)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def plot_weight_trajectories(metrics_data, out_path="figures/figure2_weight_trajectories.png"):
    """Plot evolution of evaluation weights over episodes."""
    history = metrics_data.get("training_history", [])
    if not history:
        return

    episodes = [m["episode"] for m in history]
    weight_keys = list(history[0]["weights"].keys())

    colors = ["#D32F2F", "#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#0097A7", "#5D4037", "#616161"]

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    for idx, key in enumerate(weight_keys):
        series = [m["weights"][key] for m in history]
        color = colors[idx % len(colors)]
        label_name = key.replace("_", " ").title()
        ax.plot(episodes, series, label=label_name, color=color, linewidth=1.8)

    ax.axhline(0, color="black", linestyle=":", alpha=0.6, linewidth=1.0)
    ax.set_title("Strategic Evaluation Weights Trajectory via TD(λ) Updates", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Training Episode", fontsize=10, fontweight="bold")
    ax.set_ylabel("Normalized Feature Weight (w_i)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=8.5)
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def plot_game_lengths_and_outcomes(metrics_data, out_path="figures/figure3_game_lengths_and_outcomes.png"):
    """Plot episode plies and rolling win/draw rate."""
    history = metrics_data.get("training_history", [])
    if not history:
        return

    episodes = [m["episode"] for m in history]
    plies = [m["plies"] for m in history]
    outcomes = [m["outcome"] for m in history]

    # Rolling win/draw rates (window 50)
    window = min(50, len(outcomes))
    rolling_wins = []
    rolling_draws = []
    for i in range(len(outcomes)):
        start_idx = max(0, i - window + 1)
        sub = outcomes[start_idx : i + 1]
        w_cnt = sum(1 for x in sub if x == 1.0)
        d_cnt = sum(1 for x in sub if x == 0.0)
        rolling_wins.append(100.0 * w_cnt / len(sub))
        rolling_draws.append(100.0 * d_cnt / len(sub))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=300)

    # Subplot 1: Plies per episode
    ax1.plot(episodes, plies, color="#78909C", alpha=0.45, linewidth=1.0)
    plies_ema = []
    cur_p = plies[0]
    for p in plies:
        cur_p = 0.08 * p + 0.92 * cur_p
        plies_ema.append(cur_p)
    ax1.plot(episodes, plies_ema, color="#37474F", linewidth=2.0, label="Plies (EMA)")
    ax1.set_ylabel("Episode Plies", fontsize=9, fontweight="bold")
    ax1.set_title("Self-Play Dynamics: Game Lengths & Exploration Convergence", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Subplot 2: Win & Draw Rates
    ax2.plot(episodes, rolling_wins, color="#2E7D32", linewidth=2.0, label="White Win Rate %")
    ax2.plot(episodes, rolling_draws, color="#F57F17", linewidth=2.0, label="Draw Rate %")
    ax2.set_xlabel("Training Episode", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Rate (%)", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def plot_baseline_tournament(eval_path="results/evaluation_results.json", out_path="figures/figure4_baseline_tournament.png"):
    """Plot comparative tournament benchmark results against baselines."""
    if not os.path.exists(eval_path):
        print(f"Evaluation file {eval_path} not found. Skipping tournament plot.")
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    opponents = list(data.keys())
    scores = [data[op]["score_pct"] for op in opponents]
    win_rates = [data[op]["win_rate"] for op in opponents]
    draw_rates = [data[op]["draw_rate"] for op in opponents]
    elo_diffs = [data[op]["elo_diff"] for op in opponents]

    x = np.arange(len(opponents))
    width = 0.28

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    # Left plot: Score % and Win Rate %
    rects1 = ax1.bar(x - width/2, scores, width, label="Score %", color="#1565C0")
    rects2 = ax1.bar(x + width/2, win_rates, width, label="Win Rate %", color="#2E7D32")
    ax1.set_ylabel("Percentage (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Head-to-Head Tournament Scores vs Baselines", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([op.replace(" ", "\n") for op in opponents], fontsize=8.5)
    ax1.set_ylim(0, 110)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5)

    for bar in rects1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.0f}%", ha="center", va="bottom", fontsize=8)

    # Right plot: Elo Difference
    colors = ["#2E7D32" if e >= 0 else "#C62828" for e in elo_diffs]
    rects3 = ax2.bar(x, elo_diffs, width=0.45, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Rating Difference (ΔElo)", fontsize=10, fontweight="bold")
    ax2.set_title("Empirical Elo Gain (ΔElo)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([op.replace(" ", "\n") for op in opponents], fontsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in rects3:
        yval = bar.get_height()
        va = "bottom" if yval >= 0 else "top"
        offset = 8 if yval >= 0 else -18
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f"{yval:+.0f}", ha="center", va=va, fontsize=8, fontweight="bold")

    plt.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def generate_all_plots(metrics_file="results/training_metrics.json", eval_file="results/evaluation_results.json"):
    """Generate all 4 required figures."""
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        plot_td_error(data)
        plot_weight_trajectories(data)
        plot_game_lengths_and_outcomes(data)
    else:
        print(f"Metrics file {metrics_file} not found.")

    if os.path.exists(eval_file):
        plot_baseline_tournament(eval_file)
    else:
        print(f"Evaluation file {eval_file} not found.")


if __name__ == "__main__":
    generate_all_plots()
