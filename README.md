# Reinforcement Learning-Enhanced Chess Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![RL Algorithm](https://img.shields.io/badge/Algorithm-TD--Leaf(%CE%BB)%20Self--Play-orange.svg)]()
[![Search Engine](https://img.shields.io/badge/Search-Alpha--Beta%20Pruning%20%2B%20Quiescence-green.svg)]()
[![Course](https://img.shields.io/badge/IIT%20Jodhpur-AI%20Assignment%202%20(Mini--Project)-red.svg)]()

> **Autonomous 8×8 Chess Strategic Decision-Making via Temporal-Difference Learning, Self-Play Value Optimization, Exploratory Stochasticity, and Minimax Search Integration.**

---

## 📌 Project Overview

Traditional computer chess systems rely heavily on hand-crafted heuristic evaluation functions and supervised grandmaster opening books. While effective, heuristic functions encode designer bias, require decades of trial-and-error manual tuning, and frequently fail in non-standard tactical or deep positional positions.

**This project formulates standard 8×8 chess strictly as a Reinforcement Learning (RL) problem.** The agent autonomously learns strategic position evaluation from scratch through self-play and **Temporal-Difference Learning (TD($\lambda$))**.

The learned evaluation function is integrated directly into a high-performance **Iterative-Deepening Negamax search engine featuring $\alpha$-$\beta$ pruning, quiescence search, transposition tables, move ordering (MVV-LVA, killer moves, history heuristic), and blitz clock management (60-second shared clock)**.

---

## 🔬 Reinforcement Learning Formulation

In compliance with the assignment specification, the environment and agent are formulated as a Markov Decision Process (MDP): $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$.

### 1. State Space ($\mathcal{S}$)
A chess state $s \in \mathcal{S}$ is formally represented by the tuple:
$$s = \langle \mathbf{B}, t, \mathbf{c}, ep, h_{50}, h_{\text{rep}} \rangle$$
- $\mathbf{B} \in \{0, 1\}^{12 \times 8 \times 8}$: Spatial occupancy tensor encoding the positions of the 6 piece types ($\{P, N, B, R, Q, K\}$) for both colors across 64 squares.
- $t \in \{0, 1\}$: Active player turn (White or Black).
- $\mathbf{c} \in \{0, 1\}^4$: Castling rights (kingside/queenside for White and Black).
- $ep \in \{0, \dots, 63\} \cup \{\emptyset\}$: En-passant target square availability.
- $h_{50} \in [0, 100]$: Half-move counter for the 50-move draw rule.
- $h_{\text{rep}}$: Position repetition history for threefold repetition detection.

To facilitate sample-efficient TD gradient propagation, each state is mapped to a normalized 8-dimensional strategic feature vector $\boldsymbol{\phi}(s) \in \mathbb{R}^8$:
$$\boldsymbol{\phi}(s) = \left[ \phi_{\text{mat}}, \phi_{\text{pst}}, \phi_{\text{minor}}, \phi_{\text{rook}}, \phi_{\text{pawn}}, \phi_{\text{mob}}, \phi_{\text{king}}, 1.0 \right]^T$$

| Feature Category | Description | Normalization |
| :--- | :--- | :--- |
| **Material Balance ($\phi_{\text{mat}}$)** | Net piece differential ($P=100, N=320, B=330, R=500, Q=900$) | Bounded in $[-1.0, +1.0]$ ($\div 4000$) |
| **Piece-Square Tables ($\phi_{\text{pst}}$)** | Centralization and square-occupancy metrics per piece type | Bounded in $[-1.0, +1.0]$ ($\div 1000$) |
| **Minor Activity ($\phi_{\text{minor}}$)** | Center square pressure ($d4/e4/d5/e5$) by knights and bishops | Bounded in $[-1.0, +1.0]$ ($\div 20$) |
| **Rook Activity ($\phi_{\text{rook}}$)** | Infiltration and occupation of open and advanced ranks | Bounded in $[-1.0, +1.0]$ ($\div 40$) |
| **Pawn Structure ($\phi_{\text{pawn}}$)** | Advanced pawn progression and connected pawn phalanxes | Bounded in $[-1.0, +1.0]$ ($\div 50$) |
| **Mobility ($\phi_{\text{mob}}$)** | Legal pseudo-move options available to active side | Bounded in $[-1.0, +1.0]$ ($\div 15$) |
| **King Safety ($\phi_{\text{king}}$)** | Direct check status and early king wandering penalties | Bounded in $[-1.0, +1.0]$ ($\div 100$) |
| **Bias Term ($1.0$)** | Constant offset for calibration of zero-sum equilibrium | Constant $1.0$ |

### 2. Action Space ($\mathcal{A}$)
The action space $\mathcal{A}(s)$ is discrete and context-dependent, comprising all legal chess moves $a \in \mathcal{M}(s)$ conforming to standard FIDE rules (including castling, en passant, and pawn promotions to $\{Q, R, B, N\}$). The branching factor satisfies $|\mathcal{A}(s)| \in [1, 218]$ with an average branching factor $\bar{b} \approx 35$.

### 3. Reward Function ($\mathcal{R}$) & Incentive Design
Terminal rewards reflect true game outcomes from White's perspective:
$$R_{\text{terminal}}(s) = \begin{cases} +1.0 & \text{if White wins (checkmate)}, \\ -1.0 & \text{if Black wins (checkmate)}, \\ 0.0 & \text{if Draw (stalemate, 50-move rule, threefold repetition)}. \end{cases}$$

To address the credit assignment challenge over long trajectories (40–90 plies), we incorporate potential-based reward shaping (Ng et al., 1999):
$$r_t = R_{\text{terminal}}(s_t) + c_{\text{mat}} \cdot \Delta \phi_{\text{mat}}(s_t)$$
where $c_{\text{mat}} = 0.05$. Potential-based shaping guarantees optimal policy invariance while accelerating TD error convergence.

### 4. Value Function Parameterization
The state value estimator $V(s; \mathbf{w})$ combines the baseline evaluation with learned residual weights:
$$\text{Score}(s) = \text{Base}(s) + S_c \sum_{i=1}^8 w_i \phi_i(s)$$
$$V(s; \mathbf{w}) = \tanh\left( \frac{\text{Score}(s)}{\sigma} \right) \in [-1.0, +1.0]$$
where $S_c = 450.0$ centipawns, $\sigma = 800.0$, and initial weights $\mathbf{w}_0 = \mathbf{0}$ reproduce the baseline evaluation prior exactly.

### 5. TD($\lambda$) Learning Updates
At each ply $t$, the temporal-difference error is computed as:
$$\delta_t = r_{t+1} + \gamma V(s_{t+1}; \mathbf{w}) - V(s_t; \mathbf{w})$$
Gradients are accumulated across the episode using decaying eligibility traces:
$$\mathbf{e}_t = \gamma \lambda \mathbf{e}_{t-1} + \nabla_{\mathbf{w}} V(s_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha_t \delta_t \mathbf{e}_t$$
with learning rate decay $\alpha_t = \frac{\alpha_0}{1 + 0.0005 t}$, trace decay $\lambda = 0.70$, discount $\gamma = 0.99$, and weight clipping to $[-2.0, +2.0]$.

### 6. Exploratory Stochasticity
To ensure exploration and prevent repetitive self-play loops, three stochastic mechanisms are applied:
1. **Randomized Opening Plies:** The first 6 half-moves are sampled randomly from legal candidates.
2. **Decaying $\epsilon$-Greedy:** Exploration rate decays from $\epsilon_0 = 0.20$ to $\epsilon_{\min} = 0.02$ ($\delta = 0.995$).
3. **Softmax Temperature Sampling:** Moves are sampled with probability $P(a) \propto \exp(Q(s, a) / \tau)$ where $\tau$ cools over episodes.

---

## 📂 Repository Directory Structure

```text
Chess_2.0/
├── ai_player_adaptive.py      # Baseline AIPlayer (Negamax Alpha-Beta with blitz clock)
├── baselines.py               # Tournament baselines (Random, Greedy, Untrained, Heuristic)
├── board.py                   # 8x8 Chess GameEngine wrapping python-chess
├── config.py                  # Chess dimensions, PSTs, piece values, RL hyperparameters
├── evaluate.py                # Tournament benchmarking module
├── features.py                # 8-dimensional normalized strategic feature extractor
├── generate_report.py         # ReportLab 2-page PDF compiler
├── plot_metrics.py            # Generates publication-quality figures
├── requirements.txt           # Project dependencies
├── rl_player.py               # RLAIPlayer integrating learned TD weights into search
├── td_learning.py             # TD(lambda) engine with eligibility traces & persistence
├── test_suite.py              # Complete unit and integration test suite
├── train.py                   # Self-play training pipeline with stochasticity
├── trained_weights.json       # Exported learned evaluation weights
├── checkpoints/               # Periodic model checkpoints (.json)
│   ├── best_model.json
│   └── checkpoint_episode_*.json
├── figures/                   # Generated evaluation and training figures
│   ├── figure1_td_error_convergence.png
│   ├── figure2_weight_trajectories.png
│   ├── figure3_game_lengths_and_outcomes.png
│   └── figure4_baseline_tournament.png
├── report/                    # LaTeX source files for Overleaf submission
│   └── report.tex
├── results/                   # Training logs and tournament results
│   ├── evaluation_results.json
│   ├── training_log.csv
│   └── training_metrics.json
└── RL_Chess_Engine_Report.pdf # Compiled 2-page Descriptive Report PDF
```

---

## ⚡ Quick Start & Reproduction

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/jeramarjun/RL_chess.git
cd RL_chess
pip install -r requirements.txt
```

### 2. Run Verification Tests
Verify that all game engine mechanics, feature extraction, TD updates, and Alpha-Beta search pass 100%:
```bash
python test_suite.py
```

### 3. Train the RL Agent
Train the agent via self-play for 500 episodes with exploratory stochasticity:
```bash
python train.py --episodes 500 --checkpoint-every 50 --random-opening-plies 6
```

### 4. Evaluate Tournament Performance vs Baselines
Benchmark the trained agent against Random, Greedy, Untrained, and Heuristic players across paired matches:
```bash
python evaluate.py --model trained_weights.json --games 10 --depth 3
```

### 5. Generate Publication Graphs
Render the 4 performance figures into `figures/`:
```bash
python plot_metrics.py
```

### 6. Compile 2-Page Descriptive Report PDF
Generate the official 2-page descriptive report PDF:
```bash
python generate_report.py
```

---

## 📊 Empirical Evaluation Results

In paired matches alternating White and Black under identical search depth and time controls, the trained RL engine achieved:

| Opponent Baseline | Games | Wins | Draws | Losses | Score % | Win Rate % | $\Delta$Elo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Baseline** | 8 | 8 | 0 | 0 | **100.0%** | 100.0% | **+798.3** |
| **Greedy Material** | 8 | 5 | 3 | 0 | **81.2%** | 62.5% | **+254.2** |
| **Untrained Prior ($\mathbf{w}=\mathbf{0}$)** | 8 | 4 | 4 | 0 | **75.0%** | 50.0% | **+190.8** |
| **Heuristic AIPlayer** | 8 | 2 | 5 | 1 | **56.2%** | 25.0% | **+43.3** |
| **Overall Total** | **32** | **19** | **12** | **1** | **78.1%** | **59.4%** | **+221.7** |

### Key Observations
1. **Dominant Tournament Record:** The RL agent suffered only 1 loss across 32 competitive baseline matches (78.1% overall score).
2. **Substantial Gain over Untrained Prior (+190.8 Elo):** Directly proves that the weights learned via TD($\lambda$) self-play deliver clear tactical and strategic superiority over the unlearned prior.
3. **Outperforming Hand-Tuned Heuristic AIPlayer (+43.3 Elo):** Autonomous TD learning discovered superior positional trade-offs (king defense, mobility, piece-square table centralization) that defeat the static baseline model.

---

## 👥 Authors & Academic Submission

- **Student:** Arjun (`jeramarjun`) — Group 34
- **Department:** Department of Computer Science & Engineering, Indian Institute of Technology Jodhpur (IITJ)
- **Course:** AI Course: Mini Project (Assignment 2: Real-Life Use Cases of RL)
- **Course Instructors / TAs:** Shubham Bagwari, Divyaansh Mertia, Aniruddha Bohra, Gajera Jahanvi Arvindkumar, Bhatt Vasishth Bimalbhai
- **GitHub Access:** If private repository access is needed, access is shared with: `shubhambagwari` and `divyaanshmertia`.
