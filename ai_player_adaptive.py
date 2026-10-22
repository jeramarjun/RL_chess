import math
import random
import time

from config import *


class AIPlayer:
    """
    Stronger non-RL chess player.

    Main ideas:
      * Correct side-to-move / root perspective.
      * Iterative-deepening negamax with alpha-beta pruning.
      * Quiescence search to avoid stopping in the middle of captures.
      * Transposition table with proper bounds.
      * Strong move ordering: captures, promotions, development, center,
        king safety, and history/killer moves.
      * Full root move set is searched; Monte Carlo is only a tie-breaker.
      * No learning / no RL / no neural network.
      * Every simulated/search move is always undone.
    """

    INF = 10**9
    MATE = 10**8

    def __init__(self, board):
        self.engine = board

        self.nodes_expanded = 0
        self.depth = 0

        self.start_time = 0.0
        self.time_limit = 0.5
        self.time_up = False

        # The assignment gives one shared clock to this player.
        self.total_time_budget = 58.0
        self.safety_reserve = 2.0
        self.max_moves = 75
        self.time_used = 0.0
        self.move_number = 0

        # Filled on the first call. This is critical: the search must
        # maximize from OUR perspective, whether we are White or Black.
        self.my_color = None

        self.transposition_table = {}
        self.history = {}
        self.killers = {}

        self.rng = random.Random(1729)

        # Small non-learning Monte Carlo tie-breaker.
        self.mcts_simulations = 16
        self.mcts_plies = 8

        # Search controls.
        self.max_depth = 8
        self.quiescence_depth = 5

    # ---------------------------------------------------------
    # BASIC HELPERS
    # ---------------------------------------------------------

    def _move_to_string(self, move):
        files = "abcdefgh"
        return (
            files[move.start_col]
            + str(8 - move.start_row)
            + files[move.end_col]
            + str(8 - move.end_row)
        )

    def _position_key(self):
        return (
            tuple(tuple(row) for row in self.engine.board),
            self.engine.white_to_move,
        )

    def _material_score(self):
        score = 0
        for row in self.engine.board:
            for piece in row:
                if piece != EMPTY_SQUARE:
                    score += PIECE_VALUES.get(piece, 0)
        return score

    def _is_ours(self, piece):
        return piece != EMPTY_SQUARE and piece[0] == self.my_color

    def _time_exceeded(self):
        if time.perf_counter() - self.start_time >= self.time_limit:
            self.time_up = True
            return True
        return False

    def _record_time_used(self):
        self.time_used += time.perf_counter() - self.start_time
        self.move_number += 1

    def _compute_time_budget(self, legal_moves):
        remaining = (
            self.total_time_budget
            - self.time_used
            - self.safety_reserve
        )

        # Emergency clock mode.  Once the reserve is nearly reached, do
        # not start an expensive search at all.
        if remaining <= 0.10:
            return max(0.005, remaining * 0.35)

        stage = self._game_phase(legal_moves)
        criticality = self._position_criticality(legal_moves)

        # Leave a progressively larger safety margin as the clock gets low.
        # The engine should lose on chess, not on the clock.
        if remaining < 1.0:
            reserve = min(0.30, remaining * 0.45)
        elif remaining < 3.0:
            reserve = min(0.55, remaining * 0.28)
        else:
            reserve = self.safety_reserve

        usable = max(0.05, remaining - reserve)

        # A move is normally cheap, but tactical/critical positions deserve
        # a larger slice.  Endgames also get a modest boost because each
        # pawn move can be decisive, while opening positions stay efficient.
        phase_factor = {
            "opening": 0.78,
            "middlegame": 1.00,
            "endgame": 0.90,
        }[stage]

        critical_factor = 0.72 + 0.85 * criticality
        moves_left = max(self.max_moves - self.move_number, 1)

        base = usable / moves_left
        slice_time = base * phase_factor * critical_factor

        # Hard per-move ceilings prevent one position from consuming the
        # entire remaining clock.  The ceiling relaxes only when there is
        # plenty of time left.
        if remaining < 1.0:
            ceiling = min(remaining * 0.22, 0.18)
        elif remaining < 3.0:
            ceiling = min(remaining * 0.30, 0.45)
        elif remaining < 8.0:
            ceiling = min(remaining * 0.34, 1.20)
        else:
            ceiling = min(remaining * 0.38, 4.0)

        return max(0.025, min(slice_time, ceiling))

    # ---------------------------------------------------------
    # GAME PHASE / POSITION CRITICALITY
    # ---------------------------------------------------------

    def _game_phase(self, legal_moves=None):
        """Classify the position using several independent board signals."""
        if legal_moves is None:
            legal_moves = self.engine.get_legal_moves()

        nonking = 0
        queens = 0
        rooks = 0
        minors = 0
        pawns = 0

        for row in self.engine.board:
            for piece in row:
                if piece == EMPTY_SQUARE:
                    continue
                kind = piece[1]
                if kind == "Q":
                    queens += 1
                elif kind == "R":
                    rooks += 1
                elif kind in ("N", "B"):
                    minors += 1
                elif kind == "P":
                    pawns += 1
                if kind != "K":
                    nonking += abs(PIECE_VALUES.get(piece, 0))

        # Opening: undeveloped armies and queens still on their home ranks.
        undeveloped = 0
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]
                if piece == EMPTY_SQUARE:
                    continue
                if piece[1] in ("N", "B") and (
                    (piece[0] == "w" and r == 7) or
                    (piece[0] == "b" and r == 0)
                ):
                    undeveloped += 1

        if nonking >= 430 and undeveloped >= 2 and pawns >= 10:
            return "opening"

        # Endgame is deliberately based on multiple parameters rather than
        # one arbitrary material cutoff.
        if (
            nonking <= 260
            or (queens == 0 and nonking <= 340)
            or (queens == 0 and rooks <= 2 and minors <= 2 and pawns <= 8)
        ):
            return "endgame"

        return "middlegame"

    def _position_criticality(self, legal_moves):
        """
        Return 0..1.  High means 'calculate carefully'; low means
        'make a sound move quickly'.  This affects time/depth, not the
        fundamental evaluation.
        """
        if not legal_moves:
            return 1.0

        score = 0.0
        captures = sum(
            1 for m in legal_moves if m.piece_captured != EMPTY_SQUARE
        )
        promotions = sum(
            1 for m in legal_moves if getattr(m, "promotion", None)
        )

        if self.engine.is_in_check():
            score += 0.50
        score += min(captures, 8) * 0.045
        score += min(promotions, 4) * 0.16

        # A narrow choice is often more critical than a huge quiet tree.
        if len(legal_moves) <= 4:
            score += 0.16
        elif len(legal_moves) >= 30:
            score -= 0.06

        # Strong material imbalance and low material are both common
        # decision points, especially when queens are still present.
        white_mat = black_mat = 0
        queens = 0
        for row in self.engine.board:
            for piece in row:
                if piece == EMPTY_SQUARE:
                    continue
                value = abs(PIECE_VALUES.get(piece, 0))
                if piece[0] == "w":
                    white_mat += value
                else:
                    black_mat += value
                if piece[1] == "Q":
                    queens += 1

        imbalance = abs(white_mat - black_mat)
        if imbalance >= 250:
            score += 0.10
        if queens and imbalance >= 150:
            score += 0.08

        # Being in check plus having forcing replies is particularly urgent.
        if self.engine.is_in_check() and captures:
            score += 0.14

        return max(0.0, min(1.0, score))

    def _adaptive_search_depth(self, stage, criticality, remaining):
        """Choose a practical depth cap instead of always trying depth 8."""
        if remaining < 0.30:
            return 1
        if remaining < 0.80:
            return 2
        if remaining < 2.0:
            return 3

        if stage == "opening":
            base = 4
        elif stage == "middlegame":
            base = 5
        else:
            base = 5

        # Critical positions can justify deeper calculation.
        extra = 2 if criticality >= 0.72 else (1 if criticality >= 0.42 else 0)

        if remaining >= 8.0:
            base += 1
        if remaining >= 16.0 and criticality >= 0.65:
            base += 1

        return min(self.max_depth, base + extra)

    # ---------------------------------------------------------
    # MAIN SEARCH
    # ---------------------------------------------------------

    def get_best_move(self):
        self.start_time = time.perf_counter()
        self.time_up = False
        self.nodes_expanded = 0
        self.depth = 0

        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            self._record_time_used()
            return None

        if self.my_color is None:
            self.my_color = "w" if self.engine.white_to_move else "b"

        remaining_before = (
            self.total_time_budget - self.time_used - self.safety_reserve
        )

        # Absolute emergency mode: no MCTS, no iterative deepening, no
        # expensive capture probes.  Just return a legal, ordered move.
        if remaining_before <= 0.10:
            best = self._fast_fallback(legal_moves)
            self._record_time_used()
            return best

        stage = self._game_phase(legal_moves)
        criticality = self._position_criticality(legal_moves)
        self.time_limit = self._compute_time_budget(legal_moves)

        # Always establish a legal fallback first.
        fallback = self._fast_fallback(legal_moves)

        # MCTS is useful only as a tiny secondary signal in positions where
        # there is enough clock.  It is completely disabled when time is low.
        mcts_scores = {}
        if self.time_limit >= 0.30 and remaining_before >= 2.0:
            mcts_scores = self._monte_carlo_screen(
                legal_moves,
                budget_scale=min(1.0, criticality + 0.25)
            )

        if self.time_up:
            self._record_time_used()
            return fallback

        ordered = self._order_root_moves(legal_moves, mcts_scores)
        best_move = fallback
        best_score = -self.INF

        search_depth = self._adaptive_search_depth(
            stage, criticality, remaining_before
        )

        for depth in range(1, search_depth + 1):
            if self._time_exceeded():
                break

            self.depth = depth
            self.killers.setdefault(depth, [])

            alpha = -self.INF
            beta = self.INF
            current_best = None
            current_score = -self.INF
            completed = True

            if best_move in ordered:
                ordered.remove(best_move)
                ordered.insert(0, best_move)

            for move in ordered:
                if self._time_exceeded():
                    completed = False
                    break

                self.engine.make_move(move)
                try:
                    score = self._negamax(
                        depth - 1,
                        -beta,
                        -alpha,
                        1
                    )
                    score = -score
                finally:
                    self.engine.undo_move()

                if self.time_up:
                    completed = False
                    break

                if score > current_score:
                    current_score = score
                    current_best = move

                if score > alpha:
                    alpha = score

            if completed and current_best is not None:
                best_move = current_best
                best_score = current_score

                ordered.remove(best_move)
                ordered.insert(0, best_move)

                if len(legal_moves) == 1:
                    break

                # In a quiet position, stop once the allocated depth is
                # sufficient.  In a critical position, keep deepening.
                if (
                    criticality < 0.25
                    and depth >= 4
                    and self.time_limit < 1.0
                ):
                    break

        self._record_time_used()
        return best_move

    # ---------------------------------------------------------
    # NEGAMAX + ALPHA-BETA
    # ---------------------------------------------------------

    def _negamax(self, depth, alpha, beta, ply):
        self.nodes_expanded += 1

        if self._time_exceeded():
            return 0

        # Evaluation is always converted to the side-to-move perspective.
        # This removes the catastrophic White/Black root asymmetry in the
        # previous implementation.
        key = self._position_key()
        alpha_orig = alpha
        beta_orig = beta

        cached = self.transposition_table.get(key)
        if cached is not None:
            stored_depth, stored_value, flag = cached
            if stored_depth >= depth:
                if flag == "EXACT":
                    return stored_value
                if flag == "LOWER":
                    alpha = max(alpha, stored_value)
                elif flag == "UPPER":
                    beta = min(beta, stored_value)
                if alpha >= beta:
                    return stored_value

        legal = self.engine.get_legal_moves()

        if not legal:
            if self.engine.is_in_check():
                # Side to move is checkmated.
                return -self.MATE + ply
            return 0

        if depth <= 0:
            return self._quiescence(alpha, beta, 0)

        ordered = self._order_moves(legal, ply)

        best = -self.INF
        best_move = None

        for move in ordered:
            if self._time_exceeded():
                return 0

            self.engine.make_move(move)
            try:
                value = -self._negamax(
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1
                )
            finally:
                self.engine.undo_move()

            if self.time_up:
                return 0

            if value > best:
                best = value
                best_move = move

            if value > alpha:
                alpha = value

            if alpha >= beta:
                # Killer/history updates only happen on genuine cutoffs.
                self._record_cutoff(move, depth, ply)
                break

        if best == -self.INF:
            return 0

        if best <= alpha_orig:
            flag = "UPPER"
        elif best >= beta_orig:
            flag = "LOWER"
        else:
            flag = "EXACT"

        self.transposition_table[key] = (
            depth,
            best,
            flag
        )

        return best

    # ---------------------------------------------------------
    # QUIESCENCE
    # ---------------------------------------------------------

    def _quiescence(self, alpha, beta, qply):
        self.nodes_expanded += 1

        if self._time_exceeded():
            return self._evaluate_for_side()

        legal = self.engine.get_legal_moves()

        if not legal:
            if self.engine.is_in_check():
                return -self.MATE + qply
            return 0

        stand_pat = self._evaluate_for_side()

        # When the clock is low, cap quiescence earlier.  Check evasions
        # above still remain fully legal and are searched before this point.
        q_limit = self.quiescence_depth
        if self.time_limit < 0.25:
            q_limit = 2
        elif self.time_limit < 0.50:
            q_limit = 3

        if qply >= q_limit:
            return stand_pat

        # If in check, ALL legal evasions matter. Do not use stand-pat.
        if self.engine.is_in_check():
            best = -self.INF
            for move in self._order_moves(legal, qply):
                if self._time_exceeded():
                    return 0

                self.engine.make_move(move)
                try:
                    value = -self._quiescence(
                        -beta,
                        -alpha,
                        qply + 1
                    )
                finally:
                    self.engine.undo_move()

                if self.time_up:
                    return 0

                best = max(best, value)
                alpha = max(alpha, value)

                if alpha >= beta:
                    break

            return best

        # At a quiet leaf, only examine forcing captures.
        captures = [
            m for m in legal
            if m.piece_captured != EMPTY_SQUARE
        ]

        if not captures:
            return stand_pat

        best = stand_pat

        if best >= beta:
            return best

        alpha = max(alpha, best)

        for move in self._order_moves(captures, qply):
            if self._time_exceeded():
                return best

            # Delta pruning: if the captured piece is tiny and the current
            # position is already far above beta, this capture cannot help.
            captured_value = abs(
                PIECE_VALUES.get(move.piece_captured, 0)
            )

            if qply >= 2 and stand_pat + captured_value + 80 < alpha:
                continue

            self.engine.make_move(move)
            try:
                value = -self._quiescence(
                    -beta,
                    -alpha,
                    qply + 1
                )
            finally:
                self.engine.undo_move()

            if self.time_up:
                return best

            best = max(best, value)
            alpha = max(alpha, value)

            if alpha >= beta:
                break

        return best

    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------

    def _evaluate_for_side(self):
        """Static evaluation from the CURRENT side-to-move perspective."""
        white_score = self.evaluate_board()

        return (
            white_score
            if self.engine.white_to_move
            else -white_score
        )

    def evaluate_board(self):
        """
        White-positive evaluation.

        Material remains dominant, while positional terms reward:
          * development
          * center control
          * mobility
          * connected/advanced pawns
          * rook activity
          * king safety
          * piece-square tables

        No learned weights are used.
        """
        score = self._material_score()

        white_material = 0
        black_material = 0

        white_minor_activity = 0.0
        black_minor_activity = 0.0

        white_mobility = 0
        black_mobility = 0

        white_pawns = []
        black_pawns = []

        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]

                if piece == EMPTY_SQUARE:
                    continue

                side = piece[0]
                kind = piece[1]
                value = abs(PIECE_VALUES.get(piece, 0))

                if side == "w":
                    white_material += value
                else:
                    black_material += value

                # Centrality: files d/e and ranks 4/5 are valuable.
                center = 3.5 - (
                    abs(c - 3.5) + abs(r - 3.5)
                ) * 0.5

                if kind == "N":
                    if side == "w":
                        white_minor_activity += max(0.0, center)
                    else:
                        black_minor_activity += max(0.0, center)

                elif kind == "B":
                    if side == "w":
                        white_minor_activity += max(0.0, center) * 0.8
                    else:
                        black_minor_activity += max(0.0, center) * 0.8

                elif kind == "R":
                    # Rooks become better as they move toward open/active
                    # ranks rather than sitting on the original rank.
                    if side == "w":
                        score += max(0, 7 - r) * 0.7
                    else:
                        score -= max(0, r) * 0.7

                elif kind == "P":
                    if side == "w":
                        white_pawns.append((r, c))
                    else:
                        black_pawns.append((r, c))

        # Material imbalance already has the strongest signal.
        score += 0.0 * (white_material - black_material)

        # -----------------------------------------------------
        # Development / piece-square tables
        # -----------------------------------------------------

        nonking = 0
        for row in self.engine.board:
            for piece in row:
                if piece != EMPTY_SQUARE and piece[1] != "K":
                    nonking += abs(PIECE_VALUES.get(piece, 0))

        is_endgame = nonking <= 260

        pst = 0.0

        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]
                if piece == EMPTY_SQUARE:
                    continue

                kind = piece[1]
                white = piece[0] == "w"

                # For this engine's board, Black needs the vertical mirror.
                table_row = r if white else BOARD_HEIGHT - 1 - r

                value = 0

                if kind == "P":
                    value = PAWN_PST[table_row][c]
                elif kind == "N":
                    value = KNIGHT_PST[table_row][c]
                elif kind == "B":
                    value = BISHOP_PST[table_row][c]
                elif kind == "R":
                    value = ROOK_PST[table_row][c]
                elif kind == "K" and is_endgame:
                    value = KING_PST_LATE_GAME[table_row][c]

                pst += value if white else -value

        score += pst * 0.16

        # -----------------------------------------------------
        # Mobility
        # -----------------------------------------------------
        #
        # Raw pseudo-legal mobility is much cheaper than get_legal_moves()
        # and is used only as a positional signal.

        try:
            current_mobility = len(
                self.engine._get_all_possible_moves()
            )
        except Exception:
            current_mobility = 0

        if self.engine.white_to_move:
            white_mobility = current_mobility
        else:
            black_mobility = current_mobility

        # We cannot cheaply obtain both sides without disturbing the board,
        # so use current-side mobility as a modest tiebreaker.
        if self.engine.white_to_move:
            score += min(white_mobility, 30) * 0.45
        else:
            score -= min(black_mobility, 30) * 0.45

        # -----------------------------------------------------
        # Center occupation / pawn structure
        # -----------------------------------------------------

        for r, c in white_pawns:
            if c in (3, 4):
                score += 7
            if r <= 4:
                score += (4 - r) * 1.2

        for r, c in black_pawns:
            if c in (3, 4):
                score -= 7
            if r >= 3:
                score -= (r - 3) * 1.2

        # Connected pawns are a useful structural bonus.
        white_pawn_cols = {c for _, c in white_pawns}
        black_pawn_cols = {c for _, c in black_pawns}

        for c in white_pawn_cols:
            if c - 1 in white_pawn_cols or c + 1 in white_pawn_cols:
                score += 2.5

        for c in black_pawn_cols:
            if c - 1 in black_pawn_cols or c + 1 in black_pawn_cols:
                score -= 2.5

        # -----------------------------------------------------
        # King safety / checks
        # -----------------------------------------------------

        if self.engine.is_in_check():
            if self.engine.white_to_move:
                score -= 90
            else:
                score += 90

        # Don't reward early king wandering.
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]
                if piece == "wK" and r < 5:
                    score -= 12
                elif piece == "bK" and r > 2:
                    score += 12

        return score

    # ---------------------------------------------------------
    # MOVE ORDERING
    # ---------------------------------------------------------

    def _order_root_moves(self, moves, mcts_scores):
        def score(move):
            s = self._move_order_score(move, 0)

            # Small root-only tactical refinement:
            # prefer captures that win a loose piece without allowing an
            # immediate recapture.  This does NOT change the evaluation
            # function or the search tree; it only helps the search see
            # obvious free material early.
            if move.piece_captured != EMPTY_SQUARE:
                s += self._root_capture_quality(move)

            key = self._move_to_string(move)
            s += mcts_scores.get(key, 0.0) * 0.02

            return s

        return sorted(moves, key=score, reverse=True)

    def _root_capture_quality(self, move):
        """
        Root-only tactical ordering for easy captures.

        The existing MVV-LVA ordering already puts captures first.  This
        extra term distinguishes a genuinely free capture from a capture
        that immediately hangs the capturing piece.

        It is intentionally used only for move ordering, not evaluation, so
        the engine's existing positional/search behavior is left intact.
        """
        victim = abs(PIECE_VALUES.get(move.piece_captured, 0))
        attacker = abs(PIECE_VALUES.get(move.piece_moved, 0))

        if victim <= 0:
            return 0.0

        try:
            self.engine.make_move(move)
            try:
                replies = self.engine.get_legal_moves()

                # Can the opponent immediately take the piece that just
                # captured?  Only exact destination captures matter here.
                recapture_attackers = [
                    reply for reply in replies
                    if (
                        reply.piece_captured != EMPTY_SQUARE
                        and reply.end_row == move.end_row
                        and reply.end_col == move.end_col
                    )
                ]
            finally:
                self.engine.undo_move()
        except Exception:
            # Never let this refinement interfere with the main search.
            return 0.0

        if not recapture_attackers:
            # Clearly free material: make this noticeably easier to find.
            return 1800.0 + min(victim, 900) * 1.5

        # If the capture still wins material after the forced exchange,
        # give it a smaller preference.  Avoid making speculative trades
        # dominate positional moves.
        best_recapture = min(
            abs(PIECE_VALUES.get(reply.piece_moved, 0))
            for reply in recapture_attackers
        )
        net_gain = victim - attacker

        if net_gain > 0:
            return 250.0 + min(net_gain, 500) * 0.8

        return 0.0

    def _order_moves(self, moves, ply):
        return sorted(
            moves,
            key=lambda m: self._move_order_score(m, ply),
            reverse=True
        )

    def _move_order_score(self, move, ply):
        score = 0.0

        key = self._move_to_string(move)

        # TT/PV move is not stored separately, but history gives stable
        # preference to moves that repeatedly cause cutoffs.
        score += self.history.get(key, 0)

        # Killer moves.
        killers = self.killers.get(ply, [])
        if key in killers:
            score += 5000 - killers.index(key) * 500

        # Captures: MVV-LVA.
        if move.piece_captured != EMPTY_SQUARE:
            victim = abs(PIECE_VALUES.get(move.piece_captured, 0))
            attacker = abs(PIECE_VALUES.get(move.piece_moved, 0))
            score += 100000 + victim * 100 - attacker * 2

        kind = move.piece_moved[1]

        # Promotions if the engine exposes promotion info through the move.
        promotion = getattr(move, "promotion", None)
        if promotion:
            score += 50000

        # Development.
        if kind == "N":
            score += 120
        elif kind == "B":
            score += 110
        elif kind == "R":
            score += 45
        elif kind == "Q":
            score += 35
        elif kind == "P":
            score += 12
        elif kind == "K":
            score -= 60

        # Center.
        center_distance = (
            abs(move.end_row - 3.5)
            + abs(move.end_col - 3.5)
        )
        score += max(0, 8 - center_distance * 2)

        # Pawn moves toward the center get a modest bonus.
        if kind == "P" and move.end_col in (3, 4):
            score += 14

        # Immediate A-B-A reversal is strongly discouraged unless it is
        # a capture (captures are already overwhelmingly important).
        if self._is_immediate_reversal(move) and (
            move.piece_captured == EMPTY_SQUARE
        ):
            score -= 2500

        return score

    def _record_cutoff(self, move, depth, ply):
        key = self._move_to_string(move)

        old = self.history.get(key, 0)
        self.history[key] = min(
            old + depth * depth,
            200000
        )

        killers = self.killers.setdefault(ply, [])

        if key not in killers:
            killers.insert(0, key)

        if len(killers) > 2:
            del killers[2:]

    def _fast_fallback(self, moves):
        """Very cheap legal fallback for low-clock positions."""
        ordered = self._order_moves(moves, 0)

        # In a real time scramble, avoid any board-mutation probes.
        best = ordered[0]
        best_key = -self.INF

        for move in ordered[:min(8, len(ordered))]:
            value = self._cheap_move_score(move)
            if value > best_key:
                best_key = value
                best = move

        return best

    def _cheap_move_score(self, move):
        score = self._move_order_score(move, 0)

        if move.piece_captured != EMPTY_SQUARE:
            victim = abs(PIECE_VALUES.get(move.piece_captured, 0))
            attacker = abs(PIECE_VALUES.get(move.piece_moved, 0))
            score += victim * 80 - attacker * 8

        # In emergencies, prefer safe, forcing moves without making another
        # make/undo cycle just to test them.
        if self.engine.is_in_check():
            score += 10000

        return score

    # ---------------------------------------------------------
    # STATIC FALLBACK
    # ---------------------------------------------------------

    def _static_best_move(self, moves):
        ordered = self._order_moves(moves, 0)

        best = ordered[0]
        best_value = -self.INF

        for move in ordered[:min(len(ordered), 24)]:
            if self._time_exceeded():
                break

            self.engine.make_move(move)
            try:
                value = self._evaluate_for_side()
                value = -value
            finally:
                self.engine.undo_move()

            if self.time_up:
                break

            if value > best_value:
                best_value = value
                best = move

        return best

    # ---------------------------------------------------------
    # MONTE CARLO
    # ---------------------------------------------------------
    #
    # This is intentionally NOT used as a replacement for minimax.
    # The previous version gave this tiny noisy screen too much authority,
    # which could discard tactically correct moves before alpha-beta saw
    # them. Here it only breaks close ties at the root.

    def _monte_carlo_screen(self, legal_moves, budget_scale=1.0):
        scores = {
            self._move_to_string(m): 0.0
            for m in legal_moves
        }

        visits = {
            self._move_to_string(m): 0
            for m in legal_moves
        }

        if len(legal_moves) <= 1:
            return scores

        # MCTS is only a tie-breaker.  Keep it proportional to the
        # position's criticality so quiet positions do not pay for it.
        budget = min(
            0.06,
            max(0.01, self.time_limit * 0.07 * budget_scale)
        )

        deadline = self.start_time + budget
        ordered = self._order_moves(legal_moves, 0)

        sim = 0

        while (
            sim < self.mcts_simulations
            and time.perf_counter() < deadline
            and not self.time_up
        ):
            move = ordered[sim % len(ordered)]
            key = self._move_to_string(move)

            self.engine.make_move(move)
            try:
                result = self._rollout_value()
            finally:
                self.engine.undo_move()

            scores[key] += result
            visits[key] += 1
            sim += 1

        for key in scores:
            if visits[key]:
                scores[key] /= visits[key]

        return scores

    def _rollout_value(self):
        """
        Short deterministic-ish heuristic rollout.

        It is a simulation only: no training, no parameter updates, and
        every simulated move is undone before returning.
        """
        start_score = self._evaluate_for_side()
        played = []

        try:
            for _ in range(self.mcts_plies):
                if self._time_exceeded():
                    break

                legal = self.engine.get_legal_moves()

                if not legal:
                    if self.engine.is_in_check():
                        return -self.MATE
                    return 0

                move = self._rollout_move(legal)

                self.engine.make_move(move)
                played.append(move)

            end_score = self._evaluate_for_side()

            # Convert the result back to the perspective that existed
            # before the rollout started.
            return start_score - end_score

        finally:
            while played:
                self.engine.undo_move()
                played.pop()

    def _rollout_move(self, moves):
        # Prefer forcing captures, but choose among several strong options
        # rather than blindly taking the first move.
        ordered = self._order_moves(moves, 0)

        top = ordered[:min(5, len(ordered))]

        if top:
            weights = []
            for move in top:
                w = 1.0

                if move.piece_captured != EMPTY_SQUARE:
                    w += 6.0

                if move.piece_moved[1] in ("N", "B"):
                    w += 1.5

                weights.append(w)

            return self.rng.choices(top, weights=weights, k=1)[0]

        return self.rng.choice(moves)

    # ---------------------------------------------------------
    # REPETITION / SHUFFLING
    # ---------------------------------------------------------

    def _is_immediate_reversal(self, move):
        if not self.engine.move_log:
            return False

        previous = self.engine.move_log[-1]

        return (
            move.start_row == previous.end_row
            and move.start_col == previous.end_col
            and move.end_row == previous.start_row
            and move.end_col == previous.start_col
        )
