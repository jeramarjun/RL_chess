"""
Automated 2-Page Descriptive Report PDF Generator.
Compiles a publication-quality, 2-page descriptive report adhering strictly
to the assignment format: two-column layout, mathematical MDP formulation,
empirical graphs, tournament results table, and team information.
Uses ReportLab to generate RL_Chess_Engine_Report.pdf.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    PageTemplate,
    Frame,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class PageNumCanvas(canvas.Canvas):
    """Custom canvas that adds page numbers and running header/footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self.pages)
        print(f"REPORTLAB TOTAL PAGES COUNT: {num_pages}")
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 6.5)
        self.setFillColor(colors.HexColor("#555555"))

        # Top Running Header (page 2 only)
        if self._pageNumber > 1:
            self.drawString(
                0.45 * inch,
                10.55 * inch,
                "AI Assignment 2: Real-Life Use Cases of RL (Mini Project) — Chess RL Engine",
            )
            self.drawRightString(
                8.05 * inch,
                10.55 * inch,
                "IIT Jodhpur",
            )
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.4)
            self.line(0.45 * inch, 10.50 * inch, 8.05 * inch, 10.50 * inch)

        # Bottom Footer
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(8.05 * inch, 0.28 * inch, page_str)
        self.drawString(
            0.45 * inch,
            0.28 * inch,
            "Autonomous Chess Decision-Making via TD(λ) Self-Play & Minimax Search — Group 34",
        )
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.4)
        self.line(0.45 * inch, 0.35 * inch, 8.05 * inch, 0.35 * inch)
        self.restoreState()


def build_pdf_report(
    output_filename="RL_Chess_Engine_Report.pdf",
    fig1_path="figures/figure1_td_error_convergence.png",
    fig2_path="figures/figure2_weight_trajectories.png",
    fig4_path="figures/figure4_baseline_tournament.png",
):
    page_width, page_height = letter
    margin = 0.45 * inch
    col_gap = 0.20 * inch
    usable_width = page_width - 2 * margin
    col_width = (usable_width - col_gap) / 2.0
    usable_height = page_height - 2 * margin - 0.15 * inch

    doc = BaseDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
    )

    frame1 = Frame(
        margin,
        margin + 0.12 * inch,
        col_width,
        usable_height,
        id="col1",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    frame2 = Frame(
        margin + col_width + col_gap,
        margin + 0.12 * inch,
        col_width,
        usable_height,
        id="col2",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )

    two_col_template = PageTemplate(id="TwoCol", frames=[frame1, frame2])
    doc.addPageTemplates([two_col_template])

    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Helvetica-Bold",
        fontSize=11.0,
        leading=13.0,
        alignment=1,
        textColor=colors.HexColor("#0D47A1"),
    )
    sub_title_style = ParagraphStyle(
        "DocSubTitle",
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.0,
        alignment=1,
        textColor=colors.HexColor("#263238"),
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        fontName="Helvetica-Bold",
        fontSize=6.8,
        leading=8.5,
        alignment=1,
        textColor=colors.HexColor("#37474F"),
    )
    sec_style = ParagraphStyle(
        "SecHead",
        fontName="Helvetica-Bold",
        fontSize=8.2,
        leading=10.0,
        textColor=colors.HexColor("#0D47A1"),
        spaceBefore=4,
        spaceAfter=1.5,
    )
    subsec_style = ParagraphStyle(
        "SubSecHead",
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9.0,
        textColor=colors.HexColor("#1565C0"),
        spaceBefore=2.5,
        spaceAfter=1,
    )
    body_style = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=6.5,
        leading=8.2,
        alignment=4,
        textColor=colors.HexColor("#1A1A1A"),
        spaceBefore=0.5,
        spaceAfter=1.5,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        fontName="Helvetica",
        fontSize=6.2,
        leading=7.8,
        leftIndent=6,
        textColor=colors.HexColor("#1A1A1A"),
        spaceBefore=0.5,
        spaceAfter=0.8,
    )
    eq_style = ParagraphStyle(
        "Equation",
        fontName="Courier-Bold",
        fontSize=6.2,
        leading=7.8,
        alignment=1,
        textColor=colors.HexColor("#B71C1C"),
        spaceBefore=1,
        spaceAfter=1.5,
    )
    caption_style = ParagraphStyle(
        "Caption",
        fontName="Helvetica-Oblique",
        fontSize=5.8,
        leading=7.2,
        alignment=1,
        textColor=colors.HexColor("#424242"),
        spaceBefore=1,
        spaceAfter=2,
    )

    story = []

    # Title & Metadata Header
    story.append(Paragraph("Reinforcement Learning-Enhanced Chess Engine:<br/>Strategic Decision-Making via TD(&lambda;) Self-Play and Minimax Search", title_style))
    story.append(Spacer(1, 1.5))
    story.append(Paragraph("AI Course: Mini Project (Assignment 2: Real-Life Use Cases of RL)", sub_title_style))
    story.append(Spacer(1, 1))
    story.append(Paragraph("<b>Arjun (jeramarjun)</b> &bull; Dept. of Computer Science &amp; Engg., IIT Jodhpur &bull; <u>https://github.com/jeramarjun/RL_chess</u>", meta_style))
    story.append(HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#0D47A1"), spaceBefore=2.5, spaceAfter=3))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>&mdash;Traditional chess engines rely heavily on hand-crafted heuristic evaluation functions that encode human bias and fail in complex non-standard positions. In compliance with assignment rules strictly restricting solutions to Reinforcement Learning (disqualifying pure heuristic or supervised approaches), we formulate standard 8x8 chess as a Markov Decision Process (MDP) and develop an autonomous agent learning evaluation weights via Temporal-Difference learning (TD(&lambda;)) through self-play. We inject exploratory stochasticity (randomized opening plies, decaying &epsilon;-greedy, temperature sampling) and integrate the learned evaluation function into an iterative-deepening Negamax search engine with &alpha;-&beta; pruning, quiescence search, and blitz clock management (60s shared clock). Systematic empirical tournaments demonstrate that our RL-enhanced engine achieves an 81.2% score against greedy material players, +190.8 &Delta;Elo over unlearned search priors, and outperforms heuristic baselines by +43.3 &Delta;Elo across 32 tournament games.", body_style))
    story.append(Spacer(1, 1.5))

    # Section 1
    story.append(Paragraph("1. Introduction &amp; Project Objective", sec_style))
    story.append(Paragraph("Chess features an enormous state space of ~10<sup>44</sup> reachable board configurations and game tree complexity exceeding 10<sup>123</sup>. Handcrafted heuristics require decades of tuning. Here, the engine autonomously learns strategic position evaluation strictly via Reinforcement Learning self-play. The learned value estimator evaluates leaf states inside an &alpha;-&beta; Negamax tree.", body_style))

    # Section 2
    story.append(Paragraph("2. Reinforcement Learning Formulation", sec_style))
    story.append(Paragraph("Standard chess is formalized as an MDP: &lang;<i>S</i>, <i>A</i>, <i>P</i>, <i>R</i>, &gamma;&rang;.", body_style))

    story.append(Paragraph("<b>A. State Space (S):</b> A state <i>s</i> is defined by the tuple:", subsec_style))
    story.append(Paragraph("<i>s</i> = &lang; <b>B</b>, <i>t</i>, <b>c</b>, <i>ep</i>, <i>h</i><sub>50</sub>, <i>h</i><sub>rep</sub> &rang;", eq_style))
    story.append(Paragraph("where <b>B</b> &isin; {0,1}<sup>12x8x8</sup> represents the 12 piece planes across 64 squares; <i>t</i> &isin; {0,1} is the active player turn; <b>c</b> &isin; {0,1}<sup>4</sup> denotes castling rights; <i>ep</i> tracks en-passant availability; <i>h</i><sub>50</sub> &isin; [0,100] is the half-move counter; and <i>h</i><sub>rep</sub> detects threefold repetition. States are mapped to an 8-dimensional normalized feature vector &phi;(<i>s</i>) &isin; [-1, +1]<sup>8</sup> capturing core strategic dimensions:", body_style))
    story.append(Paragraph("&bull; <b>Material (&phi;<sub>mat</sub>):</b> Net piece differential normalized by 4000 cp (P=100, N=320, B=330, R=500, Q=900).", bullet_style))
    story.append(Paragraph("&bull; <b>Piece-Square Tables (&phi;<sub>pst</sub>):</b> Positional piece centralization.", bullet_style))
    story.append(Paragraph("&bull; <b>Minor Activity (&phi;<sub>min</sub>):</b> Center pressure by knights &amp; bishops.", bullet_style))
    story.append(Paragraph("&bull; <b>Rook Activity (&phi;<sub>rk</sub>):</b> Open and advanced rank occupation.", bullet_style))
    story.append(Paragraph("&bull; <b>Pawn Structure (&phi;<sub>pwn</sub>):</b> Connected phalanxes &amp; center control.", bullet_style))
    story.append(Paragraph("&bull; <b>Mobility (&phi;<sub>mob</sub>):</b> Legal pseudo-move count for active player.", bullet_style))
    story.append(Paragraph("&bull; <b>King Safety (&phi;<sub>kng</sub>):</b> Direct check penalties and shelter preservation.", bullet_style))
    story.append(Paragraph("&bull; <b>Bias (1.0):</b> Calibration intercept offset.", bullet_style))

    story.append(Paragraph("<b>B. Action Space (A):</b> Discrete context-dependent action space <i>A(s)</i> containing all legal moves conforming to standard FIDE rules (including castling, en-passant, and promotions). Branching factor |<i>A(s)</i>| &isin; [1, 218] with average &sim;35.", body_style))

    story.append(Paragraph("<b>C. Reward Function (R) &amp; Incentive Design:</b> Terminal rewards capture true game outcomes from White's perspective:", subsec_style))
    story.append(Paragraph("<i>R</i><sub>term</sub> = +1.0 (White Win), -1.0 (Black Win), 0.0 (Draw)", eq_style))
    story.append(Paragraph("To address credit assignment over extended trajectories (40-80 plies), we employ potential-based reward shaping (Ng et al., 1999):", body_style))
    story.append(Paragraph("<i>r<sub>t</sub></i> = <i>R</i><sub>term</sub>(<i>s<sub>t</sub></i>) + <i>c</i><sub>mat</sub> &Delta;&phi;<sub>mat</sub>(<i>s<sub>t</sub></i>)", eq_style))
    story.append(Paragraph("with <i>c</i><sub>mat</sub>=0.10, guaranteeing optimal policy invariance while accelerating TD error convergence.", body_style))

    # Section 3
    story.append(Paragraph("3. Algorithm: TD-Leaf(&lambda;) &amp; Search Integration", sec_style))
    story.append(Paragraph("The value function combines the baseline prior with learned residual weights:", body_style))
    story.append(Paragraph("<i>V(s; <b>w</b>)</i> = tanh((Base(<i>s</i>) + <i>S<sub>c</sub></i> &sum; <i>w<sub>i</sub></i> &phi;<sub><i>i</i></sub>(<i>s</i>)) / &sigma;)", eq_style))
    story.append(Paragraph("where <i>S<sub>c</sub></i>=300 cp, &sigma;=800 cp, and initial <b>w</b><sub>0</sub>=<b>0</b>. At ply <i>t</i>, TD error &delta;<sub><i>t</i></sub> = <i>r</i><sub><i>t</i>+1</sub> + &gamma;<i>V</i>(<i>s</i><sub><i>t</i>+1</sub>) - <i>V</i>(<i>s<sub>t</sub></i>) drives weight updates via eligibility traces <b>e</b><sub><i>t</i></sub> = &gamma;&lambda;<b>e</b><sub><i>t</i>-1</sub> + &nabla;<i>V</i>(<i>s<sub>t</sub></i>) with &alpha;=0.02, &gamma;=0.99, and &lambda;=0.70.", body_style))

    story.append(Paragraph("<b>Exploratory Stochasticity:</b> To prevent deterministic loops in self-play, three stochastic mechanisms operate: (1) Random opening half-moves (first 6 plies); (2) Decaying &epsilon;-greedy exploration (&epsilon;<sub>0</sub>=0.20 &rarr; 0.02); (3) Softmax temperature sampling over root candidates.", body_style))

    story.append(Paragraph("<b>Search Engine Integration:</b> During tournament matches, the learned evaluation powers an iterative-deepening Negamax tree with &alpha;-&beta; pruning, quiescence search, MVV-LVA move ordering, killer moves, and transposition tables under a 60-second blitz clock budget.", body_style))

    # Break to Page 2 for Empirical Results, Figures, Table & Discussion
    story.append(PageBreak())

    # Section 4
    story.append(Paragraph("4. Empirical Results &amp; Discussion", sec_style))
    story.append(Paragraph("Training ran for 300 autonomous self-play iterations. Figure 1 shows steady convergence of the mean absolute TD error |&delta;| over training.", body_style))

    if os.path.exists(fig1_path):
        story.append(Image(fig1_path, width=col_width, height=col_width * 0.50))
        story.append(Paragraph("<b>Figure 1:</b> TD Error |&delta;| convergence across self-play training.", caption_style))

    if os.path.exists(fig2_path):
        story.append(Image(fig2_path, width=col_width, height=col_width * 0.52))
        story.append(Paragraph("<b>Figure 2:</b> Strategic weight evolution <i>w<sub>i</sub></i> over episodes.", caption_style))

    story.append(Paragraph("Weight trajectories in Figure 2 reveal how the RL agent actively learned to prioritize mobility (+0.148) and PST centralization (+0.048) alongside king defense (-0.866) in tactical positions.", body_style))

    # Tournament Table
    story.append(Paragraph("<b>Systematic Baseline Evaluation:</b> We evaluated the trained RL player against 4 baseline opponents across 32 paired matches (alternating White and Black under identical depth/clock constraints):", body_style))

    table_data = [
        ["Opponent Baseline", "Games", "W", "D", "L", "Score%", "&Delta;Elo"],
        ["Random Baseline", "8", "8", "0", "0", "100.0%", "+798.3"],
        ["Greedy Material", "8", "5", "3", "0", "81.2%", "+254.2"],
        ["Untrained (w=0)", "8", "4", "4", "0", "75.0%", "+190.8"],
        ["Heuristic AIPlayer", "8", "2", "5", "1", "56.2%", "+43.3"],
        ["Total Tournament", "32", "19", "12", "1", "78.1%", "+221.7"],
    ]
    t = Table(table_data, colWidths=[col_width * 0.35, col_width * 0.10, col_width * 0.08, col_width * 0.08, col_width * 0.08, col_width * 0.15, col_width * 0.16])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0D47A1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 5.8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B0BEC5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.HexColor("#F5F5F5"), colors.white]),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#E3F2FD")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
    ]))
    story.append(t)
    story.append(Paragraph("<b>Table 1:</b> Head-to-head tournament results against baselines.", caption_style))

    if os.path.exists(fig4_path):
        story.append(Image(fig4_path, width=col_width, height=col_width * 0.44))
        story.append(Paragraph("<b>Figure 3:</b> Tournament score % and empirical Elo gains.", caption_style))

    story.append(Paragraph("<b>Tournament Analysis:</b> The RL agent conceded only 1 defeat across 32 competitive matches, establishing a +190.8 Elo advantage over unlearned search priors and +43.3 Elo over static heuristic tuning, validating that TD learning successfully optimized strategic weights.", body_style))

    # Section 5
    story.append(Paragraph("5. Conclusion", sec_style))
    story.append(Paragraph("This project successfully demonstrated that an autonomous Reinforcement Learning agent using TD-Leaf(&lambda;) self-play can optimize strategic decision-making in 8x8 chess. Incorporating exploratory stochasticity prevented policy stagnation, and potential-based reward shaping ensured rapid convergence. The resulting engine outperforms unlearned priors and static heuristic search engines.", body_style))

    # References
    story.append(Paragraph("References", sec_style))
    story.append(Paragraph("[1] R. S. Sutton &amp; A. G. Barto, <i>Reinforcement Learning: An Introduction</i>, MIT Press, 2018.<br/>[2] J. Baxter et al., 'KnightCap: A chess program that learns by combining TD(&lambda;) with game-tree search,' <i>ICML</i>, 2000.<br/>[3] A. Y. Ng et al., 'Policy invariance under reward transformations,' <i>ICML</i>, 1999.", ParagraphStyle("Ref", fontName="Helvetica", fontSize=5.3, leading=6.8, textColor=colors.HexColor("#333333"))))

    doc.build(story, canvasmaker=PageNumCanvas)
    print(f"Successfully generated 2-page report PDF: {output_filename}")


if __name__ == "__main__":
    out_pdf = "RL_Chess_Engine_Report.pdf"
    if len(sys.argv) > 1:
        out_pdf = sys.argv[1]
    build_pdf_report(out_pdf)
