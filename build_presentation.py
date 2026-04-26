"""Generate the CIO methodology presentation for `private_assets_frequency`.

Target audience: portfolio managers and risk committee (non-quant).
Output: docs/private_assets_frequency_methodology_CIO.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

# ──────────────────────────────────────────────────────────────────
# Style / palette
# ──────────────────────────────────────────────────────────────────

NAVY = RGBColor(0x1B, 0x36, 0x5D)
TEAL = RGBColor(0x2E, 0x86, 0x8E)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREY = RGBColor(0x55, 0x5E, 0x68)
LIGHT_GREY = RGBColor(0xE7, 0xEA, 0xEE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GOLD = RGBColor(0xC9, 0xA2, 0x27)
GREEN = RGBColor(0x2E, 0x7D, 0x32)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def _add_blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])  # blank


def _add_textbox(slide, left, top, width, height, text, *,
                 size=18, bold=False, color=NAVY, align=PP_ALIGN.LEFT,
                 anchor=MSO_ANCHOR.TOP, line_spacing=None):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    if isinstance(text, str):
        lines = [text]
    else:
        lines = list(text)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if line_spacing is not None:
            p.line_spacing = line_spacing
        run = p.add_run()
        run.text = line
        run.font.name = "Calibri"
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return tb


def _add_bullets(slide, left, top, width, height, bullets, *,
                 size=18, color=NAVY, indent_size=14):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(bullets):
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        p.space_after = Pt(6)
        run = p.add_run()
        run.text = ("• " if level == 0 else "– ") + text
        run.font.name = "Calibri"
        run.font.size = Pt(size if level == 0 else indent_size)
        run.font.color.rgb = color


def _add_filled_box(slide, left, top, width, height, *,
                    fill=NAVY, line=NAVY, text=None, text_color=WHITE,
                    text_size=14, text_bold=True, align=PP_ALIGN.CENTER,
                    shape_type=MSO_SHAPE.ROUNDED_RECTANGLE):
    shape = slide.shapes.add_shape(shape_type, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(0.75)
    shape.shadow.inherit = False
    if text:
        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.08)
        tf.margin_right = Inches(0.08)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        if isinstance(text, str):
            text = [text]
        for i, line_text in enumerate(text):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            run = p.add_run()
            run.text = line_text
            run.font.name = "Calibri"
            run.font.size = Pt(text_size)
            run.font.bold = text_bold
            run.font.color.rgb = text_color
    return shape


def _add_arrow(slide, x1, y1, x2, y2, *, color=GREY, width=Pt(2)):
    arrow = slide.shapes.add_connector(2, x1, y1, x2, y2)  # straight line
    arrow.line.color.rgb = color
    arrow.line.width = width
    # Add an arrowhead at the end via MSO connector style; pptx exposes
    # arrow ends through XML — use a small triangle on top
    return arrow


def _add_down_arrow(slide, x, y_top, y_bottom, *, color=GREY):
    """Arrow connector pointing downward from y_top to y_bottom centred at x."""
    line = slide.shapes.add_connector(1, x, y_top, x, y_bottom)
    line.line.color.rgb = color
    line.line.width = Pt(2.5)
    # Decorative arrowhead — small filled triangle below the line endpoint
    head = slide.shapes.add_shape(
        MSO_SHAPE.ISOCELES_TRIANGLE,
        x - Inches(0.10),
        y_bottom - Inches(0.02),
        Inches(0.20),
        Inches(0.15),
    )
    head.rotation = 180.0
    head.fill.solid()
    head.fill.fore_color.rgb = color
    head.line.color.rgb = color


def _slide_header(slide, title, *, subtitle=None):
    # Title bar
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.0), Inches(0.0),
        SLIDE_W, Inches(0.85),
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    bar.shadow.inherit = False
    _add_textbox(
        slide,
        Inches(0.5), Inches(0.10),
        SLIDE_W - Inches(1.0), Inches(0.6),
        title,
        size=26, bold=True, color=WHITE, align=PP_ALIGN.LEFT,
        anchor=MSO_ANCHOR.MIDDLE,
    )
    if subtitle:
        _add_textbox(
            slide,
            Inches(0.5), Inches(0.95),
            SLIDE_W - Inches(1.0), Inches(0.4),
            subtitle,
            size=14, bold=False, color=GREY, align=PP_ALIGN.LEFT,
        )


def _footer(slide, page_num, total):
    _add_textbox(
        slide,
        Inches(0.5), SLIDE_H - Inches(0.4),
        Inches(7), Inches(0.3),
        "private_assets_frequency  ·  Methodology overview for CIO / risk committee",
        size=10, color=GREY,
    )
    _add_textbox(
        slide,
        SLIDE_W - Inches(1.5), SLIDE_H - Inches(0.4),
        Inches(1), Inches(0.3),
        f"{page_num} / {total}",
        size=10, color=GREY, align=PP_ALIGN.RIGHT,
    )


# ──────────────────────────────────────────────────────────────────
# Build the deck
# ──────────────────────────────────────────────────────────────────


def build():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    total_slides = 12

    # ── Slide 1: Title ──────────────────────────────────────────────
    s = _add_blank_slide(prs)
    bg = s.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H
    )
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    bg.shadow.inherit = False

    _add_textbox(
        s, Inches(0.8), Inches(2.4), Inches(11.7), Inches(1.0),
        "Reading private-asset returns honestly",
        size=44, bold=True, color=WHITE,
    )
    _add_textbox(
        s, Inches(0.8), Inches(3.5), Inches(11.7), Inches(0.7),
        "Why headline PE / RE volatility is wrong, and how we fix it",
        size=24, bold=False, color=GOLD,
    )
    _add_textbox(
        s, Inches(0.8), Inches(5.6), Inches(11.7), Inches(0.5),
        "private_assets_frequency  ·  methodology overview for CIO / risk committee",
        size=14, color=LIGHT_GREY,
    )
    _add_textbox(
        s, Inches(0.8), Inches(6.1), Inches(11.7), Inches(0.4),
        "Quantitative Research",
        size=12, color=LIGHT_GREY,
    )

    # ── Slide 2: The problem ───────────────────────────────────────
    s = _add_blank_slide(prs)
    _slide_header(s, "The problem with raw PE returns",
                  subtitle="Headline volatility is roughly half of the truth — and the diversification benefit is overstated")

    _add_bullets(
        s, Inches(0.6), Inches(1.5), Inches(7.6), Inches(5.5),
        [
            "Private-equity, real-estate, infrastructure and hedge-fund indices are *appraisal-based* or *NAV-based*.",
            "Reported returns smooth genuine market moves over multiple quarters — appraisers anchor on the prior NAV.",
            "Result: headline σ understates true risk by 50%+.",
            "Result: correlation with public markets looks artificially low — diversification benefit is overstated.",
            "Risk numbers built from the raw series — VaR, drawdown, capital ratios — are dangerously optimistic.",
            "This is well-documented (Geltner 1993; Getmansky-Lo-Makarov 2004).  It is not a controversial finding.",
        ],
        size=18,
    )

    # Right-side callout
    _add_filled_box(
        s, Inches(8.6), Inches(2.0), Inches(4.2), Inches(3.5),
        fill=LIGHT_GREY, line=NAVY,
        text=[
            "Typical observation",
            "",
            "US Buyout headline σ:  ~10%",
            "Levered Russell 2000 σ:  ~25%",
            "",
            "Headline PE σ ≈ 0.4× public.",
            "True PE σ ≈ 1.0–1.2× public.",
        ],
        text_color=NAVY, text_size=14,
    )

    _footer(s, 2, total_slides)

    # ── Slide 3: Concrete example ─────────────────────────────────
    s = _add_blank_slide(prs)
    _slide_header(s, "What the smoothing actually looks like",
                  subtitle="Same underlying economic event, two very different reported series")

    # Two stacked illustration boxes
    _add_filled_box(
        s, Inches(0.6), Inches(1.6), Inches(5.9), Inches(2.4),
        fill=LIGHT_GREY, line=GREY, text=None,
    )
    _add_textbox(
        s, Inches(0.8), Inches(1.7), Inches(5.5), Inches(0.4),
        "Public market (Russell 2000)",
        size=16, bold=True, color=NAVY,
    )
    _add_textbox(
        s, Inches(0.8), Inches(2.1), Inches(5.5), Inches(1.8),
        [
            "Q3 2008:    -8.7%",
            "Q4 2008:   -26.0%",
            "Q1 2009:    -14.7%",
            "",
            "→ Drawdown clearly visible quarter-by-quarter.",
        ],
        size=14, color=GREY,
    )

    _add_filled_box(
        s, Inches(6.7), Inches(1.6), Inches(6.0), Inches(2.4),
        fill=LIGHT_GREY, line=GREY, text=None,
    )
    _add_textbox(
        s, Inches(6.9), Inches(1.7), Inches(5.5), Inches(0.4),
        "Reported PE index (representative)",
        size=16, bold=True, color=NAVY,
    )
    _add_textbox(
        s, Inches(6.9), Inches(2.1), Inches(5.5), Inches(1.8),
        [
            "Q3 2008:    -3.2%",
            "Q4 2008:   -10.5%",
            "Q1 2009:     -6.8%",
            "",
            "→ Same crisis, but spread over more quarters.",
        ],
        size=14, color=GREY,
    )

    # Bottom interpretation
    _add_filled_box(
        s, Inches(0.6), Inches(4.4), Inches(12.1), Inches(2.3),
        fill=NAVY, line=NAVY,
        text=[
            "What's actually happening",
            "",
            "Both portfolios suffered the same economic shock.  The PE index reports a softer drawdown",
            "because appraisals rely heavily on the previous quarter's NAV, then revise toward reality slowly.",
            "Build a VaR model on the PE numbers and you will under-reserve in a crisis.",
        ],
        text_color=WHITE, text_size=15,
    )

    _footer(s, 3, total_slides)

    # ── Slide 4: Why it happens ───────────────────────────────────
    s = _add_blank_slide(prs)
    _slide_header(s, "Why the smoothing happens",
                  subtitle="Three mechanisms, all with the same statistical signature")

    rows = [
        ("Appraisal anchoring",
         "PE / RE valuations rely on the previous NAV — appraisers move slowly to avoid client dispute."),
        ("Stale marks",
         "Loans in private credit are carried at par for quarters before being written down at a credit event."),
        ("Reporting lag",
         "Hedge fund returns aggregate constituent reports over 1–2 months — month t reflects month t−1 economics."),
    ]
    y = Inches(1.6)
    for label, desc in rows:
        _add_filled_box(
            s, Inches(0.6), y, Inches(3.0), Inches(1.3),
            fill=NAVY, line=NAVY,
            text=label, text_color=WHITE, text_size=18,
        )
        _add_filled_box(
            s, Inches(3.7), y, Inches(9.0), Inches(1.3),
            fill=LIGHT_GREY, line=GREY,
            text=desc, text_color=NAVY, text_size=14, text_bold=False,
            align=PP_ALIGN.LEFT,
        )
        y += Inches(1.5)

    _add_textbox(
        s, Inches(0.6), Inches(6.2), Inches(12.1), Inches(0.7),
        "All three produce the same statistical fingerprint:  reported return = (1−λ)·true return + λ·last NAV.",
        size=16, bold=True, color=TEAL,
    )

    _footer(s, 4, total_slides)

    # ── Slide 5: What desmoothing does conceptually ──────────────
    s = _add_blank_slide(prs)
    _slide_header(s, "What desmoothing does, in plain language",
                  subtitle="Recover the economic return that actually happened from the smoothed report")

    # Top conceptual equation block
    _add_filled_box(
        s, Inches(0.6), Inches(1.5), Inches(12.1), Inches(1.5),
        fill=LIGHT_GREY, line=NAVY,
        text=[
            "If reported NAV is a weighted average of last NAV and true return:",
            "reported_t  =  (1 − λ)  ×  true_t   +   λ  ×  reported_{t−1}",
            "then we can invert that relationship to recover true_t.",
        ],
        text_color=NAVY, text_size=15, text_bold=False,
    )

    # λ interpretation
    _add_textbox(
        s, Inches(0.6), Inches(3.2), Inches(12.0), Inches(0.5),
        "λ = the smoothing intensity.  We estimate it from the data, with priors from the literature.",
        size=15, bold=True, color=NAVY,
    )

    # Three boxes: low, mid, high lambda
    boxes = [
        ("λ ≈ 0.0",
         "No smoothing.\nManaged-futures, exchange-traded.",
         GREEN),
        ("λ ≈ 0.5",
         "Moderate smoothing.\nTypical buyout / infrastructure.",
         GOLD),
        ("λ ≈ 0.85",
         "Heavy smoothing.\nReal estate / venture capital.",
         RED),
    ]
    x = Inches(0.6)
    for label, desc, color in boxes:
        _add_filled_box(
            s, x, Inches(3.9), Inches(4.0), Inches(0.7),
            fill=color, line=color,
            text=label, text_color=WHITE, text_size=18,
        )
        _add_filled_box(
            s, x, Inches(4.65), Inches(4.0), Inches(1.4),
            fill=LIGHT_GREY, line=GREY,
            text=desc, text_color=NAVY, text_size=13, text_bold=False,
        )
        x += Inches(4.05)

    # Bottom takeaway
    _add_textbox(
        s, Inches(0.6), Inches(6.3), Inches(12.0), Inches(0.7),
        "The output is the *economic* return path consistent with the reported series — what you would have seen with daily marks.",
        size=14, color=GREY, align=PP_ALIGN.LEFT,
    )

    _footer(s, 5, total_slides)

    # ── Slide 6: Pipeline architecture diagram ───────────────────
    s = _add_blank_slide(prs)
    _slide_header(s, "How the pipeline works",
                  subtitle="Five stages.  Each stage is replaceable;  the same plumbing handles every asset class.")

    # Centred 5-stage flow
    stage_x = Inches(0.6)
    stage_w = Inches(2.3)
    gap = Inches(0.1)
    y = Inches(1.7)
    h = Inches(1.4)

    stages = [
        ("Stage 0", "Preprocessing", "Carry/MTM split for credit;\nlag fix for HF;\npass-through otherwise.", TEAL),
        ("Stage 1", "Desmoothing", "Estimate λ from the data,\nrecover the economic path,\nemit identifiability checks.", NAVY),
        ("Stage 2", "Factor decomposition", "Regress on public-market\nfactors; extract residual\ndistribution.", NAVY),
        ("Stage 3", "Temporal disaggregation", "Push quarterly → monthly\n(or monthly → daily)\nwhile preserving cumulative returns.", NAVY),
        ("Stage 4", "Daily extension", "Either a smoothed daily\npath (backtest) or N\nsimulation paths (VaR).", NAVY),
    ]

    x = Inches(0.4)
    for i, (label, title, body, color) in enumerate(stages):
        # Header box
        _add_filled_box(
            s, x, y, stage_w, Inches(0.5),
            fill=color, line=color,
            text=label, text_color=WHITE, text_size=14,
        )
        # Title
        _add_filled_box(
            s, x, y + Inches(0.5), stage_w, Inches(0.5),
            fill=GOLD, line=GOLD,
            text=title, text_color=NAVY, text_size=13,
        )
        # Body
        _add_filled_box(
            s, x, y + Inches(1.0), stage_w, Inches(1.6),
            fill=LIGHT_GREY, line=GREY,
            text=body, text_color=NAVY, text_size=11,
            text_bold=False,
        )
        # Arrow to next stage
        if i < 4:
            arrow = slide_arrow_right(s, x + stage_w, y + Inches(1.3),
                                       Inches(0.3), Inches(0.15))
        x += stage_w + Inches(0.30)

    # Validation strip below
    _add_filled_box(
        s, Inches(0.4), Inches(5.0), Inches(12.5), Inches(0.7),
        fill=NAVY, line=NAVY,
        text="At every stage:  validation gate (e.g. Stage 3 → Stage 4 round-trip aggregation < 1e-10) + diagnostic warnings",
        text_color=WHITE, text_size=13,
    )

    # Sanity-anchor strip
    _add_filled_box(
        s, Inches(0.4), Inches(5.85), Inches(12.5), Inches(0.7),
        fill=GOLD, line=GOLD,
        text="At the end:  sanity anchors against a public proxy — vol, correlation, crisis drawdown, beta",
        text_color=NAVY, text_size=13,
    )

    _footer(s, 6, total_slides)

    # ── Slide 7: Asset class coverage ────────────────────────────
    s = _add_blank_slide(prs)
    _slide_header(s, "Coverage across asset classes",
                  subtitle="One library, asset-class-specific priors and preprocessing")

    headers = ["Asset class", "Smoothing model", "Native frequency", "Stage 0 preprocessing"]
    rows = [
        ["Private equity (buyout, VC, mezz, distressed)", "AR(1) Bayesian", "Quarterly", "—"],
        ["Private infrastructure (core, opportunistic)", "AR(1) Bayesian", "Quarterly", "—"],
        ["Private credit (direct lending, senior debt)", "Threshold AR(1)", "Quarterly", "Carry / MTM split"],
        ["Hedge funds (eq L/S, macro, event, RV)", "MA(q) Getmansky-Lo-Makarov", "Monthly", "Reporting-lag adjustment"],
        ["Hedge funds (managed futures)", "NoSmoothing", "Monthly", "—"],
        ["Private real estate (core)", "AR(1) Bayesian", "Quarterly", "—"],
    ]

    table_left = Inches(0.6)
    table_top = Inches(1.7)
    col_widths = [Inches(4.4), Inches(3.4), Inches(2.0), Inches(2.7)]
    row_h = Inches(0.55)

    # Header row
    x = table_left
    for w, header in zip(col_widths, headers):
        _add_filled_box(
            s, x, table_top, w, row_h,
            fill=NAVY, line=NAVY,
            text=header, text_color=WHITE, text_size=13, text_bold=True,
            align=PP_ALIGN.LEFT,
        )
        x += w
    # Data rows
    y = table_top + row_h
    for r, row in enumerate(rows):
        x = table_left
        bg = LIGHT_GREY if r % 2 == 0 else WHITE
        for w, cell in zip(col_widths, row):
            _add_filled_box(
                s, x, y, w, row_h,
                fill=bg, line=GREY,
                text=cell, text_color=NAVY, text_size=12, text_bold=False,
                align=PP_ALIGN.LEFT,
            )
            x += w
        y += row_h

    _add_textbox(
        s, Inches(0.6), Inches(5.6), Inches(12.0), Inches(1.0),
        "Each preset ships with strategy-specific factor priors (β, α) calibrated from the literature.\n"
        "Users can override any parameter for bespoke strategies.",
        size=14, color=GREY,
    )

    _footer(s, 7, total_slides)

    # ── Slide 8: How we know it's working ────────────────────────
    s = _add_blank_slide(prs)
    _slide_header(s, "How we know it's working",
                  subtitle="Independent of the desmoother — these are the cross-checks the risk committee can read directly")

    # Four anchor cards
    anchors = [
        ("Volatility anchor",
         "Desmoothed σ within 0.5×–3.0× public proxy σ.",
         "Catches under-correction (still smoothed) and over-correction."),
        ("Correlation anchor",
         "Desmoothed correlation with public proxy in [0.4, 0.95].",
         "Below 0.4 → factor model is missing a factor.\nAbove 0.95 → the output is just the public proxy."),
        ("Crisis drawdown anchor",
         "Desmoothed crisis drawdown ≥ 50% × public proxy's.",
         "GFC and COVID windows ship as defaults; custom windows configurable."),
        ("Beta anchor",
         "Estimated β to public proxy consistent with leverage.",
         "Unleveraged VC β > 1.5 or buyout β < 0.5 trips a warning."),
    ]
    x_positions = [Inches(0.6), Inches(6.95)]
    y_positions = [Inches(1.6), Inches(4.3)]
    for i, (title, rule, desc) in enumerate(anchors):
        x = x_positions[i % 2]
        y = y_positions[i // 2]
        _add_filled_box(
            s, x, y, Inches(5.95), Inches(0.6),
            fill=NAVY, line=NAVY,
            text=title, text_color=WHITE, text_size=16, text_bold=True,
        )
        _add_filled_box(
            s, x, y + Inches(0.6), Inches(5.95), Inches(0.7),
            fill=GOLD, line=GOLD,
            text=rule, text_color=NAVY, text_size=13, text_bold=True,
        )
        _add_filled_box(
            s, x, y + Inches(1.3), Inches(5.95), Inches(1.2),
            fill=LIGHT_GREY, line=GREY,
            text=desc, text_color=NAVY, text_size=12, text_bold=False,
        )

    _footer(s, 8, total_slides)

    # ── Slide 9: What you get out ────────────────────────────────
    s = _add_blank_slide(prs)
    _slide_header(s, "What the pipeline delivers",
                  subtitle="One call returns everything you need for risk and reporting")

    out_rows = [
        ("monthly_returns",
         "Wide DataFrame — strategies × monthly dates.\nReady for risk / VaR / capital models.",
         NAVY),
        ("daily_returns",
         "When daily factors are supplied — strategies × business days.\nReady for backtesting and intraday risk.",
         NAVY),
        ("warnings",
         "Plain-English list of any identifiability or sanity-anchor flags.\nSurface this to the committee, do not ignore it.",
         GOLD),
        ("sanity_anchors",
         "Per-strategy vol / correlation / drawdown / beta against a public proxy.",
         TEAL),
        ("uncertainty_bands",
         "When uncertainty_mode='full' — 5/25/50/75/95 percentile DataFrames.\nUse for stress-testing the model itself.",
         RED),
    ]

    y = Inches(1.6)
    for label, desc, color in out_rows:
        _add_filled_box(
            s, Inches(0.6), y, Inches(3.5), Inches(0.95),
            fill=color, line=color,
            text=label, text_color=WHITE, text_size=14, text_bold=True,
        )
        _add_filled_box(
            s, Inches(4.2), y, Inches(8.5), Inches(0.95),
            fill=LIGHT_GREY, line=GREY,
            text=desc, text_color=NAVY, text_size=12, text_bold=False,
            align=PP_ALIGN.LEFT,
        )
        y += Inches(1.05)

    _footer(s, 9, total_slides)

    # ── Slide 10: Uncertainty mode ───────────────────────────────
    s = _add_blank_slide(prs)
    _slide_header(s, "Honest uncertainty:  posterior bands",
                  subtitle="The pipeline tells you how much it does *not* know")

    _add_bullets(
        s, Inches(0.6), Inches(1.5), Inches(7.6), Inches(5.5),
        [
            "λ (the smoothing intensity) is estimated, not known.",
            "Two values of λ can fit the same observed series almost equally well — this is real uncertainty.",
            "uncertainty_mode='full' draws 100 samples from the joint posterior and re-runs Stage 3 for each.",
            "Output: 5/25/50/75/95 pointwise percentile bands on the monthly path.",
            "Build VaR on the 95th percentile band, not the median, when the credible interval is wide.",
            "Identifiability diagnostics fire when bands are too wide — the committee sees the warning before the number is used.",
        ],
        size=18,
    )

    # Sketch on the right: 3 stacked horizontal bars representing the bands
    band_x = Inches(8.7)
    band_w = Inches(4.0)
    band_top = Inches(2.0)
    bands_data = [
        ("95%", Inches(3.6)),
        ("75%", Inches(2.2)),
        ("50%", Inches(1.0)),
        ("25%", Inches(2.2)),
        ("5%",  Inches(3.6)),
    ]
    colors_for_band = [
        RGBColor(0xC4, 0xCD, 0xD9),
        RGBColor(0x8B, 0xA0, 0xB7),
        NAVY,
        RGBColor(0x8B, 0xA0, 0xB7),
        RGBColor(0xC4, 0xCD, 0xD9),
    ]
    y0 = band_top
    for (label, w), c in zip(bands_data, colors_for_band):
        _add_filled_box(
            s, band_x + (band_w - w) / 2, y0, w, Inches(0.4),
            fill=c, line=c,
            text=label, text_color=WHITE, text_size=12, text_bold=True,
        )
        y0 += Inches(0.5)
    _add_textbox(
        s, band_x, band_top - Inches(0.4), band_w, Inches(0.4),
        "Posterior band width (illustrative)",
        size=12, color=GREY, align=PP_ALIGN.CENTER,
    )

    _footer(s, 10, total_slides)

    # ── Slide 11: Caveats / honest framing ──────────────────────
    s = _add_blank_slide(prs)
    _slide_header(s, "What this is — and what it isn't",
                  subtitle="Use the output for risk decisions, but keep the framing straight")

    # Two columns: IS / IS NOT
    _add_filled_box(
        s, Inches(0.6), Inches(1.6), Inches(6.0), Inches(0.6),
        fill=GREEN, line=GREEN,
        text="What it IS", text_color=WHITE, text_size=18, text_bold=True,
    )
    _add_filled_box(
        s, Inches(6.75), Inches(1.6), Inches(6.0), Inches(0.6),
        fill=RED, line=RED,
        text="What it ISN'T", text_color=WHITE, text_size=18, text_bold=True,
    )

    is_items = [
        "A statistically rigorous, model-implied path of returns consistent with the observed series.",
        "Suitable for VaR, drawdown, capital, and correlation analytics that the raw series fails to support.",
        "Transparent about its own uncertainty via diagnostics, sanity anchors, and uncertainty bands.",
        "Reproducible — same seed, same priors, same answer.",
    ]
    isnt_items = [
        "NOT the *true* daily return path — that doesn't exist; it's a conditional expectation.",
        "NOT a substitute for fund-level due diligence on individual managers.",
        "NOT magic — when the raw data is uninformative, the diagnostics will say so loudly.",
        "NOT a forecasting tool — it characterises *historical* risk, not future returns.",
    ]
    for i, (it, isnt) in enumerate(zip(is_items, isnt_items)):
        y = Inches(2.3 + i * 1.05)
        _add_filled_box(
            s, Inches(0.6), y, Inches(6.0), Inches(0.95),
            fill=LIGHT_GREY, line=GREEN,
            text="✓  " + it, text_color=NAVY, text_size=13, text_bold=False,
            align=PP_ALIGN.LEFT,
        )
        _add_filled_box(
            s, Inches(6.75), y, Inches(6.0), Inches(0.95),
            fill=LIGHT_GREY, line=RED,
            text="✗  " + isnt, text_color=NAVY, text_size=13, text_bold=False,
            align=PP_ALIGN.LEFT,
        )

    _footer(s, 11, total_slides)

    # ── Slide 12: Summary / takeaways ───────────────────────────
    s = _add_blank_slide(prs)
    _slide_header(s, "Three takeaways",
                  subtitle="What to remember when the next risk meeting starts")

    takeaways = [
        ("1",
         "Headline PE / RE / HF risk is wrong by design.",
         "Reported indices are smoothed.  Building risk numbers on the raw series under-states σ by ~50% and over-states diversification.  This is the literature, not opinion."),
        ("2",
         "Desmoothing is an honest, model-driven correction.",
         "The library implements Geltner / Getmansky-Lo-Makarov / MSCI methodology, with strategy-specific priors and explicit identifiability checks."),
        ("3",
         "The output comes with its own quality controls.",
         "Sanity anchors against public proxies, posterior uncertainty bands, and inter-stage validation give the risk committee the receipts before they sign off."),
    ]
    y = Inches(1.5)
    for num, head, body in takeaways:
        _add_filled_box(
            s, Inches(0.6), y, Inches(0.9), Inches(1.6),
            fill=GOLD, line=GOLD,
            text=num, text_color=NAVY, text_size=44, text_bold=True,
        )
        _add_filled_box(
            s, Inches(1.6), y, Inches(11.0), Inches(0.55),
            fill=NAVY, line=NAVY,
            text=head, text_color=WHITE, text_size=17, text_bold=True,
            align=PP_ALIGN.LEFT,
        )
        _add_filled_box(
            s, Inches(1.6), y + Inches(0.55), Inches(11.0), Inches(1.05),
            fill=LIGHT_GREY, line=GREY,
            text=body, text_color=NAVY, text_size=13, text_bold=False,
            align=PP_ALIGN.LEFT,
        )
        y += Inches(1.85)

    _footer(s, 12, total_slides)

    # ──────────────────────────────────────────────────────────────
    out = Path("docs/private_assets_frequency_methodology_CIO.pptx")
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(f"wrote {out}  ({out.stat().st_size:,} bytes,  {total_slides} slides)")


def slide_arrow_right(slide, x, y, w, h):
    """Add a small right-pointing arrow shape between pipeline boxes."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW, x, y, w, h,
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = GREY
    shape.line.color.rgb = GREY
    shape.shadow.inherit = False
    return shape


if __name__ == "__main__":
    build()
