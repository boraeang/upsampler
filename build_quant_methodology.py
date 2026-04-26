"""Build the quant methodology presentation for `private_assets_frequency`.

Audience: quantitative researchers / portfolio analysts.
Output: docs/private_assets_frequency_methodology.pptx (24 slides, 16:9, dark theme).

Equations: Greek-letter inline equations as Consolas text in teal; complex
fractions / integrals / GLS formulas rendered as transparent-background PNGs
via matplotlib mathtext and embedded.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

# ──────────────────────────────────────────────────────────────────
# Palette & typography
# ──────────────────────────────────────────────────────────────────

BG_TITLE   = RGBColor(0x1B, 0x1F, 0x2A)  # near-black
BG_CONTENT = RGBColor(0x22, 0x27, 0x2E)  # dark charcoal
BG_PANEL   = RGBColor(0x2A, 0x30, 0x38)  # slightly lighter for inset cards
TEXT       = RGBColor(0xE8, 0xE8, 0xE8)  # off-white
TEXT_DIM   = RGBColor(0xA0, 0xA8, 0xB0)  # muted grey
TEAL       = RGBColor(0x00, 0xB4, 0xD8)  # equations / accent
GOLD       = RGBColor(0xD4, 0xA8, 0x43)  # secondary accent / assumption tags
RED        = RGBColor(0xE0, 0x7A, 0x5F)  # warnings / "where this breaks"
GREEN      = RGBColor(0x81, 0xB2, 0x9A)  # positive / result
RULE       = RGBColor(0x40, 0x47, 0x52)  # divider rules

# Hex strings for matplotlib calls
TEAL_HEX  = "#00B4D8"
GOLD_HEX  = "#D4A843"
TEXT_HEX  = "#E8E8E8"
RED_HEX   = "#E07A5F"
GREEN_HEX = "#81B29A"

EQ_FONT     = "Consolas"   # equations / inline notation
BODY_FONT   = "Calibri"
TITLE_FONT  = "Calibri"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# Where the rendered equation PNGs land
EQ_DIR = Path(tempfile.mkdtemp(prefix="paf_eqs_"))

# matplotlib base config for equation rendering
rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "savefig.transparent": True,
    "savefig.dpi": 220,
    "axes.facecolor": "none",
})


# ──────────────────────────────────────────────────────────────────
# Equation rendering
# ──────────────────────────────────────────────────────────────────


def render_equation(name: str, latex: str, *, color: str = TEAL_HEX,
                    fontsize: int = 30, width: float = 8.0,
                    height: float = 1.4) -> Path:
    """Render a LaTeX-style mathtext expression to a transparent PNG."""
    out = EQ_DIR / f"{name}.png"
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.text(
        0.5, 0.5, latex,
        ha="center", va="center",
        fontsize=fontsize, color=color,
    )
    ax.axis("off")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    return out


# Pre-render the canonical equations so we can size & position confidently.
EQ = {}

EQ["smoothing_value"]   = render_equation(
    "smoothing_value",
    r"$P_t \;=\; P_{t-1} \;+\; (1 - \lambda)\,(V_t - P_{t-1})$",
    fontsize=32, width=10, height=1.0,
)
EQ["smoothing_return"]  = render_equation(
    "smoothing_return",
    r"$s_t \;=\; (1 - \lambda)\,r_t \;+\; \lambda\,s_{t-1}$",
    fontsize=32, width=10, height=1.0,
)
EQ["desmooth"]          = render_equation(
    "desmooth",
    r"$r_t \;=\; \dfrac{s_t \;-\; \lambda\,s_{t-1}}{1 - \lambda}$",
    fontsize=38, width=8, height=1.6,
)
EQ["desmooth_annual"]   = render_equation(
    "desmooth_annual",
    r"$r^{\text{annual}}_t \;=\; \dfrac{s_t \;-\; \lambda\,s_{t-4}}{1 - \lambda}$",
    fontsize=34, width=10, height=1.6,
)
EQ["factor_model"]      = render_equation(
    "factor_model",
    r"$r_t^{\text{annual}} \;=\; \boldsymbol{\beta}^{\top} \mathbf{F}_t^{\text{annual}} \;+\; \alpha \;+\; \varepsilon_t$",
    fontsize=30, width=10, height=1.0,
)
EQ["posterior_kernel"]  = render_equation(
    "posterior_kernel",
    r"$p(\lambda,\boldsymbol{\beta},\alpha,\sigma_\varepsilon \,|\, \mathbf{s}) "
    r"\;\propto\; \mathcal{L}(\mathbf{s}\,|\,\lambda,\boldsymbol{\beta},\alpha,\sigma_\varepsilon)\, "
    r"\pi(\lambda)\,\pi(\boldsymbol{\beta})\,\pi(\alpha)\,\pi(\sigma_\varepsilon)$",
    fontsize=22, width=12, height=1.0,
)
EQ["posterior_marg"]    = render_equation(
    "posterior_marg",
    r"$p(\lambda \,|\, \mathbf{s}) \;=\; "
    r"\int\int\int "
    r"p(\lambda,\boldsymbol{\beta},\alpha,\sigma_\varepsilon \,|\, \mathbf{s})\,"
    r"d\boldsymbol{\beta}\,d\alpha\,d\sigma_\varepsilon$",
    fontsize=24, width=12, height=1.0,
)
EQ["vasicek"]           = render_equation(
    "vasicek",
    r"$\hat\beta_{\text{Vas}} \;=\; w \cdot \hat\beta_{\text{OLS}} \;+\; (1 - w)\,\beta_{\text{prior}}$",
    fontsize=30, width=10, height=1.0,
)
EQ["ma_smoothing"]      = render_equation(
    "ma_smoothing",
    r"$s_t \;=\; \theta_0\, r_t \;+\; \theta_1\, r_{t-1} \;+\; \cdots \;+\; \theta_q\, r_{t-q},"
    r"\quad \sum_j \theta_j = 1,\;\; \theta_j \geq 0$",
    fontsize=22, width=13, height=1.0,
)
EQ["ma_inversion"]      = render_equation(
    "ma_inversion",
    r"$\mathbf{S} \;=\; \boldsymbol{\Theta}\,\mathbf{R} "
    r"\quad\Longrightarrow\quad \mathbf{R} \;=\; \boldsymbol{\Theta}^{-1}\mathbf{S}$",
    fontsize=30, width=10, height=1.0,
)
EQ["ma_autocov"]        = render_equation(
    "ma_autocov",
    r"$\mathrm{Var}(s_t) = \sigma_r^2 \sum_j \theta_j^2,"
    r"\quad \mathrm{Cov}(s_t, s_{t-k}) = \sigma_r^2 \sum_{j=0}^{q-k} \theta_j\,\theta_{j+k}$",
    fontsize=22, width=13, height=1.0,
)
EQ["threshold"]         = render_equation(
    "threshold",
    r"$s_t \;=\; (1 - \lambda(\mathrm{regime}_t))\,r_t \;+\; \lambda(\mathrm{regime}_t)\,s_{t-1}$",
    fontsize=24, width=13, height=1.0,
)
EQ["credit_decomp"]     = render_equation(
    "credit_decomp",
    r"$r^{\text{total}}_t \;=\; r^{\text{carry}}_t \;+\; r^{\text{MTM}}_t,"
    r"\quad r^{\text{carry}}_t \approx y_t \cdot \Delta t$",
    fontsize=24, width=13, height=1.0,
)
EQ["factor_decomp"]     = render_equation(
    "factor_decomp",
    r"$r_t \;=\; \sum_{k=1}^{K} \beta_k\, F_{k,t} \;+\; \alpha \;+\; \varepsilon_t$",
    fontsize=30, width=10, height=1.0,
)
EQ["chow_lin"]          = render_equation(
    "chow_lin",
    r"$\hat{\mathbf{y}}_m \;=\; \mathbf{X}_m\hat{\boldsymbol{\beta}} \;+\; "
    r"\mathbf{V}\,\mathbf{C}^{\top}(\mathbf{C}\,\mathbf{V}\,\mathbf{C}^{\top})^{-1}\,"
    r"(\mathbf{y}_Q - \mathbf{C}\,\mathbf{X}_m\hat{\boldsymbol{\beta}})$",
    fontsize=22, width=14, height=1.4,
)
EQ["agg_constraint"]    = render_equation(
    "agg_constraint",
    r"$\prod_{m \in Q}(1 + r_m) \;=\; 1 + r_Q "
    r"\quad\Longleftrightarrow\quad \sum_{m \in Q}\log(1 + r_m) = \log(1 + r_Q)$",
    fontsize=22, width=14, height=1.0,
)
EQ["kalman_state"]      = render_equation(
    "kalman_state",
    r"$\varepsilon_d \;=\; \varphi\,\varepsilon_{d-1} \;+\; \eta_d,\;\;\;\;"
    r"\eta_d \sim \mathcal{N}(0, \sigma_\eta^2)$",
    fontsize=28, width=11, height=1.0,
)
EQ["kalman_obs"]        = render_equation(
    "kalman_obs",
    r"$\varepsilon^{\text{month}}_m \;=\; \sum_{d \in m} \varepsilon_d$",
    fontsize=30, width=8, height=1.0,
)
EQ["sim_path"]          = render_equation(
    "sim_path",
    r"$r_d^{(j)} \;=\; \boldsymbol{\beta}^{\top}\mathbf{F}_d \;+\; \varepsilon_d^{(j)},\;\;\;\;"
    r"\varepsilon_d^{(j)} \sim \mathcal{D}\left(\hat\theta_\varepsilon\right)$",
    fontsize=24, width=12, height=1.0,
)


# ──────────────────────────────────────────────────────────────────
# pptx helpers
# ──────────────────────────────────────────────────────────────────


def _new_slide(prs, *, bg=BG_CONTENT):
    s = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bg_shape = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = bg
    bg_shape.line.fill.background()
    bg_shape.shadow.inherit = False
    return s


def _text(slide, x, y, w, h, text, *, size=18, bold=False, color=TEXT,
          font=BODY_FONT, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
          line_spacing=None, italic=False):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
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
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
    return tb


def _bullets(slide, x, y, w, h, items, *, size=16, color=TEXT, font=BODY_FONT,
             space_after=8):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        p.space_after = Pt(space_after)
        run = p.add_run()
        bullet_char = "•" if level == 0 else "–"
        run.text = f"{bullet_char}  {text}"
        run.font.name = font
        run.font.size = Pt(size if level == 0 else max(size - 2, 12))
        run.font.color.rgb = color


def _slide_title(slide, title, *, subtitle=None):
    # Title text
    _text(
        slide, Inches(0.55), Inches(0.4),
        SLIDE_W - Inches(1.1), Inches(0.7),
        title, size=30, bold=True, color=TEXT, font=TITLE_FONT,
        anchor=MSO_ANCHOR.TOP,
    )
    # Underline rule
    rule = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.55), Inches(1.05),
        Inches(1.6), Inches(0.04),
    )
    rule.fill.solid()
    rule.fill.fore_color.rgb = TEAL
    rule.line.fill.background()
    rule.shadow.inherit = False
    if subtitle:
        _text(
            slide, Inches(0.55), Inches(1.2),
            SLIDE_W - Inches(1.1), Inches(0.4),
            subtitle, size=14, color=TEXT_DIM, italic=True,
        )


def _footer(slide, page_num, total):
    # Bottom rule
    rule = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.55), SLIDE_H - Inches(0.45),
        SLIDE_W - Inches(1.1), Inches(0.015),
    )
    rule.fill.solid()
    rule.fill.fore_color.rgb = RULE
    rule.line.fill.background()
    rule.shadow.inherit = False
    _text(
        slide, Inches(0.55), SLIDE_H - Inches(0.36),
        Inches(8.0), Inches(0.3),
        "Quantitative Research — Internal Use Only",
        size=10, color=TEXT_DIM,
    )
    _text(
        slide, SLIDE_W - Inches(2.0), SLIDE_H - Inches(0.36),
        Inches(1.45), Inches(0.3),
        f"{page_num} / {total}",
        size=10, color=TEXT_DIM, align=PP_ALIGN.RIGHT,
    )


def _add_image(slide, png_path, x, y, *, width=None, height=None):
    if width is not None and height is not None:
        return slide.shapes.add_picture(str(png_path), x, y, width=width, height=height)
    if width is not None:
        return slide.shapes.add_picture(str(png_path), x, y, width=width)
    if height is not None:
        return slide.shapes.add_picture(str(png_path), x, y, height=height)
    return slide.shapes.add_picture(str(png_path), x, y)


def _equation_panel(slide, x, y, w, h, png_path, *, fill=BG_PANEL):
    """Place a centred equation PNG inside a subtle panel."""
    panel = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    panel.fill.solid()
    panel.fill.fore_color.rgb = fill
    panel.line.color.rgb = RULE
    panel.line.width = Pt(0.5)
    panel.shadow.inherit = False
    # Centre the PNG inside; scale to fit the panel with a small margin
    pad_x = Inches(0.2)
    pad_y = Inches(0.15)
    pic = _add_image(slide, png_path, x + pad_x, y + pad_y,
                     width=w - 2 * pad_x, height=h - 2 * pad_y)
    return panel


def _assumption_tag(slide, x, y, label):
    """Small gold pill containing an assumption label like 'A1'."""
    w = Inches(0.55)
    h = Inches(0.30)
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    pill.fill.solid()
    pill.fill.fore_color.rgb = GOLD
    pill.line.color.rgb = GOLD
    pill.shadow.inherit = False
    tf = pill.text_frame
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = label
    run.font.name = BODY_FONT
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = BG_TITLE


def _breaks_callout(slide, x, y, w, h, body_lines, *, title="Where this breaks"):
    # Vertical red border
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Inches(0.06), h)
    bar.fill.solid()
    bar.fill.fore_color.rgb = RED
    bar.line.fill.background()
    bar.shadow.inherit = False
    # Inset panel
    panel = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        x + Inches(0.06), y, w - Inches(0.06), h,
    )
    panel.fill.solid()
    panel.fill.fore_color.rgb = BG_PANEL
    panel.line.color.rgb = RULE
    panel.line.width = Pt(0.5)
    panel.shadow.inherit = False
    # Header
    _text(
        slide, x + Inches(0.22), y + Inches(0.1),
        w - Inches(0.4), Inches(0.35),
        title.upper(), size=11, bold=True, color=RED, font=BODY_FONT,
    )
    # Body
    _text(
        slide, x + Inches(0.22), y + Inches(0.45),
        w - Inches(0.4), h - Inches(0.55),
        body_lines, size=12, color=TEXT, line_spacing=1.15,
    )


def _result_callout(slide, x, y, w, h, body_lines, *, title="Result"):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Inches(0.06), h)
    bar.fill.solid()
    bar.fill.fore_color.rgb = GREEN
    bar.line.fill.background()
    bar.shadow.inherit = False
    panel = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        x + Inches(0.06), y, w - Inches(0.06), h,
    )
    panel.fill.solid()
    panel.fill.fore_color.rgb = BG_PANEL
    panel.line.color.rgb = RULE
    panel.line.width = Pt(0.5)
    panel.shadow.inherit = False
    _text(
        slide, x + Inches(0.22), y + Inches(0.1),
        w - Inches(0.4), Inches(0.35),
        title.upper(), size=11, bold=True, color=GREEN, font=BODY_FONT,
    )
    _text(
        slide, x + Inches(0.22), y + Inches(0.45),
        w - Inches(0.4), h - Inches(0.55),
        body_lines, size=12, color=TEXT, line_spacing=1.15,
    )


def _filled_box(slide, x, y, w, h, *, fill=BG_PANEL, line=RULE,
                text_lines=None, text_color=TEXT, text_size=13,
                text_bold=False, font=BODY_FONT, align=PP_ALIGN.CENTER,
                anchor=MSO_ANCHOR.MIDDLE,
                shape_type=MSO_SHAPE.RECTANGLE):
    shape = slide.shapes.add_shape(shape_type, x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(0.5)
    shape.shadow.inherit = False
    if text_lines is not None:
        if isinstance(text_lines, str):
            text_lines = [text_lines]
        tf = shape.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = Inches(0.08)
        tf.margin_right = Inches(0.08)
        for i, line_text in enumerate(text_lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            run = p.add_run()
            run.text = line_text
            run.font.name = font
            run.font.size = Pt(text_size)
            run.font.bold = text_bold
            run.font.color.rgb = text_color
    return shape


def _eq_text(slide, x, y, w, h, equation, *, size=20, color=TEAL,
             align=PP_ALIGN.CENTER):
    """Render a Unicode/Consolas equation as a centred text line."""
    return _text(
        slide, x, y, w, h, equation,
        size=size, color=color, font=EQ_FONT, align=align,
        anchor=MSO_ANCHOR.MIDDLE,
    )


# ──────────────────────────────────────────────────────────────────
# Slide builders
# ──────────────────────────────────────────────────────────────────


TOTAL = 24


def slide_01_title(prs):
    s = _new_slide(prs, bg=BG_TITLE)
    # Subtle teal accent bar on the left
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.18), SLIDE_H)
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()
    bar.shadow.inherit = False

    _text(
        s, Inches(0.9), Inches(2.4), SLIDE_W - Inches(1.5), Inches(1.5),
        "Private Assets Frequency Upsampling:",
        size=42, bold=True, color=TEXT, font=TITLE_FONT,
    )
    _text(
        s, Inches(0.9), Inches(3.1), SLIDE_W - Inches(1.5), Inches(0.9),
        "Model Architecture & Methodology",
        size=42, bold=True, color=TEXT, font=TITLE_FONT,
    )
    _text(
        s, Inches(0.9), Inches(4.2), SLIDE_W - Inches(1.5), Inches(0.6),
        "Desmoothing  ·  Factor Decomposition  ·  Temporal Disaggregation",
        size=20, color=TEAL, font=BODY_FONT, italic=True,
    )
    _text(
        s, Inches(0.9), Inches(6.4), Inches(8), Inches(0.4),
        "Quantitative Research  —  Internal Use Only",
        size=12, color=TEXT_DIM, font=BODY_FONT,
    )


def slide_02_problem(prs):
    s = _new_slide(prs)
    _slide_title(s, "The problem we're solving",
                 subtitle="Reported indices smooth genuine market moves; risk numbers built on the raw series under-state σ by ~60%.")

    # Left column — observe vs need
    _text(s, Inches(0.55), Inches(1.85), Inches(6.0), Inches(0.4),
          "OBSERVE  vs.  NEED", size=13, bold=True, color=GOLD)

    obs_box_y = Inches(2.3)
    _filled_box(
        s, Inches(0.55), obs_box_y, Inches(2.4), Inches(0.55),
        fill=BG_PANEL, line=TEAL,
        text_lines="OBSERVE",
        text_color=TEAL, text_size=12, text_bold=True,
    )
    _eq_text(
        s, Inches(3.05), obs_box_y - Inches(0.05), Inches(3.6), Inches(0.65),
        "s_t  (smoothed, low-frequency)", size=18, color=TEXT, align=PP_ALIGN.LEFT,
    )

    need_box_y = Inches(3.1)
    _filled_box(
        s, Inches(0.55), need_box_y, Inches(2.4), Inches(0.55),
        fill=BG_PANEL, line=GREEN,
        text_lines="NEED",
        text_color=GREEN, text_size=12, text_bold=True,
    )
    _eq_text(
        s, Inches(3.05), need_box_y - Inches(0.05), Inches(3.6), Inches(0.65),
        "r_t  (true, high-frequency)", size=18, color=TEXT, align=PP_ALIGN.LEFT,
    )

    _text(
        s, Inches(0.55), Inches(4.2), Inches(6.5), Inches(2.5),
        [
            "VaR, CVaR, drawdown, capital ratios, factor exposure",
            "and cross-asset correlation analytics built on s_t are",
            "biased by construction.  None of the standard fixes",
            "(rolling windows, GARCH, copulas) recover the latent",
            "process — they assume the inputs are already correct.",
        ],
        size=14, color=TEXT_DIM, line_spacing=1.25,
    )

    # Right column — MSCI simulation table
    _text(
        s, Inches(7.4), Inches(1.85), Inches(5.4), Inches(0.4),
        "MSCI SIMULATION RESULT  (US Buyout vs. Russell 2000)",
        size=13, bold=True, color=GOLD,
    )
    headers = ["", "vol", "β", "ρ"]
    rows = [
        ("Raw observed",      "5.3%",  "0.30", "57%",  RED),
        ("Desmoothed (Bayes)", "12.6%", "1.01", "81%",  GREEN),
        ("True (latent)",     "12.5%", "1.00", "80%",  TEXT_DIM),
    ]
    th_y = Inches(2.4)
    th_x = Inches(7.4)
    cw = [Inches(2.5), Inches(0.95), Inches(0.95), Inches(0.95)]
    # Header row
    x = th_x
    for w, h in zip(cw, headers):
        _filled_box(
            s, x, th_y, w, Inches(0.4),
            fill=BG_TITLE, line=RULE,
            text_lines=h, text_color=GOLD, text_size=12, text_bold=True,
        )
        x += w
    y = th_y + Inches(0.4)
    for label, vol, beta, rho, color in rows:
        x = th_x
        cells = [label, vol, beta, rho]
        for i, (w, c) in enumerate(zip(cw, cells)):
            text_color = color if i == 0 else TEXT
            text_bold = (i == 0)
            _filled_box(
                s, x, y, w, Inches(0.55),
                fill=BG_PANEL, line=RULE,
                text_lines=c, text_color=text_color, text_size=14,
                text_bold=text_bold, align=PP_ALIGN.LEFT if i == 0 else PP_ALIGN.CENTER,
            )
            x += w
        y += Inches(0.55)

    _result_callout(
        s, Inches(7.4), Inches(5.4), Inches(5.4), Inches(1.4),
        [
            "The Bayesian desmoother recovers the latent risk to within",
            "sampling error.  The raw series misses 60% of the volatility,",
            "70% of the beta, and 24 ρ-points of public-market correlation.",
        ],
        title="The result we want to reproduce",
    )

    _footer(s, 2, TOTAL)


def slide_03_pipeline(prs):
    s = _new_slide(prs)
    _slide_title(s, "Pipeline architecture",
                 subtitle="Five sequential stages.  Each stage is a pluggable component; each emits a validation gate.")

    # 5-stage row of boxes
    stages = [
        ("STAGE 0", "Preprocessing",
         "Carry / MTM split (credit)\nReporting-lag fix (HF)\nPass-through (PE/RE/Infra)",
         GOLD),
        ("STAGE 1", "Desmoothing",
         "AR(1) Bayesian → PE / RE / Infra\nMA(q) GLM → Hedge funds\nThreshold AR(1) → Credit",
         TEAL),
        ("STAGE 2", "Factor decomposition",
         "OLS / WLS regression\nResidual diagnostics\nDistribution fitting",
         TEAL),
        ("STAGE 3", "Temporal disaggregation",
         "Chow-Lin AR(1) residuals\nFernández (RW)\nLitterman (Δu AR(1))",
         TEAL),
        ("STAGE 4", "Daily extension",
         "Mode A: Kalman smoother\nMode B: Monte Carlo\nBusiness-day calendar",
         TEAL),
    ]
    y_top = Inches(1.85)
    box_w = Inches(2.42)
    box_h = Inches(2.4)
    gap = Inches(0.10)
    x = Inches(0.55)
    for code, title, body, color in stages:
        # header
        _filled_box(
            s, x, y_top, box_w, Inches(0.5),
            fill=color, line=color,
            text_lines=code, text_color=BG_TITLE, text_size=12, text_bold=True,
        )
        # title
        _filled_box(
            s, x, y_top + Inches(0.5), box_w, Inches(0.5),
            fill=BG_PANEL, line=color,
            text_lines=title, text_color=TEXT, text_size=15, text_bold=True,
        )
        # body
        _filled_box(
            s, x, y_top + Inches(1.0), box_w, Inches(1.4),
            fill=BG_PANEL, line=RULE,
            text_lines=body.split("\n"), text_color=TEXT_DIM, text_size=11,
            anchor=MSO_ANCHOR.MIDDLE,
        )
        x += box_w + gap

    # Arrows between stages (small triangles between boxes)
    arrow_y = y_top + Inches(0.6)
    cx = Inches(0.55) + box_w
    for _ in range(4):
        tri = s.shapes.add_shape(
            MSO_SHAPE.RIGHT_ARROW,
            cx + Inches(-0.04),
            arrow_y - Inches(0.02),
            gap + Inches(0.08),
            Inches(0.18),
        )
        tri.fill.solid()
        tri.fill.fore_color.rgb = TEAL
        tri.line.color.rgb = TEAL
        tri.shadow.inherit = False
        cx += box_w + gap

    # Validation strip
    _filled_box(
        s, Inches(0.55), Inches(4.55), SLIDE_W - Inches(1.1), Inches(0.55),
        fill=BG_TITLE, line=GOLD,
        text_lines=[
            "INTER-STAGE VALIDATION   ·   Stage 1→2: σ desmoothed > σ observed   "
            "·   Stage 2→3: residuals ⊥ factors   ·   Stage 3→4: ∏(1+r_m) = 1+r_Q  within  10⁻¹⁰"
        ],
        text_color=TEXT, text_size=11, text_bold=False, font=EQ_FONT,
    )
    # Sanity-anchor strip
    _filled_box(
        s, Inches(0.55), Inches(5.2), SLIDE_W - Inches(1.1), Inches(0.55),
        fill=BG_TITLE, line=TEAL,
        text_lines=[
            "POST-PIPELINE SANITY ANCHORS   ·   vol-ratio ∈ [0.5×, 3.0×]  proxy   "
            "·   ρ ∈ [0.4, 0.95]   ·   crisis DD ≥ 50%·proxy   ·   β within leverage band"
        ],
        text_color=TEXT, text_size=11, text_bold=False, font=EQ_FONT,
    )

    # Posterior propagation strip
    _filled_box(
        s, Inches(0.55), Inches(5.85), SLIDE_W - Inches(1.1), Inches(0.55),
        fill=BG_TITLE, line=GREEN,
        text_lines=[
            "UNCERTAINTY MODE = 'full'   ·   N posterior draws of (λ, β, σ_ε)  →  "
            "pointwise 5/25/50/75/95 percentile bands on disaggregated output"
        ],
        text_color=TEXT, text_size=11, text_bold=False, font=EQ_FONT,
    )

    _text(
        s, Inches(0.55), Inches(6.55), SLIDE_W - Inches(1.1), Inches(0.4),
        "FallbackPolicy ∈ {strict, warn, auto} controls escalation of any flagged warning.",
        size=12, color=TEXT_DIM, italic=True,
    )

    _footer(s, 3, TOTAL)


def slide_04_notation(prs):
    s = _new_slide(prs)
    _slide_title(s, "Notation",
                 subtitle="Defined once.  Subscripts denote time;  superscripts denote frequency or component.")

    # Two columns of definitions
    items_left = [
        ("s_t",          "observed (smoothed) return at period t"),
        ("r_t",          "true (latent) return at period t"),
        ("P_t",          "reported valuation (NAV)"),
        ("V_t",          "true (unobservable) market valuation"),
        ("λ",            "AR(1) smoothing parameter ∈ [0, 1)"),
        ("θ_j",          "MA(q) weight at lag j  (Σ θ_j = 1, θ_j ≥ 0)"),
        ("φ",            "AR(1) persistence of daily idiosyncratic"),
    ]
    items_right = [
        ("β_k",          "factor loading on factor k"),
        ("F_{k,t}",      "return of factor k at time t"),
        ("α",            "intercept (annualised alpha)"),
        ("ε_t",          "idiosyncratic return"),
        ("σ_ε",          "idiosyncratic volatility"),
        ("y_Q  / y_m",   "low / high-frequency aggregated returns"),
        ("C, V",         "aggregation matrix and residual covariance"),
    ]

    def _draw_col(items, x_left):
        y = Inches(1.9)
        for sym, desc in items:
            _filled_box(
                s, x_left, y, Inches(1.3), Inches(0.55),
                fill=BG_PANEL, line=TEAL,
                text_lines=sym, text_color=TEAL, text_size=15, text_bold=True,
                font=EQ_FONT,
            )
            _text(
                s, x_left + Inches(1.45), y + Inches(0.05),
                Inches(4.6), Inches(0.5),
                desc, size=14, color=TEXT, anchor=MSO_ANCHOR.MIDDLE,
            )
            y += Inches(0.65)

    _draw_col(items_left, Inches(0.55))
    _draw_col(items_right, Inches(7.0))

    # Footer note
    _text(
        s, Inches(0.55), Inches(6.6), SLIDE_W - Inches(1.1), Inches(0.4),
        "Bold-face vectors denote the K-dimensional factor / coefficient stack.  "
        "Frequency superscripts (annual / monthly / daily) appear where ambiguity matters.",
        size=12, color=TEXT_DIM, italic=True,
    )
    _footer(s, 4, TOTAL)


def slide_05_smoothing_process(prs):
    s = _new_slide(prs)
    _slide_title(s, "Appraisal smoothing as AR(1)",
                 subtitle="Stage 1, Model 1.  PE / Infrastructure / Real Estate.")

    # Equation panels
    _text(s, Inches(0.55), Inches(1.85), Inches(7), Inches(0.4),
          "Valuation dynamics", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(2.25), Inches(7.7), Inches(1.0),
        EQ["smoothing_value"],
    )

    _text(s, Inches(0.55), Inches(3.4), Inches(7), Inches(0.4),
          "Implied return relationship", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(3.8), Inches(7.7), Inches(1.0),
        EQ["smoothing_return"],
    )

    _text(s, Inches(0.55), Inches(4.95), Inches(7), Inches(0.4),
          "Inversion (Geltner desmoothing)", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(5.35), Inches(7.7), Inches(1.4),
        EQ["desmooth"],
    )

    # Right column — assumptions + breaks
    assum_x = Inches(8.55)
    _text(s, assum_x, Inches(1.85), Inches(4.3), Inches(0.4),
          "Assumptions", size=13, bold=True, color=GOLD)

    assumption_rows = [
        ("A1", "Smoothing follows a first-order AR process on valuations."),
        ("A2", "λ is constant within an annual window; may vary seasonally across quarters."),
        ("A3", "True returns r_t are independent of the smoothing process itself."),
    ]
    y = Inches(2.3)
    for tag, body in assumption_rows:
        _assumption_tag(s, assum_x, y, tag)
        _text(
            s, assum_x + Inches(0.7), y - Inches(0.02),
            Inches(3.6), Inches(0.7),
            body, size=12, color=TEXT, line_spacing=1.15,
        )
        y += Inches(0.85)

    _breaks_callout(
        s, assum_x, Inches(4.95), Inches(4.3), Inches(1.85),
        [
            "λ is *not* constant in practice — appraisers exercise discretion;",
            "marking activity concentrates in Q4.  Extreme returns trigger non-",
            "linear marking behaviour (e.g., write-down events) that the AR(1)",
            "kernel cannot represent.",
        ],
    )

    _footer(s, 5, TOTAL)


def slide_06_rolling_annual(prs):
    s = _new_slide(prs)
    _slide_title(s, "Why rolling annual returns",
                 subtitle="Quarterly smoothing is seasonal — Q4 carries disproportionate marking activity.")

    # Equation
    _equation_panel(
        s, Inches(0.55), Inches(2.0), Inches(8.0), Inches(1.5),
        EQ["desmooth_annual"],
    )

    # Diagram of the four overlapping windows
    _text(s, Inches(0.55), Inches(3.7), Inches(7), Inches(0.4),
          "Four overlapping annual streams (each starts at a different quarter)",
          size=13, bold=True, color=GOLD)

    diagram_x = Inches(0.55)
    diagram_y = Inches(4.15)
    cell_w = Inches(0.55)
    quarter_labels = ["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"]
    n_q = len(quarter_labels)

    # Top row of quarter labels
    for i, q in enumerate(quarter_labels):
        x = diagram_x + cell_w * i
        _filled_box(
            s, x, diagram_y, cell_w, Inches(0.35),
            fill=BG_TITLE, line=RULE,
            text_lines=q, text_color=TEXT_DIM, text_size=10,
        )

    # Four offset rows
    offsets = [(0, "offset 0:  Q1→Q4"),
               (1, "offset 1:  Q2→Q1"),
               (2, "offset 2:  Q3→Q2"),
               (3, "offset 3:  Q4→Q3")]
    color_list = [TEAL, GOLD, GREEN, RED]
    for k, (off, label) in enumerate(offsets):
        row_y = diagram_y + Inches(0.4) + Inches(0.45) * k
        for i in range(off, off + 4):
            x = diagram_x + cell_w * i
            _filled_box(
                s, x, row_y, cell_w, Inches(0.35),
                fill=color_list[k], line=color_list[k],
                text_lines="•", text_color=BG_TITLE, text_size=12, text_bold=True,
            )
        _text(
            s, diagram_x + cell_w * n_q + Inches(0.15),
            row_y - Inches(0.02), Inches(3.0), Inches(0.4),
            label, size=12, color=color_list[k], font=EQ_FONT,
            anchor=MSO_ANCHOR.MIDDLE,
        )

    # Right column — motivation
    _text(s, Inches(9.0), Inches(1.85), Inches(3.8), Inches(0.4),
          "Why this matters", size=13, bold=True, color=GOLD)
    _bullets(
        s, Inches(9.0), Inches(2.25), Inches(3.8), Inches(4.0),
        [
            "Quarterly λ has Q4 bias from year-end marking.",
            "An annual aggregation smooths the seasonal contamination.",
            "Each offset stream is approximately iid within itself.",
            "Across streams: 3-quarter overlap → 4× effective coverage at the cost of induced autocorrelation.",
            "ESS reported in diagnostics, not naïve N.",
        ],
        size=13, color=TEXT, space_after=8,
    )

    _footer(s, 6, TOTAL)


def slide_07_single_step(prs):
    s = _new_slide(prs)
    _slide_title(s, "Single-step vs. two-step estimation",
                 subtitle="Estimation choice has first-order impact on β bias and σ recovery.")

    # Two-column comparison
    # Two-step
    _filled_box(
        s, Inches(0.55), Inches(1.85), Inches(6.0), Inches(0.55),
        fill=BG_PANEL, line=RED,
        text_lines="TWO-STEP   (estimate λ, then regress)",
        text_color=RED, text_size=14, text_bold=True,
        align=PP_ALIGN.LEFT,
    )
    _bullets(
        s, Inches(0.55), Inches(2.5), Inches(6.0), Inches(2.2),
        [
            "Step 1:  estimate λ̂ from autocorrelation alone.",
            "Step 2:  desmooth using λ̂, regress r_t on F_t.",
            "Errors in λ̂ propagate non-linearly into β.",
            "Risk numbers are biased upward because residual variance absorbs the λ-error.",
        ],
        size=13, color=TEXT_DIM,
    )
    _text(
        s, Inches(0.55), Inches(4.6), Inches(6.0), Inches(0.4),
        "MSCI sim:  RMSE(β̂) ≈ 0.35",
        size=15, color=RED, font=EQ_FONT, bold=True,
    )

    # Single-step
    _filled_box(
        s, Inches(6.85), Inches(1.85), Inches(6.0), Inches(0.55),
        fill=BG_PANEL, line=GREEN,
        text_lines="SINGLE-STEP   (joint estimation)",
        text_color=GREEN, text_size=14, text_bold=True,
        align=PP_ALIGN.LEFT,
    )
    _bullets(
        s, Inches(6.85), Inches(2.5), Inches(6.0), Inches(2.2),
        [
            "Treat (λ, β, α, σ_ε) as one parameter vector.",
            "Likelihood evaluated jointly on the AR(1)-implied desmoothed series.",
            "Posterior captures the λ↔β trade-off explicitly.",
            "No noise amplification from a fixed λ̂.",
        ],
        size=13, color=TEXT_DIM,
    )
    _text(
        s, Inches(6.85), Inches(4.6), Inches(6.0), Inches(0.4),
        "MSCI sim:  RMSE(β̂) ≈ 0.15",
        size=15, color=GREEN, font=EQ_FONT, bold=True,
    )

    # Joint regression equation
    _text(s, Inches(0.55), Inches(5.2), Inches(7), Inches(0.4),
          "The joint regression at each λ on the grid", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(5.6), SLIDE_W - Inches(1.1), Inches(1.0),
        EQ["factor_model"],
    )

    # Assumption tag
    _assumption_tag(s, SLIDE_W - Inches(1.4), Inches(6.7), "A4")
    _text(
        s, Inches(0.55), Inches(6.7), Inches(11.0), Inches(0.4),
        "Factor loadings β are constant over the estimation window.",
        size=12, color=TEXT_DIM, italic=True,
    )

    _footer(s, 7, TOTAL)


def slide_08_bayesian(prs):
    s = _new_slide(prs)
    _slide_title(s, "Bayesian estimation framework",
                 subtitle="Grid integration over λ;  conjugate Normal-Inverse-Gamma posterior on (β, α, σ_ε) at each grid point.")

    # Joint posterior
    _text(s, Inches(0.55), Inches(1.85), Inches(7), Inches(0.4),
          "Joint posterior (kernel)", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(2.25), Inches(12.2), Inches(1.0),
        EQ["posterior_kernel"],
    )

    # Marginalisation
    _text(s, Inches(0.55), Inches(3.4), Inches(7), Inches(0.4),
          "Marginal posterior on λ", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(3.8), Inches(12.2), Inches(1.0),
        EQ["posterior_marg"],
    )

    # Priors table on the left
    _text(s, Inches(0.55), Inches(4.95), Inches(6.0), Inches(0.4),
          "Priors (default presets)", size=13, bold=True, color=GOLD)
    prior_rows = [
        ("λ",        "Beta(2, 2)  on  [0, 1]"),
        ("β",        "𝒩(μ_β, σ²_β)  — strategy-specific"),
        ("α",        "𝒩(0, 0.05²)"),
        ("σ_ε",      "Inv-Gamma(3, 0.02)"),
    ]
    y = Inches(5.4)
    for sym, body in prior_rows:
        _filled_box(s, Inches(0.55), y, Inches(0.7), Inches(0.35),
                    fill=BG_PANEL, line=TEAL,
                    text_lines=sym, text_color=TEAL, text_size=14,
                    text_bold=True, font=EQ_FONT)
        _text(s, Inches(1.35), y - Inches(0.02), Inches(5.0), Inches(0.4),
              body, size=13, color=TEXT, font=EQ_FONT,
              anchor=MSO_ANCHOR.MIDDLE)
        y += Inches(0.4)

    # Computation strategy on the right
    _text(s, Inches(7.4), Inches(4.95), Inches(5.4), Inches(0.4),
          "Computation strategy", size=13, bold=True, color=GOLD)
    _bullets(
        s, Inches(7.4), Inches(5.4), Inches(5.4), Inches(1.7),
        [
            "Grid over λ:  50 points on [0.01, 0.95].",
            "Conditional posterior of (β, α, σ²) at each λ:  Normal-Inv-Gamma → closed form.",
            "Jacobian factor  −n·log(1−λ)  enters the marginal.",
            "Trapezoidal quadrature over the λ grid.",
            "Deterministic, exact up to grid resolution.  No MCMC.",
        ],
        size=12, color=TEXT,
    )

    _assumption_tag(s, Inches(0.55), Inches(7.0), "A5")
    _text(
        s, Inches(1.25), Inches(7.0), Inches(11.0), Inches(0.4),
        "Conjugate NIG structure is appropriate for the regression residuals  (i.i.d. Gaussian).",
        size=12, color=TEXT_DIM, italic=True,
    )

    _footer(s, 8, TOTAL)


def slide_09_vasicek(prs):
    s = _new_slide(prs)
    _slide_title(s, "Vasicek shrinkage & induced priors",
                 subtitle="Bayesian estimation as data-driven shrinkage; cross-sectional pooling across peer strategies.")

    # Vasicek formula
    _text(s, Inches(0.55), Inches(1.85), Inches(7), Inches(0.4),
          "Bayesian fit ≡ Vasicek shrinkage", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(2.25), Inches(8.0), Inches(1.0),
        EQ["vasicek"],
    )
    _eq_text(
        s, Inches(0.55), Inches(3.35), Inches(8.0), Inches(0.5),
        "w  ≡  signal-to-noise weight  =  σ²_prior / (σ²_prior + σ²_OLS)",
        size=14, color=TEAL, align=PP_ALIGN.LEFT,
    )

    # Induced-prior loop
    _text(s, Inches(0.55), Inches(4.0), Inches(7), Inches(0.4),
          "Induced prior loop", size=13, bold=True, color=GOLD)

    loop_steps = [
        ("1", "Fit each peer strategy independently with the user prior."),
        ("2", "Empirical cross-section gives mean / std per parameter."),
        ("3", "Use the empirical distribution as the new prior."),
        ("4", "Re-fit and iterate until ‖Δμ‖∞ < tol  or  10 iterations."),
    ]
    y = Inches(4.45)
    for n, body in loop_steps:
        _filled_box(s, Inches(0.55), y, Inches(0.4), Inches(0.4),
                    fill=GREEN, line=GREEN,
                    text_lines=n, text_color=BG_TITLE,
                    text_size=14, text_bold=True)
        _text(s, Inches(1.05), y - Inches(0.02), Inches(7.5), Inches(0.45),
              body, size=13, color=TEXT, anchor=MSO_ANCHOR.MIDDLE)
        y += Inches(0.45)

    # Right column — strategy prior table
    _text(s, Inches(9.0), Inches(1.85), Inches(3.8), Inches(0.4),
          "Strategy-specific β priors", size=13, bold=True, color=GOLD)
    rows = [
        ("Large Buyout",     "𝒩(1.15, 0.5²)"),
        ("Early Venture",    "𝒩(0.83, 0.25²)"),
        ("Mezzanine",        "𝒩(1.00, 0.5²)"),
        ("Distressed",       "𝒩(1.00, 0.5²)"),
        ("Infra Core",       "𝒩(0.50, 0.3²)"),
        ("Direct Lending",   "𝒩(0.70, 0.3²)"),
        ("Eq L/S (HF)",      "𝒩(0.40, 0.2²)"),
        ("Real Estate Core", "𝒩(0.60, 0.3²)"),
    ]
    y = Inches(2.3)
    for label, body in rows:
        _filled_box(s, Inches(9.0), y, Inches(2.05), Inches(0.36),
                    fill=BG_PANEL, line=RULE,
                    text_lines=label, text_color=TEXT, text_size=11,
                    text_bold=True, align=PP_ALIGN.LEFT)
        _filled_box(s, Inches(11.10), y, Inches(1.75), Inches(0.36),
                    fill=BG_PANEL, line=RULE,
                    text_lines=body, text_color=TEAL, text_size=11,
                    font=EQ_FONT)
        y += Inches(0.41)

    _assumption_tag(s, Inches(0.55), Inches(7.0), "A6")
    _text(
        s, Inches(1.25), Inches(7.0), Inches(11.0), Inches(0.4),
        "Peer-group structure is known and stable  (strategies correctly classified).",
        size=12, color=TEXT_DIM, italic=True,
    )

    _footer(s, 9, TOTAL)


def slide_10_identifiability(prs):
    s = _new_slide(prs)
    _slide_title(s, "Identifiability & diagnostics",
                 subtitle="The λ ↔ β confound is the central fragility.  We detect it; we do not pretend it isn't there.")

    # Top: the core challenge
    _filled_box(
        s, Inches(0.55), Inches(1.85), SLIDE_W - Inches(1.1), Inches(1.0),
        fill=BG_PANEL, line=GOLD,
        text_lines=[
            "Multiple (λ, β) pairs explain the same observed series.",
            "Likelihood surface is shallow along a ridge in λ-direction → posterior may be prior-driven.",
        ],
        text_color=TEXT, text_size=14,
    )

    # Diagnostics table
    _text(s, Inches(0.55), Inches(3.05), Inches(7), Inches(0.4),
          "Diagnostics shipped with every fit", size=13, bold=True, color=GOLD)
    diags = [
        ("CI(λ) width",  "warn if  CI₉₀(λ) > 0.4"),
        ("KL(post‖prior)", "warn if  < 0.1 nats   (data uninformative)"),
        ("β stability",  "warn if sign change OR > 50% magnitude swing across λ grid"),
        ("ESS",          "report effective sample size after autocorrelation correction"),
    ]
    y = Inches(3.5)
    for label, body in diags:
        _filled_box(s, Inches(0.55), y, Inches(2.7), Inches(0.5),
                    fill=BG_PANEL, line=TEAL,
                    text_lines=label, text_color=TEAL, text_size=13,
                    text_bold=True, font=EQ_FONT)
        _text(s, Inches(3.4), y - Inches(0.02), Inches(9.4), Inches(0.55),
              body, size=13, color=TEXT, font=EQ_FONT,
              anchor=MSO_ANCHOR.MIDDLE)
        y += Inches(0.55)

    _breaks_callout(
        s, Inches(0.55), Inches(5.85), SLIDE_W - Inches(1.1), Inches(1.1),
        [
            "Short histories  (T < 40 quarters)  are essentially uninformative — posterior collapses onto the prior.",
            "Strategies with extreme λ (≈ 0 or ≈ 1) are the worst-conditioned;  uncertainty_mode='full' is mandatory there.",
            "Periods where public/private correlation genuinely changes will trigger β instability flags;  this is correct behaviour.",
        ],
    )

    _footer(s, 10, TOTAL)


def slide_11_ma_intro(prs):
    s = _new_slide(prs)
    _slide_title(s, "Hedge fund smoothing is MA(q), not AR(1)",
                 subtitle="Stage 1, Model 2.  Getmansky, Lo & Makarov (2004).")

    _text(s, Inches(0.55), Inches(1.85), Inches(7), Inches(0.4),
          "Smoothing equation", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(2.25), SLIDE_W - Inches(1.1), Inches(1.0),
        EQ["ma_smoothing"],
    )

    # PE vs HF contrast
    _filled_box(
        s, Inches(0.55), Inches(3.5), Inches(6.0), Inches(0.5),
        fill=BG_PANEL, line=GOLD,
        text_lines="PE  (AR on valuations — catch-up dynamic)",
        text_color=GOLD, text_size=13, text_bold=True,
        align=PP_ALIGN.LEFT,
    )
    _filled_box(
        s, Inches(6.85), Inches(3.5), Inches(6.0), Inches(0.5),
        fill=BG_PANEL, line=TEAL,
        text_lines="HF  (MA on returns — staleness)",
        text_color=TEAL, text_size=13, text_bold=True,
        align=PP_ALIGN.LEFT,
    )
    _text(
        s, Inches(0.55), Inches(4.05), Inches(6.0), Inches(0.5),
        "P_t  =  P_{t-1}  +  (1−λ)(V_t − P_{t-1})",
        size=14, color=TEXT, font=EQ_FONT,
    )
    _text(
        s, Inches(6.85), Inches(4.05), Inches(6.0), Inches(0.5),
        "s_t  =  Σ_{j=0..q}  θ_j r_{t-j}",
        size=14, color=TEXT, font=EQ_FONT,
    )

    # Typical θ profiles
    _text(s, Inches(0.55), Inches(4.7), Inches(7), Inches(0.4),
          "Typical θ profiles", size=13, bold=True, color=GOLD)
    profile_rows = [
        ("Equity L/S, Macro  (q=2)",  "θ ≈ [0.70, 0.20, 0.10]"),
        ("Relative Value  (q=3)",     "θ ≈ [0.55, 0.25, 0.15, 0.05]"),
        ("Event-Driven  (q=4)",       "θ ≈ [0.40, 0.25, 0.20, 0.10, 0.05]"),
    ]
    y = Inches(5.15)
    for label, body in profile_rows:
        _filled_box(s, Inches(0.55), y, Inches(4.0), Inches(0.4),
                    fill=BG_PANEL, line=RULE,
                    text_lines=label, text_color=TEXT, text_size=12,
                    align=PP_ALIGN.LEFT)
        _filled_box(s, Inches(4.65), y, Inches(8.2), Inches(0.4),
                    fill=BG_PANEL, line=RULE,
                    text_lines=body, text_color=TEAL, text_size=12, font=EQ_FONT,
                    align=PP_ALIGN.LEFT)
        y += Inches(0.45)

    _assumption_tag(s, Inches(0.55), Inches(6.65), "A7")
    _assumption_tag(s, Inches(1.20), Inches(6.65), "A8")
    _text(
        s, Inches(1.95), Inches(6.65), Inches(11.0), Inches(0.4),
        "Reported returns are weighted lagged sums of true returns;  weights non-negative and sum to 1.",
        size=12, color=TEXT_DIM, italic=True,
    )

    _footer(s, 11, TOTAL)


def slide_12_ma_inversion(prs):
    s = _new_slide(prs)
    _slide_title(s, "MA(q) inversion & estimation",
                 subtitle="Recovery via banded Toeplitz inversion; Bayesian estimation on the ordered simplex.")

    _text(s, Inches(0.55), Inches(1.85), Inches(7), Inches(0.4),
          "Matrix recovery", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(2.25), Inches(7.5), Inches(1.0),
        EQ["ma_inversion"],
    )

    _text(s, Inches(0.55), Inches(3.35), Inches(7), Inches(0.4),
          "Identification from autocovariances", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(3.75), Inches(12.2), Inches(1.0),
        EQ["ma_autocov"],
    )

    # Two columns: Bayesian & numerics
    _text(s, Inches(0.55), Inches(4.95), Inches(6.0), Inches(0.4),
          "Bayesian extension", size=13, bold=True, color=GOLD)
    _bullets(
        s, Inches(0.55), Inches(5.4), Inches(6.0), Inches(1.6),
        [
            "Ordered Dirichlet prior:  θ_0 ≥ θ_1 ≥ ... ≥ θ_q.",
            "Default concentration  c = (5, 3, 2, 1)  truncated to length q+1.",
            "q ≤ 3:  grid integration on the ordered simplex.",
            "q > 3:  Laplace approximation around the MAP.",
        ],
        size=12, color=TEXT,
    )

    _text(s, Inches(7.0), Inches(4.95), Inches(6.0), Inches(0.4),
          "Numerical stability", size=13, bold=True, color=GOLD)
    _bullets(
        s, Inches(7.0), Inches(5.4), Inches(6.0), Inches(1.6),
        [
            "Hard floor:  θ_0 ≥ 0.2.",
            "Exponential-decay taper on higher lags.",
            "Monitor cond(Θ);  warn if  > 100.",
            "Posterior θ_0 < 0.3  trips the same warning.",
        ],
        size=12, color=TEXT,
    )

    _assumption_tag(s, Inches(0.55), Inches(7.0), "A9")
    _text(
        s, Inches(1.25), Inches(7.0), Inches(11.0), Inches(0.4),
        "Number of lags q is fixed per strategy (or selected via BIC) — not a free parameter at fit time.",
        size=12, color=TEXT_DIM, italic=True,
    )
    _footer(s, 12, TOTAL)


def slide_13_ma_fragility(prs):
    s = _new_slide(prs)
    _slide_title(s, "MA(q) fragility & mitigations",
                 subtitle="Inversion noise amplifies exponentially with q.  We build the safety net before the user trips on it.")

    # RMSE-vs-q diagram
    _text(s, Inches(0.55), Inches(1.85), Inches(7), Inches(0.4),
          "RMSE(θ̂)  vs  q  (illustrative)", size=13, bold=True, color=GOLD)

    chart_x = Inches(0.55)
    chart_y = Inches(2.30)
    chart_w = Inches(6.6)
    chart_h = Inches(3.6)

    # Chart background
    _filled_box(
        s, chart_x, chart_y, chart_w, chart_h,
        fill=BG_PANEL, line=RULE,
    )
    # Axes labels
    _text(s, chart_x, chart_y + chart_h + Inches(0.05),
          chart_w, Inches(0.3),
          "q  (number of MA lags)",
          size=11, color=TEXT_DIM, align=PP_ALIGN.CENTER)
    _text(s, chart_x - Inches(0.65), chart_y + chart_h / 2 - Inches(0.2),
          Inches(0.6), Inches(0.4),
          "RMSE", size=11, color=TEXT_DIM, align=PP_ALIGN.RIGHT,
          anchor=MSO_ANCHOR.MIDDLE)

    # Plot 3 curves (T=80, T=150, T=300) — qualitative
    qs = [1, 2, 3, 4, 5]
    curves = [
        ("T=80",  [0.18, 0.27, 0.45, 0.95, 1.40], RED),
        ("T=150", [0.12, 0.18, 0.30, 0.55, 0.95], GOLD),
        ("T=300", [0.08, 0.12, 0.18, 0.25, 0.40], GREEN),
    ]
    # Map to chart coords
    margin = Inches(0.4)
    plot_x0 = chart_x + margin
    plot_y0 = chart_y + margin
    plot_w = chart_w - 2 * margin
    plot_h = chart_h - 2 * margin
    max_q = max(qs)
    max_rmse = 1.5
    for label, ys, color in curves:
        for i in range(len(qs) - 1):
            x1 = plot_x0 + plot_w * (qs[i] - 1) / (max_q - 1)
            x2 = plot_x0 + plot_w * (qs[i+1] - 1) / (max_q - 1)
            y1 = plot_y0 + plot_h * (1.0 - ys[i] / max_rmse)
            y2 = plot_y0 + plot_h * (1.0 - ys[i+1] / max_rmse)
            line = s.shapes.add_connector(1, x1, y1, x2, y2)
            line.line.color.rgb = color
            line.line.width = Pt(2.4)
        # marker on the last point + label
        last_x = plot_x0 + plot_w * (qs[-1] - 1) / (max_q - 1)
        last_y = plot_y0 + plot_h * (1.0 - ys[-1] / max_rmse)
        _text(
            s, last_x + Inches(0.05), last_y - Inches(0.15),
            Inches(0.8), Inches(0.3),
            label, size=11, color=color, font=EQ_FONT, bold=True,
        )
    # x-axis ticks
    for q in qs:
        x = plot_x0 + plot_w * (q - 1) / (max_q - 1)
        _text(s, x - Inches(0.15), plot_y0 + plot_h + Inches(0.05),
              Inches(0.3), Inches(0.25),
              str(q), size=10, color=TEXT_DIM, align=PP_ALIGN.CENTER)

    # Right column — mitigations
    _text(s, Inches(7.5), Inches(1.85), Inches(5.3), Inches(0.4),
          "Mitigations layered into the fit", size=13, bold=True, color=GOLD)
    _bullets(
        s, Inches(7.5), Inches(2.3), Inches(5.3), Inches(3.6),
        [
            "Aggressive priors → ordered-Dirichlet keeps θ_0 large.",
            "Exponential-decay prior on tail lags.",
            "Cap at q ≤ 3 for grid integration.",
            "Auto-fallback chain:",
            ("cond(Θ) > 100  →  reduce q by 1.", 1),
            ("q = 1 still fails  →  fall back to AR(1).", 1),
            ("posterior θ_0 < 0.3  →  warn and surface in result.", 1),
        ],
        size=12, color=TEXT,
    )

    _breaks_callout(
        s, Inches(0.55), Inches(6.10), SLIDE_W - Inches(1.1), Inches(0.85),
        [
            "Above q=3 with T<150, identification is essentially gone.  The estimator can hit the prior mode and produce an artificially smooth θ — diagnostics will fire, but the user must act on them.",
        ],
    )
    _footer(s, 13, TOTAL)


def slide_14_threshold(prs):
    s = _new_slide(prs)
    _slide_title(s, "Threshold AR(1) — private credit",
                 subtitle="Stage 1, Model 3.  Two regimes selected by an exogenous indicator.")

    # Smoothing equation
    _text(s, Inches(0.55), Inches(1.85), Inches(7), Inches(0.4),
          "Smoothing process", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(2.25), SLIDE_W - Inches(1.1), Inches(1.0),
        EQ["threshold"],
    )

    # Regime table
    _text(s, Inches(0.55), Inches(3.4), Inches(7), Inches(0.4),
          "Regime semantics", size=13, bold=True, color=GOLD)
    rows = [
        ("Normal",  "λ_high  ∈  [0.5, 0.9]",
         "Loans carried at par;  minimal marking activity.", GOLD),
        ("Stress",  "λ_low  ∈  [0.0, 0.3]",
         "Forced marking to reality;  fast NAV adjustment.", RED),
    ]
    y = Inches(3.85)
    for name, lam, desc, color in rows:
        _filled_box(s, Inches(0.55), y, Inches(1.5), Inches(0.55),
                    fill=color, line=color,
                    text_lines=name, text_color=BG_TITLE, text_size=14,
                    text_bold=True)
        _filled_box(s, Inches(2.10), y, Inches(2.5), Inches(0.55),
                    fill=BG_PANEL, line=RULE,
                    text_lines=lam, text_color=TEAL, text_size=14,
                    font=EQ_FONT)
        _filled_box(s, Inches(4.65), y, Inches(8.2), Inches(0.55),
                    fill=BG_PANEL, line=RULE,
                    text_lines=desc, text_color=TEXT, text_size=12,
                    align=PP_ALIGN.LEFT)
        y += Inches(0.6)

    # Carry / MTM split
    _text(s, Inches(0.55), Inches(5.20), Inches(7), Inches(0.4),
          "Stage 0 carry / MTM decomposition", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(5.6), SLIDE_W - Inches(1.1), Inches(1.0),
        EQ["credit_decomp"],
    )

    _assumption_tag(s, Inches(0.55), Inches(6.85), "A10")
    _assumption_tag(s, Inches(1.20), Inches(6.85), "A11")
    _text(
        s, Inches(1.95), Inches(6.85), Inches(11.0), Inches(0.4),
        "Returns decomposable into deterministic carry + stochastic MTM;  regime is exogenous and observable.",
        size=12, color=TEXT_DIM, italic=True,
    )
    _footer(s, 14, TOTAL)


def slide_15_threshold_estim(prs):
    s = _new_slide(prs)
    _slide_title(s, "Threshold AR(1) estimation & limitations",
                 subtitle="2-D conjugate posterior;  every shortcut here has a cost we surface upfront.")

    # Estimation
    _text(s, Inches(0.55), Inches(1.85), Inches(6.0), Inches(0.4),
          "Estimation", size=13, bold=True, color=GOLD)
    _bullets(
        s, Inches(0.55), Inches(2.3), Inches(6.0), Inches(2.5),
        [
            "2-D grid over (λ_high, λ_low) — typically 30 × 30.",
            "At each pair: NIG conditional posterior on (β, α, σ_ε).",
            "Regime-aware Jacobian:",
            ("−n_n · log(1−λ_high)  −  n_s · log(1−λ_low)", 1),
            "Marginalise via 2-D trapezoidal quadrature.",
            "Priors:  λ_high ~ Beta(5,2),  λ_low ~ Beta(2,5).",
        ],
        size=12, color=TEXT,
    )

    # Limitations
    _text(s, Inches(7.0), Inches(1.85), Inches(5.8), Inches(0.4),
          "Where this breaks", size=13, bold=True, color=RED)
    _bullets(
        s, Inches(7.0), Inches(2.3), Inches(5.8), Inches(2.5),
        [
            "Real regimes are continuum, not binary.",
            "Exogenous threshold introduces look-ahead bias if calibrated in-sample.",
            "Carry/MTM split assumes y_t deterministic — only approximate.",
            "Stress-regime sample size is structurally small;  posterior is prior-driven.",
        ],
        size=12, color=TEXT_DIM,
    )

    # Fallback
    _filled_box(
        s, Inches(0.55), Inches(5.0), SLIDE_W - Inches(1.1), Inches(1.4),
        fill=BG_PANEL, line=GREEN,
        text_lines=[
            "FALLBACK CHAIN",
            "regime indicator missing  →  degrade to standard AR(1) on the total return  (lossy but defined).",
            "fewer than min_regime_obs in stress  →  warn and report posterior on λ_high only.",
        ],
        text_color=TEXT, text_size=13, anchor=MSO_ANCHOR.MIDDLE,
    )

    _assumption_tag(s, Inches(0.55), Inches(6.7), "A11")
    _text(
        s, Inches(1.25), Inches(6.7), Inches(11.0), Inches(0.4),
        "(re-emphasised) regime threshold is fixed and exogenous;  endogenous Markov-switching is out of scope.",
        size=12, color=TEXT_DIM, italic=True,
    )
    _footer(s, 15, TOTAL)


def slide_16_factor_decomp(prs):
    s = _new_slide(prs)
    _slide_title(s, "Stage 2 — factor decomposition",
                 subtitle="Post-desmoothing OLS / WLS;  residual diagnostics drive Stage 3 / Stage 4.")

    # Equation
    _text(s, Inches(0.55), Inches(1.85), Inches(7), Inches(0.4),
          "Decomposition", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(2.25), SLIDE_W - Inches(1.1), Inches(1.0),
        EQ["factor_decomp"],
    )

    # Three-column residual diagnostics
    diag_cols = [
        ("Autocorrelation",  "Ljung-Box  at lags  5, 10, 20",
         "Reject  ⇒  factor model missing a serially-correlated factor."),
        ("Heteroscedasticity", "ARCH-LM  at  nlags = 5",
         "Reject  ⇒  fit GARCH-style residuals at Stage 4 simulation."),
        ("Normality / tails", "Jarque-Bera  +  fit  𝒩 / t / skew-t (Hansen 94)",
         "Best-AIC distribution drives Stage 4 Mode B sampling."),
    ]
    x = Inches(0.55)
    cw = (SLIDE_W - Inches(1.3)) / 3
    for label, test, interp in diag_cols:
        _filled_box(s, x, Inches(3.5), cw - Inches(0.15), Inches(0.5),
                    fill=BG_TITLE, line=TEAL,
                    text_lines=label, text_color=TEAL, text_size=13,
                    text_bold=True)
        _filled_box(s, x, Inches(4.0), cw - Inches(0.15), Inches(0.5),
                    fill=BG_PANEL, line=RULE,
                    text_lines=test, text_color=TEXT, text_size=12,
                    font=EQ_FONT)
        _filled_box(s, x, Inches(4.5), cw - Inches(0.15), Inches(1.4),
                    fill=BG_PANEL, line=RULE,
                    text_lines=interp, text_color=TEXT_DIM, text_size=11)
        x += cw + Inches(0.05)

    # Cross-strategy block
    _filled_box(
        s, Inches(0.55), Inches(6.05), SLIDE_W - Inches(1.1), Inches(0.7),
        fill=BG_PANEL, line=GOLD,
        text_lines=[
            "Cross-strategy residual covariance Σ_ε  drives:  multivariate Stage 3 rotation,  joint Stage 4 simulation.",
        ],
        text_color=TEXT, text_size=13,
    )

    _assumption_tag(s, Inches(0.55), Inches(6.92), "A12")
    _assumption_tag(s, Inches(1.20), Inches(6.92), "A13")
    _text(
        s, Inches(1.95), Inches(6.92), Inches(11.0), Inches(0.4),
        "Factor loadings stable over the window;  residuals uncorrelated with factors  (else the model is misspecified).",
        size=12, color=TEXT_DIM, italic=True,
    )

    _footer(s, 16, TOTAL)


def slide_17_chow_lin(prs):
    s = _new_slide(prs)
    _slide_title(s, "Chow-Lin temporal disaggregation",
                 subtitle="GLS BLUE solution under a low-frequency aggregation constraint.")

    _text(s, Inches(0.55), Inches(1.85), Inches(7), Inches(0.4),
          "Closed-form GLS solution", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(2.25), SLIDE_W - Inches(1.1), Inches(1.5),
        EQ["chow_lin"],
    )

    # Three-method comparison
    methods = [
        ("Chow-Lin (1971)",  "u_m  ∼  AR(1)  with  ρ ∈ (−1, 1)",
         "Estimate ρ by profile log-likelihood grid."),
        ("Fernández (1981)",  "u_m  ∼  random walk   (ρ = 1)",
         "No ρ estimation;  V = L L′ (cumulative-sum)."),
        ("Litterman (1983)",  "Δu_m  ∼  AR(1)",
         "Litterman  ≡  Fernández  at  ρ = 0."),
    ]
    y = Inches(4.0)
    for name, model, est in methods:
        _filled_box(s, Inches(0.55), y, Inches(3.0), Inches(0.55),
                    fill=BG_PANEL, line=TEAL,
                    text_lines=name, text_color=TEAL, text_size=14,
                    text_bold=True)
        _filled_box(s, Inches(3.65), y, Inches(4.4), Inches(0.55),
                    fill=BG_PANEL, line=RULE,
                    text_lines=model, text_color=TEXT, text_size=12,
                    font=EQ_FONT, align=PP_ALIGN.LEFT)
        _filled_box(s, Inches(8.15), y, Inches(4.7), Inches(0.55),
                    fill=BG_PANEL, line=RULE,
                    text_lines=est, text_color=TEXT_DIM, text_size=11,
                    align=PP_ALIGN.LEFT)
        y += Inches(0.6)

    _assumption_tag(s, Inches(0.55), Inches(5.95), "A14")
    _assumption_tag(s, Inches(1.20), Inches(5.95), "A15")
    _text(
        s, Inches(1.95), Inches(5.95), Inches(11.0), Inches(0.4),
        "Monthly residual is stationary  (or unit-root)  AR(1);  link  y_m ↔ X_m  is linear.",
        size=12, color=TEXT_DIM, italic=True,
    )

    _result_callout(
        s, Inches(0.55), Inches(6.45), SLIDE_W - Inches(1.1), Inches(0.6),
        [
            "Validated:  the round-trip aggregation  ∏(1+r_m) = 1+r_Q  holds to  10⁻¹⁰  across all three methods.",
        ],
        title="Stage 3 → 4 gate",
    )

    _footer(s, 17, TOTAL)


def slide_18_aggregation(prs):
    s = _new_slide(prs)
    _slide_title(s, "Multiplicative vs. additive aggregation",
                 subtitle="The arithmetic-vs-geometric drift is material for PE-magnitude returns.")

    _text(s, Inches(0.55), Inches(1.85), Inches(7), Inches(0.4),
          "Aggregation constraint", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(2.25), SLIDE_W - Inches(1.1), Inches(1.0),
        EQ["agg_constraint"],
    )

    # Numerical illustration
    _text(s, Inches(0.55), Inches(3.4), Inches(7), Inches(0.4),
          "Numerical illustration  ·  Q4 2008,  r_Q = −22%", size=13, bold=True, color=GOLD)

    # Two columns
    _filled_box(
        s, Inches(0.55), Inches(3.85), Inches(6.0), Inches(0.5),
        fill=BG_PANEL, line=RED,
        text_lines="ADDITIVE  (linear approximation)",
        text_color=RED, text_size=13, text_bold=True, align=PP_ALIGN.LEFT,
    )
    _eq_text(s, Inches(0.55), Inches(4.45), Inches(6.0), Inches(0.5),
             "r_m,1  +  r_m,2  +  r_m,3  =  −22.0%", size=14, color=TEXT,
             align=PP_ALIGN.LEFT)
    _eq_text(s, Inches(0.55), Inches(4.95), Inches(6.0), Inches(0.5),
             "→  per-month  =  −7.33%", size=15, color=TEXT_DIM,
             align=PP_ALIGN.LEFT)

    _filled_box(
        s, Inches(6.85), Inches(3.85), Inches(6.0), Inches(0.5),
        fill=BG_PANEL, line=GREEN,
        text_lines="MULTIPLICATIVE  (exact)",
        text_color=GREEN, text_size=13, text_bold=True, align=PP_ALIGN.LEFT,
    )
    _eq_text(s, Inches(6.85), Inches(4.45), Inches(6.0), Inches(0.5),
             "(1+r_m,1)(1+r_m,2)(1+r_m,3) = 0.78", size=14, color=TEXT,
             align=PP_ALIGN.LEFT)
    _eq_text(s, Inches(6.85), Inches(4.95), Inches(6.0), Inches(0.5),
             "→  per-month  =  −7.94%", size=15, color=GREEN,
             align=PP_ALIGN.LEFT)

    # Drift commentary
    _filled_box(
        s, Inches(0.55), Inches(5.7), SLIDE_W - Inches(1.1), Inches(0.9),
        fill=BG_PANEL, line=GOLD,
        text_lines=[
            "Drift  =  61 bps per month  =  ~190 bps over the quarter.",
            "Material for VaR / capital;  default is multiplicative.  Algebra runs in log-space → exponentiate at output.",
        ],
        text_color=TEXT, text_size=13, anchor=MSO_ANCHOR.MIDDLE,
    )
    _footer(s, 18, TOTAL)


def slide_19_chow_lin_finance(prs):
    s = _new_slide(prs)
    _slide_title(s, "Chow-Lin: defending its use in finance",
                 subtitle="A common critique deserves a direct answer.")

    # The critique block
    _filled_box(
        s, Inches(0.55), Inches(1.85), SLIDE_W - Inches(1.1), Inches(0.85),
        fill=BG_PANEL, line=RED,
        text_lines=[
            "CRITIQUE  ·  \"Chow-Lin is a macro-econometrics tool.  Why are you using it on finance returns?\"",
        ],
        text_color=RED, text_size=14, anchor=MSO_ANCHOR.MIDDLE,
    )

    # The rebuttal block
    _text(s, Inches(0.55), Inches(2.95), Inches(7), Inches(0.4),
          "Rebuttal  ·  what makes it valid here", size=13, bold=True, color=GOLD)
    _bullets(
        s, Inches(0.55), Inches(3.4), SLIDE_W - Inches(1.1), Inches(2.0),
        [
            "The factor model gives us indicators.  PE return  =  β · F  +  ε.",
            "Linearity (A15) is satisfied by construction — we are not assuming it of the underlying economic process; we built it.",
            "Chow-Lin then distributes the residual ε under an AR(1) covariance — exactly what Stage 2 fits.",
            "Multiplicative aggregation (A18 implicit) is enforced via log-space algebra → no compounding drift.",
        ],
        size=13, color=TEXT,
    )

    # Where it really limits
    _breaks_callout(
        s, Inches(0.55), Inches(5.6), SLIDE_W - Inches(1.1), Inches(1.45),
        [
            "Chow-Lin distributes the residual *linearly*.  Factor exposures with",
            "option-like payoffs (deep-OTM VC, distressed credit equity-kicker)",
            "violate this — Stage 4 simulation Mode B is the right tool there.",
            "For 21:1 monthly-to-daily ratios the dense Toeplitz inversion is the slow leg;  banded scipy primitives are the recommended swap.",
        ],
    )
    _footer(s, 19, TOTAL)


def slide_20_kalman(prs):
    s = _new_slide(prs)
    _slide_title(s, "Stage 4 Mode A — Kalman smoother",
                 subtitle="Daily point estimate via state-space BLUE on the AR(1) idiosyncratic.")

    # State + observation eqs
    _text(s, Inches(0.55), Inches(1.85), Inches(7), Inches(0.4),
          "State-space model", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(2.25), Inches(7.0), Inches(1.0),
        EQ["kalman_state"],
    )
    _equation_panel(
        s, Inches(7.7), Inches(2.25), Inches(5.1), Inches(1.0),
        EQ["kalman_obs"],
    )

    # Body
    _bullets(
        s, Inches(0.55), Inches(3.5), Inches(12.2), Inches(2.5),
        [
            "Run forward filter + Rauch-Tung-Striebel smoother  →  ε̂_d = E[ε_d | all monthly observations].",
            "Equivalent in BLUE form:  ε̂ = V·C′(CVC′)⁻¹·ε^month  with V = AR(1) covariance.",
            "Total daily return:   r_d  =  β·F_d  +  α/n_d  +  ε̂_d.",
            "Multiplicative round-trip preserved exactly via log-space workspace.",
            "Conditional expectation — *not* a sampled realisation.  Use Mode B for VaR.",
        ],
        size=13, color=TEXT,
    )

    _assumption_tag(s, Inches(0.55), Inches(6.4), "A16")
    _assumption_tag(s, Inches(1.20), Inches(6.4), "A17")
    _text(
        s, Inches(1.95), Inches(6.4), Inches(11.0), Inches(0.4),
        "Daily idiosyncratic is stationary AR(1);  daily systematic = β · F_d  exactly  (no daily β drift).",
        size=12, color=TEXT_DIM, italic=True,
    )

    _breaks_callout(
        s, Inches(0.55), Inches(6.85), SLIDE_W - Inches(1.1), Inches(0.55),
        [
            "Memory:  dense  n × n  AR(1) covariance — fine to ~3000 days, blows up at 30-year daily;  switch to banded scipy primitives.",
        ],
    )
    _footer(s, 20, TOTAL)


def slide_21_simulation(prs):
    s = _new_slide(prs)
    _slide_title(s, "Stage 4 Mode B — Monte Carlo simulation",
                 subtitle="N constraint-respecting daily paths for VaR / CVaR / drawdown distributions.")

    _text(s, Inches(0.55), Inches(1.85), Inches(7), Inches(0.4),
          "One simulated path", size=13, bold=True, color=GOLD)
    _equation_panel(
        s, Inches(0.55), Inches(2.25), SLIDE_W - Inches(1.1), Inches(1.0),
        EQ["sim_path"],
    )

    # Procedure
    _text(s, Inches(0.55), Inches(3.4), Inches(7), Inches(0.4),
          "Procedure (per path j)", size=13, bold=True, color=GOLD)
    _bullets(
        s, Inches(0.55), Inches(3.85), Inches(7.6), Inches(3.0),
        [
            "Draw  (λ, β, σ_ε)  from joint posterior  →  uncertainty propagation.",
            "Generate daily systematic:   r_sys,d  =  β · F_d.",
            "Draw daily idiosyncratic from fitted distribution  𝒟  ∈ {𝒩, t, skew-t}, scaled to daily frequency.",
            "Cross-strategy correlation enforced via Cholesky on Σ_ε   (Gaussian copula).",
            "Within each month, subtract block-mean and add  ε^month / n_d  →  exact constraint.",
        ],
        size=12, color=TEXT,
    )

    # Right column
    _filled_box(
        s, Inches(8.4), Inches(3.85), Inches(4.4), Inches(2.7),
        fill=BG_PANEL, line=TEAL,
        text_lines=[
            "OUTPUT TENSOR",
            "",
            "paths   ∈   ℝ^(N × T_d × K)",
            "",
            "→  pointwise quantiles for VaR, CVaR",
            "→  drawdown distribution per path",
            "→  cross-strategy joint scenarios",
        ],
        text_color=TEXT, text_size=13, anchor=MSO_ANCHOR.MIDDLE,
    )

    _result_callout(
        s, Inches(0.55), Inches(6.85), SLIDE_W - Inches(1.1), Inches(0.55),
        [
            "Round-trip property:  every sampled path satisfies  ∏(1+r_d)_{d∈m} = 1+r_m  exactly.",
        ],
        title="Constraint",
    )
    _footer(s, 21, TOTAL)


def slide_22_asset_table(prs):
    s = _new_slide(prs)
    _slide_title(s, "Model selection by asset class",
                 subtitle="Each preset ships with strategy-specific factor priors and Stage 0 preprocessing.")

    headers = ["Asset class", "Smoothing model", "Key factors", "Typical λ / θ_0", "Special handling"]
    rows = [
        ("PE — Buyout",          "AR(1) Bayesian",   "Equity, Size",                  "λ ∈ [0.4, 0.7]",  "Rolling annual"),
        ("PE — Venture",         "AR(1) Bayesian",   "Small-cap growth",              "λ ∈ [0.5, 0.8]",  "Tighter β prior"),
        ("Infrastructure",       "AR(1) Bayesian",   "Utilities, Inflation, Duration", "λ ∈ [0.5, 0.8]", "Duration + inflation"),
        ("Private credit",       "Threshold AR(1)",  "Credit spreads, Rates",         "λ_high / λ_low",  "Carry / MTM split"),
        ("HF — Eq L/S",          "MA(q=2)",          "Equity, SMB, MOM",              "θ_0 ∈ [0.6, 0.8]","Reporting-lag fix"),
        ("HF — Event-driven",    "MA(q=4)*",         "Equity, Credit",                "θ_0 ∈ [0.3, 0.5]","Higher illiquidity"),
        ("Managed futures",      "NoSmoothing",      "Trend factors",                 "—",               "Exchange-traded"),
        ("Real estate",          "AR(1) Bayesian",   "REITs, Rates",                  "λ ∈ [0.6, 0.85]", "Q4 seasonality"),
    ]
    cw = [Inches(2.4), Inches(2.5), Inches(2.6), Inches(2.0), Inches(2.7)]
    table_x = Inches(0.55)
    table_y = Inches(1.85)

    # Header
    x = table_x
    for w, h in zip(cw, headers):
        _filled_box(s, x, table_y, w, Inches(0.45),
                    fill=BG_TITLE, line=TEAL,
                    text_lines=h, text_color=TEAL, text_size=12,
                    text_bold=True, align=PP_ALIGN.LEFT)
        x += w
    y = table_y + Inches(0.45)
    for i, row in enumerate(rows):
        bg = BG_CONTENT if i % 2 == 0 else BG_PANEL
        x = table_x
        for j, (w, cell) in enumerate(zip(cw, row)):
            color = TEXT if j > 0 else GOLD
            _filled_box(s, x, y, w, Inches(0.5),
                        fill=bg, line=RULE,
                        text_lines=cell, text_color=color, text_size=11,
                        align=PP_ALIGN.LEFT,
                        text_bold=(j == 0),
                        font=EQ_FONT if j in (3,) else BODY_FONT)
            x += w
        y += Inches(0.5)

    _text(
        s, Inches(0.55), Inches(6.6), SLIDE_W - Inches(1.1), Inches(0.4),
        "* This build's grid integration supports q ≤ 3;  Event-Driven preset is pinned to q=3 until Laplace mode ships.",
        size=11, color=TEXT_DIM, italic=True,
    )
    _footer(s, 22, TOTAL)


def slide_23_assumptions(prs):
    s = _new_slide(prs)
    _slide_title(s, "Assumption register",
                 subtitle="Every claim made anywhere in this deck.  Photograph this slide.")

    headers = ["#", "Assumption", "Where used", "Severity if violated"]
    rows = [
        ("A1",  "Smoothing follows AR(1) on valuations",
                "Stage 1, AR(1)",                      "moderate", GOLD),
        ("A2",  "λ constant within annual window",
                "Stage 1, AR(1)",                      "moderate", GOLD),
        ("A3",  "True returns independent of smoother",
                "Stage 1, AR(1)",                      "moderate", GOLD),
        ("A4",  "β constant over estimation window",
                "Stage 1 + Stage 2",                   "critical",  RED),
        ("A5",  "NIG conjugacy fits residuals",
                "Stage 1 Bayesian",                    "minor",     GREEN),
        ("A6",  "Peer group structure stable",
                "Induced priors",                      "moderate", GOLD),
        ("A7",  "HF returns are MA(q) of true returns",
                "Stage 1, MA(q)",                      "moderate", GOLD),
        ("A8",  "θ_j ≥ 0 and Σ θ_j = 1",
                "Stage 1, MA(q)",                      "minor",     GREEN),
        ("A9",  "q is known (or BIC-selected)",
                "Stage 1, MA(q)",                      "moderate", GOLD),
        ("A10", "Carry / MTM additive decomposition",
                "Stage 0 credit",                      "moderate", GOLD),
        ("A11", "Regime is exogenous and observable",
                "Stage 1 threshold",                   "critical",  RED),
        ("A12", "Factor loadings stable over window",
                "Stage 2",                             "critical",  RED),
        ("A13", "Residuals ⊥ factors",
                "Stage 2 → Stage 3",                   "critical",  RED),
        ("A14", "Stage 3 residual ∼ AR(1)",
                "Stage 3 Chow-Lin",                    "moderate", GOLD),
        ("A15", "Linear y_m ↔ X_m relationship",
                "Stage 3 Chow-Lin",                    "moderate", GOLD),
        ("A16", "Daily idio ∼ stationary AR(1)",
                "Stage 4A Kalman",                     "minor",     GREEN),
        ("A17", "r_sys,d  =  β · F_d  exactly",
                "Stage 4A Kalman",                     "moderate", GOLD),
    ]

    cw = [Inches(0.7), Inches(4.4), Inches(3.4), Inches(3.7)]
    table_x = Inches(0.55)
    table_y = Inches(1.7)

    x = table_x
    for w, h in zip(cw, headers):
        _filled_box(s, x, table_y, w, Inches(0.4),
                    fill=BG_TITLE, line=TEAL,
                    text_lines=h, text_color=TEAL, text_size=11,
                    text_bold=True, align=PP_ALIGN.LEFT)
        x += w
    row_h = Inches(0.30)
    y = table_y + Inches(0.4)
    for i, (tag, body, where, sev_label, sev_color) in enumerate(rows):
        bg = BG_CONTENT if i % 2 == 0 else BG_PANEL
        x = table_x
        # Tag cell
        _filled_box(s, x, y, cw[0], row_h,
                    fill=bg, line=RULE,
                    text_lines=tag, text_color=GOLD, text_size=10, text_bold=True,
                    font=EQ_FONT, align=PP_ALIGN.CENTER)
        x += cw[0]
        # Assumption cell
        _filled_box(s, x, y, cw[1], row_h,
                    fill=bg, line=RULE,
                    text_lines=body, text_color=TEXT, text_size=10,
                    align=PP_ALIGN.LEFT)
        x += cw[1]
        # Where-used cell
        _filled_box(s, x, y, cw[2], row_h,
                    fill=bg, line=RULE,
                    text_lines=where, text_color=TEXT_DIM, text_size=10,
                    align=PP_ALIGN.LEFT)
        x += cw[2]
        # Severity cell
        _filled_box(s, x, y, cw[3], row_h,
                    fill=bg, line=RULE,
                    text_lines=sev_label, text_color=sev_color, text_size=10,
                    text_bold=True, align=PP_ALIGN.LEFT)
        y += row_h

    # Severity legend
    _text(
        s, Inches(0.55), Inches(7.0), SLIDE_W - Inches(1.1), Inches(0.4),
        "Severity legend:  green = minor   ·   gold = moderate   ·   red = critical (output is unreliable if violated).",
        size=11, color=TEXT_DIM, italic=True,
    )
    _footer(s, 23, TOTAL)


def slide_24_limitations(prs):
    s = _new_slide(prs)
    _slide_title(s, "Known limitations & open questions",
                 subtitle="What we are not pretending to solve — and where we expect to extend.")

    # Limitations
    _text(s, Inches(0.55), Inches(1.85), Inches(6.0), Inches(0.4),
          "Limitations  (already in production)", size=13, bold=True, color=RED)
    _bullets(
        s, Inches(0.55), Inches(2.3), Inches(6.0), Inches(4.5),
        [
            "Identifiability:  λ ↔ β confound makes outputs prior-sensitive in short histories.",
            "Model risk:  outputs are model-implied paths, not recovered truth.",
            "Factor stability:  β fitted full-sample; time-varying exposures not captured.",
            "Tail allocation:  Chow-Lin distributes residuals linearly within a quarter.",
            "Survivorship / backfill:  PE and HF indices carry both;  desmoothing does not correct them.",
            "MA(q) for q ≥ 4:  Laplace mode pending;  q=3 cap meanwhile.",
            "Long-horizon Kalman:  dense covariance bites past ~3000 daily states.",
        ],
        size=12, color=TEXT,
    )

    # Open questions
    _text(s, Inches(7.0), Inches(1.85), Inches(5.8), Inches(0.4),
          "Open questions  (research roadmap)", size=13, bold=True, color=TEAL)
    _bullets(
        s, Inches(7.0), Inches(2.3), Inches(5.8), Inches(4.5),
        [
            "Unified state-space formulation across all stages — elegance vs. modularity / debuggability trade-off.",
            "Fund-level data to estimate λ heterogeneity within a strategy (vs. peer-group pooling).",
            "Time-varying β:  rolling-window vs. DCC vs. Kalman β.",
            "Joint multivariate Chow-Lin with full Σ_ε (currently marginal + within-block rotation).",
            "Daily-level uncertainty bands  (currently propagated through Stage 3 only).",
        ],
        size=12, color=TEXT,
    )

    # Closing tag
    _filled_box(
        s, Inches(0.55), Inches(6.6), SLIDE_W - Inches(1.1), Inches(0.5),
        fill=BG_TITLE, line=TEAL,
        text_lines=[
            "All models are wrong.  These are less wrong than the alternatives — and we know exactly where they're wrong.",
        ],
        text_color=TEAL, text_size=14, text_bold=True,
    )

    _footer(s, 24, TOTAL)


# ──────────────────────────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────────────────────────


def build():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    builders = [
        slide_01_title,
        slide_02_problem,
        slide_03_pipeline,
        slide_04_notation,
        slide_05_smoothing_process,
        slide_06_rolling_annual,
        slide_07_single_step,
        slide_08_bayesian,
        slide_09_vasicek,
        slide_10_identifiability,
        slide_11_ma_intro,
        slide_12_ma_inversion,
        slide_13_ma_fragility,
        slide_14_threshold,
        slide_15_threshold_estim,
        slide_16_factor_decomp,
        slide_17_chow_lin,
        slide_18_aggregation,
        slide_19_chow_lin_finance,
        slide_20_kalman,
        slide_21_simulation,
        slide_22_asset_table,
        slide_23_assumptions,
        slide_24_limitations,
    ]
    for fn in builders:
        fn(prs)

    out = Path("docs/private_assets_frequency_methodology.pptx")
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(
        f"wrote {out}  "
        f"({out.stat().st_size:,} bytes,  {len(prs.slides)} slides,  "
        f"{len(EQ)} equation PNGs in {EQ_DIR})"
    )


if __name__ == "__main__":
    build()
