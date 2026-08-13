"""Shared styling for concept-contribution figures.

Deliberately depends only on matplotlib / numpy / PIL so it can be imported both
by visualize_concept_coherence.py (which pulls in torch, CLIP and the
Label-free-CBM data utils) and by plot_manual_contributions.py (which runs
anywhere from a JSON spec of already-computed values).
"""
import numpy as np

# Publication palette. Refined versions of the original #E74C3C / #3498DB so the
# figure reads the same but prints cleanly.
POS_COLOR = "#C44E52"
NEG_COLOR = "#4C72B0"
LABEL_COLOR = "#2B2B2B"
GRID_COLOR = "#E4E4E4"
SPINE_COLOR = "#9A9A9A"

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
}


def draw_contribution_panel(ax_img, ax_bar, image, names, values,
                            gt_name=None, pred_name=None,
                            show_xlabel=True, label_fontsize=11,
                            title_fontsize=12, value_fontsize=10):
    """Render one image + horizontal contribution bar chart pair.

    Args:
        ax_img:    axes for the photo (may be None to skip the image panel)
        ax_bar:    axes for the bars
        image:     PIL image, or None
        names:     concept names, already ordered top to bottom
        values:    matching contribution values (signed)
        gt_name:   ground-truth class, drawn above the image
        pred_name: predicted class, drawn above the bars
    """
    if ax_img is not None:
        if image is not None:
            ax_img.imshow(image)
        ax_img.axis("off")
        if gt_name:
            ax_img.set_title(f"Ground truth: {gt_name}", fontsize=title_fontsize,
                             color=LABEL_COLOR, pad=10)

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        ax_bar.axis("off")
        return

    y_pos = np.arange(len(values))
    span = np.abs(values).max()
    if span == 0:
        span = 1.0
    # Opacity ramps with magnitude so the dominant concept reads as dominant.
    alphas = 0.55 + 0.45 * (np.abs(values) / span)
    colors = [POS_COLOR if v > 0 else NEG_COLOR for v in values]

    ax_bar.barh(y_pos, values, color=colors, edgecolor="none", height=0.62, zorder=3)
    for bar, alpha in zip(ax_bar.patches, alphas):
        bar.set_alpha(float(alpha))

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(names, fontsize=label_fontsize, color=LABEL_COLOR)
    ax_bar.invert_yaxis()
    if show_xlabel:
        ax_bar.set_xlabel("Concept contribution", fontsize=label_fontsize,
                          color=LABEL_COLOR, labelpad=8)
    if pred_name:
        ax_bar.set_title(f"Predicted: {pred_name}", fontsize=title_fontsize,
                         color=LABEL_COLOR, pad=10)

    # Headroom so the value label on the longest bar is not clipped
    lo, hi = min(0.0, values.min()), max(0.0, values.max())
    ax_bar.set_xlim(lo - 0.10 * span if lo < 0 else 0.0, hi + 0.13 * span)

    offset = 0.015 * span
    for i, v in enumerate(values):
        ax_bar.text(v + offset * np.sign(v), i, f"{v:+.2f}",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=value_fontsize, color=LABEL_COLOR)

    ax_bar.grid(axis="x", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_bar.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax_bar.spines[side].set_visible(False)
    ax_bar.spines["bottom"].set_color(SPINE_COLOR)
    ax_bar.tick_params(axis="y", length=0)
    ax_bar.tick_params(axis="x", colors=LABEL_COLOR, labelsize=value_fontsize)
