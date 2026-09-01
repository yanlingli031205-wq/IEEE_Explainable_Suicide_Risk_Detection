from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# IEEE PDF checkers reject Matplotlib's default Type-3 font embedding.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


COLORS = {
    "input": "#E8EEF7",
    "model": "#E7D9F7",
    "token": "#D9EAD3",
    "latent": "#FCE5CD",
    "decoder": "#D0E0E3",
    "output": "#FFF2CC",
    "ink": "#25324A",
}


def box(ax, xy, width, height, text, color, fontsize=8.2, weight="normal"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=0.9, edgecolor=COLORS["ink"], facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", fontsize=fontsize,
        color=COLORS["ink"], weight=weight, linespacing=1.15,
    )
    return patch


def arrow(ax, start, end, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10,
        linewidth=1.0, color=COLORS["ink"],
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2,
    ))


def architecture():
    fig, ax = plt.subplots(figsize=(7.15, 4.05))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    box(ax, (0.15, 2.55), 1.25, 1.0, "Reddit post", COLORS["input"], 9, "bold")
    box(ax, (0.15, 4.35), 1.25, 0.85, "Risk cards", COLORS["input"], 8.5)
    box(ax, (0.15, 0.85), 1.25, 0.85, "24 label cards", COLORS["input"], 8.5)

    box(
        ax, (1.85, 2.2), 2.15, 1.7,
        "Qwen3.8-27B\n4-bit frozen backbone\n+ task-specific LoRA",
        COLORS["model"], 9, "bold",
    )
    arrow(ax, (1.4, 3.05), (1.85, 3.05))
    arrow(ax, (1.4, 4.78), (2.2, 3.9), -0.08)
    arrow(ax, (1.4, 1.28), (2.2, 2.2), 0.08)

    box(ax, (4.45, 4.35), 1.65, 0.9, "Final-token\nYES/NO margins", COLORS["token"], 8.3)
    box(ax, (4.45, 2.65), 1.65, 0.9, "Layer-63\nanswer state", COLORS["latent"], 8.3)
    box(ax, (4.45, 0.95), 1.65, 0.9, "Exact-span\ncandidate margins", COLORS["decoder"], 8.3)
    arrow(ax, (4.0, 3.45), (4.45, 4.65), -0.08)
    arrow(ax, (4.0, 3.05), (4.45, 3.10))
    arrow(ax, (4.0, 2.65), (4.45, 1.40), 0.08)

    box(ax, (6.55, 4.25), 1.55, 1.1, "Risk latent\nreadout", COLORS["latent"], 8.6, "bold")
    box(ax, (6.55, 2.55), 1.55, 1.1, "Label route\n7 latent\n17 token", COLORS["latent"], 7.5, "bold")
    box(ax, (6.55, 0.75), 1.55, 1.3, "Candidate meta-\ncalibration\n+ event-set decoder", COLORS["decoder"], 7.6, "bold")

    arrow(ax, (6.10, 4.80), (6.55, 4.80))
    arrow(ax, (6.10, 3.10), (6.55, 3.10))
    arrow(ax, (5.55, 4.35), (6.90, 3.65), 0.08)
    arrow(ax, (6.10, 1.40), (6.55, 1.40))

    box(ax, (8.55, 4.35), 1.25, 0.9, "4-level risk", COLORS["output"], 8.5, "bold")
    box(ax, (8.55, 2.65), 1.25, 0.9, "Factor set", COLORS["output"], 8.5, "bold")
    box(ax, (8.55, 0.95), 1.25, 0.9, "Verbatim\nevidence set", COLORS["output"], 8.3, "bold")
    arrow(ax, (8.10, 4.80), (8.55, 4.80))
    arrow(ax, (8.10, 3.10), (8.55, 3.10))
    arrow(ax, (8.10, 1.40), (8.55, 1.40))

    ax.text(
        5.0, 5.86,
        "Task-specific readouts from a shared semantic verifier",
        ha="center", va="center", fontsize=11, weight="bold", color=COLORS["ink"],
    )
    fig.tight_layout(pad=0.15)
    fig.savefig(FIG / "architecture.pdf", bbox_inches="tight")
    fig.savefig(FIG / "architecture.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def validation_protocol():
    fig, ax = plt.subplots(figsize=(7.15, 1.75))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    box(ax, (0.15, 0.55), 1.65, 0.9, "User-grouped\n3-fold partition", COLORS["input"], 8.5, "bold")
    box(ax, (2.25, 0.55), 1.65, 0.9, "Fold 0\ndevelopment", COLORS["model"], 8.5, "bold")
    box(ax, (4.35, 0.55), 1.65, 0.9, "Freeze layer, C,\nlabels, decoder", COLORS["token"], 8.2, "bold")
    box(ax, (6.45, 0.55), 1.65, 0.9, "Folds 1 and 2\nconfirmation", COLORS["latent"], 8.5, "bold")
    box(ax, (8.55, 0.55), 1.30, 0.9, "Full-data refit\nand test", COLORS["output"], 8.2, "bold")
    for left, right in ((1.80, 2.25), (3.90, 4.35), (6.00, 6.45), (8.10, 8.55)):
        arrow(ax, (left, 1.00), (right, 1.00))
    ax.text(5.0, 0.14, "Layer, route, regularization, and thresholds are fixed before confirmation", ha="center", fontsize=8.0, color=COLORS["ink"])
    fig.tight_layout(pad=0.1)
    fig.savefig(FIG / "validation_protocol.pdf", bbox_inches="tight")
    fig.savefig(FIG / "validation_protocol.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    validation_protocol()
    print(f"Figures written to {FIG}")
