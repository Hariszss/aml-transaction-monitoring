"""Shared matplotlib styling (one consistent look for all figures)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (colour follows the approach, never its rank).
COLORS = {
    "large_amount": "#2a78d6",
    "amount_vs_history": "#eb6834",
    "fan_in": "#1baf7a",
    "logistic_regression": "#eda100",
}
LABELS = {
    "large_amount": "Rule 1: large amount",
    "amount_vs_history": "Rule 2: amount vs own history",
    "fan_in": "Rule 3: fan-in (24h)",
    "logistic_regression": "Logistic regression (secondary)",
}
NEUTRAL = "#8c8b86"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def setup():
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.edgecolor": NEUTRAL,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
