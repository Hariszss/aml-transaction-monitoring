"""Step 5: final comparison on the held-out test period (8-10 Sept).

Thresholds were fixed on validation data (steps 3 and 4); here they are
applied once to the test period. Outputs:
  results/tables/test_operating_points.csv   alerts / precision / recall per setting
  results/tables/test_any_rule.csv           the three rules combined ("any rule fires")
  results/tables/test_pr_auc.csv             threshold-free ranking quality
  results/tables/test_recall_by_typology.csv which simulated patterns are caught
  results/tables/daily_alert_rate_moderate.csv alert volume per day (does it stay stable?)
  figures/fig3_recall_vs_alert_volume.png    main figure
  figures/fig4_precision_vs_alert_volume.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tm import config, plotting  # noqa: E402
from tm.evaluation import alert_curve, operating_points, pr_auc, summarize_alerts  # noqa: E402
from tm.model import attach_lr_scores  # noqa: E402
from tm.patterns import KEY_COLUMNS, read_patterns  # noqa: E402
from tm.rules import RULES, rule_scores, threshold_for_alert_budget  # noqa: E402

plt = plotting.plt
FMT = lambda v: f"{v:.4g}"  # noqa: E731


def load():
    cols = ["row_id", "split", "label", "payment_format", "timestamp"] + [c for c, _ in RULES.values()] + KEY_COLUMNS
    df = pd.read_parquet(config.FEATURES_FILE, columns=cols)
    lr = pd.read_parquet(config.FEATURES_FILE.parent / "lr_scores.parquet", columns=["row_id", "label", "lr_score"])
    df = attach_lr_scores(df, lr)  # joined on row_id; raises if rows are missing or mismatched
    return df[df["split"] == "validation"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)


def scores_for(df, approach):
    if approach == "logistic_regression":
        return df["lr_score"].to_numpy(dtype=float)
    return rule_scores(df, approach)


def main() -> None:
    val, test = load()
    y_test = test["label"].to_numpy()
    approaches = list(RULES) + ["logistic_regression"]
    print(f"test: {len(test):,} transactions, {y_test.sum():,} positive labels "
          f"(prevalence {y_test.mean():.4%})")

    points, curves, auc_rows, flags = [], {}, [], {}
    for a in approaches:
        s_val, s_test = scores_for(val, a), scores_for(test, a)
        op = operating_points(a, s_val, s_test, y_test, config.ALERT_BUDGETS, threshold_for_alert_budget)
        points.append(op)
        curves[a] = alert_curve(y_test, s_test)
        auc_rows.append({"approach": a, "pr_auc_test": pr_auc(y_test, s_test),
                         "max_recall_test": curves[a]["recall"].iloc[-1]})
        flags[a] = {r.setting: s_test >= r.threshold for r in op.itertuples()}
    points = pd.concat(points, ignore_index=True)
    points.to_csv(config.TABLES_DIR / "test_operating_points.csv", index=False)
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(config.TABLES_DIR / "test_pr_auc.csv", index=False)

    show = ["approach", "setting", "threshold", "validation_alert_rate", "n_alerts", "alert_rate",
            "tp", "fp", "fn", "precision", "recall", "f1", "fp_per_tp", "lift"]
    pd.set_option("display.width", 250)
    print(points[show].to_string(index=False, float_format=FMT))
    print(auc.to_string(index=False, float_format=FMT))

    # The three rules together: an alert if any rule fires.
    any_rows = []
    for setting in config.ALERT_BUDGETS:
        combined = np.zeros(len(test), dtype=bool)
        for rule in RULES:
            combined |= flags[rule][setting]
        any_rows.append({"approach": "any_rule", "setting": setting, **summarize_alerts(y_test, combined)})
    any_rule = pd.DataFrame(any_rows)
    any_rule.to_csv(config.TABLES_DIR / "test_any_rule.csv", index=False)
    print(any_rule[["setting", "n_alerts", "alert_rate", "tp", "fp", "fn", "precision", "recall", "fp_per_tp"]]
          .to_string(index=False, float_format=FMT))

    # Reference point (not a monitoring rule): flag every ACH payment. Payment
    # format is a strong artefact of this simulation, see README.
    ach = summarize_alerts(y_test, (test["payment_format"] == "ACH").to_numpy())
    reference = pd.DataFrame([{"approach": "reference_all_ach", "setting": "fixed", **ach}])
    reference.to_csv(config.TABLES_DIR / "test_reference_ach.csv", index=False)
    print(reference[["n_alerts", "alert_rate", "tp", "fp", "precision", "recall"]].to_string(index=False, float_format=FMT))

    typology_recall(test, flags)
    daily_alert_rates(val, test, points)
    make_figures(curves, points, y_test.mean(), ach)


def daily_alert_rates(val: pd.DataFrame, test: pd.DataFrame, points: pd.DataFrame) -> None:
    """Share of each day's transactions flagged at the moderate setting.

    Thresholds were fixed on 6-7 Sept; this shows whether the alert volume
    they produce stays near 1% on other days."""
    both = pd.concat([val, test], ignore_index=True)
    day = both["timestamp"].dt.date.astype(str)
    out = pd.DataFrame({"day": day, "split": both["split"], "transactions": 1, "positive_labels": both["label"]})
    for a in points["approach"].unique():
        t = points.loc[(points["approach"] == a) & (points["setting"] == "moderate"), "threshold"].iloc[0]
        out[a] = scores_for(both, a) >= t
    table = out.groupby(["day", "split"]).agg(
        transactions=("transactions", "sum"), positive_labels=("positive_labels", "sum"),
        **{f"alert_rate_{a}": (a, "mean") for a in points["approach"].unique()}
    ).reset_index()
    table.to_csv(config.TABLES_DIR / "daily_alert_rate_moderate.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def typology_recall(test: pd.DataFrame, flags: dict) -> None:
    """Recall per simulated laundering pattern at the moderate (1%) setting."""
    patterns = read_patterns(config.RAW_PATTERNS).drop_duplicates(KEY_COLUMNS)
    pos = test[test["label"] == 1].copy()
    for a in flags:
        pos[a] = flags[a]["moderate"][test["label"].to_numpy() == 1]
    pos = pos.merge(patterns[KEY_COLUMNS + ["typology"]], on=KEY_COLUMNS, how="left")
    pos["typology"] = pos["typology"].fillna("not in patterns file")
    table = pos.groupby("typology").agg(
        positives=("label", "size"), **{f"recall_{a}": (a, "mean") for a in flags}
    ).sort_values("positives", ascending=False).reset_index()
    table.to_csv(config.TABLES_DIR / "test_recall_by_typology.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def make_figures(curves: dict, points: pd.DataFrame, prevalence: float, ach: dict) -> None:
    plotting.setup()
    budgets = config.ALERT_BUDGETS

    def decorate(ax):
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1.0)
        ax.set_xlabel("Alert volume: share of test transactions flagged (log scale)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:g}%"))
        for name, b in budgets.items():
            ax.axvline(b, color=plotting.NEUTRAL, linestyle=":", linewidth=1)
            ax.text(b * 1.06, 1.0, name, transform=ax.get_xaxis_transform(), fontsize=8,
                    color=plotting.TEXT_SECONDARY, va="top")

    # Figure 3 (main): recall vs alert volume.
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for a, c in curves.items():
        ax.plot(c["alert_rate"], c["recall"], color=plotting.COLORS[a], linewidth=2,
                label=plotting.LABELS[a], drawstyle="steps-post")
        p = points[points["approach"] == a]
        ax.plot(p["alert_rate"], p["recall"], "o", markersize=7, color=plotting.COLORS[a],
                markeredgecolor="white", markeredgewidth=1.5)
    ax.plot([1e-4, 1], [1e-4, 1], color=plotting.NEUTRAL, linewidth=1, linestyle="--",
            label="Random alerting")
    ax.plot(ach["alert_rate"], ach["recall"], "X", markersize=9, color=plotting.TEXT_SECONDARY,
            label="Reference: flag every ACH payment (simulator artefact)")
    decorate(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall: share of positive labels alerted")
    ax.set_title("Recall versus alert volume on the test period (8-10 Sept)")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.text(0.01, 0.01, "Dots: thresholds fixed on validation data for 0.1% / 1% / 5% alert budgets.",
             fontsize=8, color=plotting.TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(config.FIGURES_DIR / "fig3_recall_vs_alert_volume.png")
    plt.close(fig)

    # Figure 4: precision vs alert volume (how many alerts carry a positive label).
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for a, c in curves.items():
        c = c[c["n_alerts"] >= 20]  # precision of a handful of alerts is noise
        ax.plot(c["alert_rate"], c["precision"], color=plotting.COLORS[a], linewidth=2,
                label=plotting.LABELS[a], drawstyle="steps-post")
        p = points[(points["approach"] == a) & (points["n_alerts"] >= 20)]
        ax.plot(p["alert_rate"], p["precision"], "o", markersize=7, color=plotting.COLORS[a],
                markeredgecolor="white", markeredgewidth=1.5)
    ax.axhline(prevalence, color=plotting.NEUTRAL, linestyle="--", linewidth=1,
               label=f"Random alerting (prevalence {prevalence:.2%})")
    ax.plot(ach["alert_rate"], ach["precision"], "X", markersize=9, color=plotting.TEXT_SECONDARY,
            label="Reference: flag every ACH payment")
    decorate(ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:g}%"))
    ax.set_ylabel("Precision: share of alerts with positive label (log)")
    ax.set_title("Precision versus alert volume on the test period (8-10 Sept)")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.set_ylim(0.0002, 0.2)
    fig.text(0.01, 0.01, "Curves start at 20 alerts. Dots: thresholds fixed on validation data.",
             fontsize=8, color=plotting.TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(config.FIGURES_DIR / "fig4_precision_vs_alert_volume.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
