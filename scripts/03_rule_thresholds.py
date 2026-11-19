"""Step 3: choose rule thresholds on the validation period (6-7 Sept).

For each rule and each alert budget (restrictive 0.1%, moderate 1%,
sensitive 5% of transactions) we choose the lowest threshold that stays
within the budget on validation data. The test period is not used here.

Before that, a small design check on the *training* period compares the
counterparty-count scores that were considered for rule 3 (see README):
an earlier version used the sender's number of recipients as well, which
mostly flagged very busy accounts.

Outputs: results/tables/rule_design_train.csv
         results/tables/rule_thresholds_validation.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from tm import config  # noqa: E402
from tm.evaluation import alert_curve, summarize_alerts  # noqa: E402
from tm.rules import RULES, apply_threshold, rule_scores, threshold_for_alert_budget  # noqa: E402


def recall_within_budget(curve: pd.DataFrame, budget: float) -> float:
    ok = curve[curve["alert_rate"] <= budget]
    return float(ok["recall"].iloc[-1]) if len(ok) else 0.0


def design_check_on_train(df: pd.DataFrame) -> None:
    train = df[df["split"] == "train"]
    y = train["label"].to_numpy()
    candidates = {
        "sender distinct recipients 24h": train["sender_n_recipients_24h"],
        "receiver distinct senders 24h (fan-in)": train["receiver_n_senders_24h"],
        "max of both (first version of rule 3)": train[["sender_n_recipients_24h", "receiver_n_senders_24h"]].max(axis=1),
    }
    rows = []
    for name, score in candidates.items():
        curve = alert_curve(y, score.to_numpy(dtype=float))
        rows.append({"candidate_score": name,
                     **{f"recall_at_{k}": recall_within_budget(curve, b) for k, b in config.ALERT_BUDGETS.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(config.TABLES_DIR / "rule_design_train.csv", index=False)
    print("Design check on training period (recall within each alert budget):")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main() -> None:
    cols = ["split", "label", "sender_n_recipients_24h"] + [c for c, _ in RULES.values()]
    df = pd.read_parquet(config.FEATURES_FILE, columns=cols)
    design_check_on_train(df)
    val = df[df["split"] == "validation"].reset_index(drop=True)
    y = val["label"].to_numpy()
    print(f"validation: {len(val):,} transactions, {y.sum():,} positive labels")

    rows = []
    for rule, (col, desc) in RULES.items():
        s = rule_scores(val, rule)
        for setting, budget in config.ALERT_BUDGETS.items():
            t = threshold_for_alert_budget(s, budget)
            rows.append({"rule": rule, "score": col, "setting": setting, "budget": budget,
                         "threshold": t, **summarize_alerts(y, apply_threshold(s, t))})
        # Maximum reach of the rule: lowest possible threshold.
        curve = alert_curve(y, s)
        print(f"{rule}: max recall if every scorable transaction alerted = {curve['recall'].iloc[-1]:.3f} "
              f"({curve['alert_rate'].iloc[-1]:.1%} of transactions)")
    out = pd.DataFrame(rows)
    out.to_csv(config.TABLES_DIR / "rule_thresholds_validation.csv", index=False)
    pd.set_option("display.width", 200)
    print(out[["rule", "setting", "threshold", "n_alerts", "alert_rate", "tp", "fp", "precision", "recall"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
