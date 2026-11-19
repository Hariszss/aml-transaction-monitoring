"""Independent check of the reported evaluation numbers.

This script deliberately does NOT import tm.evaluation or tm.rules. It
re-implements threshold selection, alert counting, the metrics and the
precision-recall AUC from scratch, starting from the saved labels and scores
(data/processed/features.parquet and lr_scores.parquet). It then compares
the results with:
  - results/tables/test_operating_points.csv, test_any_rule.csv,
    test_reference_ach.csv, test_pr_auc.csv, daily_alert_rate_moderate.csv
  - the results table in README.md

Run after run_all.py:  python review/verify_reported_metrics.py
Exit code 0 = everything matches.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
BUDGETS = {"restrictive": 0.001, "moderate": 0.01, "sensitive": 0.05}
SCORE_COLUMNS = {
    "large_amount": "amount_usd",
    "amount_vs_history": "amount_vs_history",
    "fan_in": "receiver_n_senders_24h",
    "logistic_regression": "lr_score",
}
problems = []


def check(cond, msg):
    if not cond:
        problems.append(msg)


def threshold_for_budget(scores, budget):
    """Lowest score value v such that #(scores >= v) <= budget * n."""
    n = len(scores)
    finite = np.sort(scores[np.isfinite(scores)])  # ascending
    best = np.inf
    for v in np.unique(finite)[::-1]:  # descending
        n_ge = len(finite) - np.searchsorted(finite, v, side="left")
        if n_ge <= budget * n:
            best = v
        else:
            break
    return best


def metrics(y, flag):
    tp = int(np.sum(flag & (y == 1)))
    fp = int(np.sum(flag & (y == 0)))
    fn = int(np.sum(~flag & (y == 1)))
    alerts = tp + fp
    return {
        "n_alerts": alerts, "tp": tp, "fp": fp, "fn": fn,
        "alert_rate": alerts / len(y),
        "precision": tp / alerts if alerts else np.nan,
        "recall": tp / (tp + fn),
        "fp_per_tp": fp / tp if tp else np.inf,
    }


def average_precision(y, scores):
    """AP = sum over distinct thresholds of (recall step) * precision."""
    s = scores.copy()
    finite = s[np.isfinite(s)]
    s[~np.isfinite(s)] = finite.min() - 1.0
    df = pd.DataFrame({"s": s, "y": y}).groupby("s")["y"].agg(["sum", "count"]).sort_index(ascending=False)
    tp = df["sum"].cumsum().to_numpy()
    alerts = df["count"].cumsum().to_numpy()
    recall = tp / y.sum()
    precision = tp / alerts
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def main():
    cols = ["row_id", "split", "label", "payment_format", "timestamp", "amount_usd", "amount_vs_history",
            "receiver_n_senders_24h"]
    df = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet", columns=cols)
    lr = pd.read_parquet(ROOT / "data" / "processed" / "lr_scores.parquet")
    # Independent alignment check: join on row_id and verify a one-to-one match.
    check(lr["row_id"].is_unique and df["row_id"].is_unique, "row_id not unique")
    check(set(lr["row_id"]) == set(df["row_id"]), "row_id sets of features and scores differ")
    df = df.merge(lr[["row_id", "label", "lr_score"]].rename(columns={"label": "score_label"}), on="row_id")
    check((df["score_label"] == df["label"]).all(), "labels stored with scores disagree with features")
    val, test = df[df["split"] == "validation"], df[df["split"] == "test"]
    y_val, y_test = val["label"].to_numpy(), test["label"].to_numpy()
    print(f"test transactions {len(test):,}, positives {y_test.sum():,}")

    # 1. Operating points.
    reported = pd.read_csv(TABLES / "test_operating_points.csv")
    mine, flags = {}, {}
    for a, col in SCORE_COLUMNS.items():
        sv = val[col].astype(float).fillna(-np.inf).to_numpy()
        st = test[col].astype(float).fillna(-np.inf).to_numpy()
        for setting, b in BUDGETS.items():
            t = threshold_for_budget(sv, b)
            f = st >= t
            flags[(a, setting)] = f
            m = metrics(y_test, f)
            mine[(a, setting)] = m
            r = reported[(reported["approach"] == a) & (reported["setting"] == setting)].iloc[0]
            check(np.isclose(t, r["threshold"]), f"{a}/{setting}: threshold {t} vs {r['threshold']}")
            for k in ["n_alerts", "tp", "fp", "fn"]:
                check(m[k] == r[k], f"{a}/{setting}: {k} {m[k]} vs {r[k]}")
            for k in ["alert_rate", "precision", "recall", "fp_per_tp"]:
                check(np.isclose(m[k], r[k]), f"{a}/{setting}: {k} {m[k]} vs {r[k]}")
    print(f"operating points checked: {len(mine)}")

    # 2. Any-rule combination.
    any_rep = pd.read_csv(TABLES / "test_any_rule.csv")
    for setting in BUDGETS:
        f = flags[("large_amount", setting)] | flags[("amount_vs_history", setting)] | flags[("fan_in", setting)]
        m = metrics(y_test, f)
        mine[("any_rule", setting)] = m
        r = any_rep[any_rep["setting"] == setting].iloc[0]
        for k in ["n_alerts", "tp", "fp", "fn"]:
            check(m[k] == r[k], f"any_rule/{setting}: {k} {m[k]} vs {r[k]}")

    # 3. ACH reference.
    m = metrics(y_test, (test["payment_format"] == "ACH").to_numpy())
    r = pd.read_csv(TABLES / "test_reference_ach.csv").iloc[0]
    for k in ["n_alerts", "tp", "fp", "fn"]:
        check(m[k] == r[k], f"ACH reference: {k} {m[k]} vs {r[k]}")
    print(f"ACH reference: {m['n_alerts']:,} alerts, recall {m['recall']:.4f}, precision {m['precision']:.4f}")

    # 4. PR-AUC.
    auc = pd.read_csv(TABLES / "test_pr_auc.csv")
    for a, col in SCORE_COLUMNS.items():
        ap = average_precision(y_test, test[col].astype(float).fillna(-np.inf).to_numpy())
        rep = auc.loc[auc["approach"] == a, "pr_auc_test"].iloc[0]
        check(np.isclose(ap, rep, rtol=1e-6), f"PR-AUC {a}: {ap} vs {rep}")
        print(f"PR-AUC {a}: recomputed {ap:.5f}, reported {rep:.5f}")

    # 5. Daily alert rates at the moderate setting.
    daily = pd.read_csv(TABLES / "daily_alert_rate_moderate.csv")
    both = pd.concat([val, test])
    day = both["timestamp"].dt.date.astype(str).to_numpy()
    for a, col in SCORE_COLUMNS.items():
        t = reported[(reported["approach"] == a) & (reported["setting"] == "moderate")]["threshold"].iloc[0]
        f = both[col].astype(float).fillna(-np.inf).to_numpy() >= t
        for d in daily["day"]:
            rate = f[day == d].mean()
            rep = daily.loc[daily["day"] == d, f"alert_rate_{a}"].iloc[0]
            check(np.isclose(rate, rep), f"daily {a} {d}: {rate} vs {rep}")

    # 6. README results table (rounded values as printed).
    check_markdown_table(ROOT / "README.md", mine, full=True)

    if problems:
        print(f"\n{len(problems)} DISCREPANCIES:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("\nAll recomputed numbers match the result files and the documentation tables.")


def _num(text):
    return float(text.replace(",", "").replace("%", "").strip())


NAME = {"Rule 1": "large_amount", "Rule 2": "amount_vs_history", "Rule 3": "fan_in",
        "Any": "any_rule", "Logistic": "logistic_regression"}


def check_markdown_table(path, mine, full):
    text = path.read_text(encoding="utf-8")
    current, n_rows = None, 0
    for line in text.splitlines():
        if full:
            m = re.match(r"\|\s*([^|]*)\|\s*(restrictive|moderate|sensitive)\s*\|(.*)", line)
            if not m:
                continue
            label, setting, rest = m.group(1).strip(), m.group(2), m.group(3)
            if label:
                current = next((v for k, v in NAME.items() if label.startswith(k)), None)
            cells = [c for c in rest.split("|") if c.strip()]
            alerts, pct, tp, fp, missed, prec, rec, fptp = [_num(c) for c in cells[:8]]
        else:
            m = re.match(r"\|\s*(Rule 1|Rule 2|Rule 3|Any rule|Logistic regression)\s*\|(.*)", line)
            if not m:
                continue
            current = next(v for k, v in NAME.items() if m.group(1).startswith(k))
            setting = "moderate"
            cells = [c for c in m.group(2).split("|") if c.strip()]
            alerts, tp, fp, rec, prec, fptp = [_num(c) for c in cells[:6]]
            pct = missed = None
        ref = mine[(current, setting)]
        where = f"{path.name} {current}/{setting}"
        check(alerts == ref["n_alerts"], f"{where}: alerts {alerts} vs {ref['n_alerts']}")
        check(tp == ref["tp"] and fp == ref["fp"], f"{where}: TP/FP {tp}/{fp} vs {ref['tp']}/{ref['fp']}")
        if missed is not None:
            check(missed == ref["fn"], f"{where}: missed {missed} vs {ref['fn']}")
            check(pct == round(100 * ref["alert_rate"], 2), f"{where}: % flagged {pct} vs {100 * ref['alert_rate']:.4f}")
        check(prec == round(100 * ref["precision"], 2), f"{where}: precision {prec} vs {100 * ref['precision']:.4f}")
        check(rec == round(100 * ref["recall"], 1), f"{where}: recall {rec} vs {100 * ref['recall']:.3f}")
        check(fptp == round(ref["fp_per_tp"]), f"{where}: FP per TP {fptp} vs {ref['fp_per_tp']:.2f}")
        n_rows += 1
    print(f"{path.name}: {n_rows} table rows checked")
    check(n_rows == (13 if full else 5), f"{path.name}: expected {13 if full else 5} rows, parsed {n_rows}")


if __name__ == "__main__":
    main()
