"""Step 4 (secondary): logistic-regression baseline.

Fitted on the training period (1-5 Sept) with default regularisation (C=1)
and class_weight="balanced". No hyper-parameter search: the model is only an
exploratory point of comparison for the rules.

Outputs: data/processed/lr_scores.parquet, results/tables/lr_coefficients.csv
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from tm import config  # noqa: E402
from tm.evaluation import pr_auc  # noqa: E402
from tm.model import build_pipeline, coefficient_table, make_model_frame  # noqa: E402

SOURCE_COLUMNS = [
    "row_id", "split", "label", "amount_usd", "amount_vs_history", "sender_n_out_24h",
    "sender_n_recipients_24h", "receiver_n_senders_24h", "sender_n_out_7d",
    "is_cross_bank", "is_cross_currency", "is_self_transfer", "payment_format",
]


def main() -> None:
    start = time.time()
    df = pd.read_parquet(config.FEATURES_FILE, columns=SOURCE_COLUMNS)
    X = make_model_frame(df)
    y = df["label"].to_numpy()
    is_train = (df["split"] == "train").to_numpy()

    pipe = build_pipeline(C=1.0, seed=config.RANDOM_SEED)
    pipe.fit(X[is_train], y[is_train])
    print(f"fitted on {is_train.sum():,} training rows in {time.time() - start:.0f}s")

    scores = pd.DataFrame({"row_id": df["row_id"], "split": df["split"], "label": y,
                           "lr_score": pipe.predict_proba(X)[:, 1]})
    scores.to_parquet(config.FEATURES_FILE.parent / "lr_scores.parquet", index=False)

    for split in ["train", "validation"]:
        m = scores["split"] == split
        print(f"PR-AUC {split}: {pr_auc(scores.loc[m, 'label'], scores.loc[m, 'lr_score']):.4f}")

    coefs = coefficient_table(pipe)
    coefs.to_csv(config.TABLES_DIR / "lr_coefficients.csv", index=False)
    print(coefs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
