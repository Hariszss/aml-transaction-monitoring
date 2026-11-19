"""Secondary comparison: a plain logistic-regression baseline.

It uses the same pre-transaction information as the rules plus a few simple
transaction attributes. All preprocessing is fitted on the training period
only (inside one scikit-learn Pipeline), so nothing is learned from the
validation or test periods.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "log_amount_usd",
    "log_amount_vs_history",
    "has_history",
    "log_sender_n_out_24h",
    "log_sender_n_recipients_24h",
    "log_receiver_n_senders_24h",
    "log_sender_n_out_7d",
    "is_cross_bank",
    "is_cross_currency",
    "is_self_transfer",
]
CATEGORICAL_FEATURES = ["payment_format"]


def make_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, row-wise transformations (no fitted parameters)."""
    out = pd.DataFrame(index=df.index)
    out["log_amount_usd"] = np.log1p(df["amount_usd"].clip(lower=0))
    out["has_history"] = df["amount_vs_history"].notna().astype(int)
    # No history -> neutral ratio of 1 (log = 0); has_history tells the model.
    out["log_amount_vs_history"] = np.log(df["amount_vs_history"].fillna(1.0).clip(lower=1e-6))
    for col in ["sender_n_out_24h", "sender_n_recipients_24h", "receiver_n_senders_24h", "sender_n_out_7d"]:
        out[f"log_{col}"] = np.log1p(df[col])
    for col in ["is_cross_bank", "is_cross_currency", "is_self_transfer"]:
        out[col] = df[col].astype(int)
    out["payment_format"] = df["payment_format"].astype(str)
    return out


def build_pipeline(C: float = 1.0, seed: int = 42) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    # class_weight="balanced" up-weights the rare positive label during fitting.
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def attach_lr_scores(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Join model scores to transactions on the stable `row_id`.

    Fails if any transaction has no score, if a row_id is duplicated, or if
    the labels stored with the scores disagree with the transactions' labels.
    """
    merged = df.merge(
        scores[["row_id", "label", "lr_score"]].rename(columns={"label": "_score_label"}),
        on="row_id", how="left", validate="one_to_one",
    )
    if merged["lr_score"].isna().any():
        raise ValueError(f"{int(merged['lr_score'].isna().sum())} transactions have no model score")
    if not (merged["_score_label"].to_numpy() == merged["label"].to_numpy()).all():
        raise ValueError("labels stored with the model scores do not match the transactions")
    return merged.drop(columns="_score_label")


def coefficient_table(pipe: Pipeline) -> pd.DataFrame:
    names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_.ravel()
    return (
        pd.DataFrame({"feature": names, "coefficient": coefs})
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )
