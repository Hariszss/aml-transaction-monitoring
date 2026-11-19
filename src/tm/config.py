"""Project-wide settings: paths, analysis period and the chronological split.

All dates are calendar days in the simulated data (timestamps have no time zone;
we treat them as UTC throughout).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_TRANSACTIONS = PROJECT_ROOT / "data" / "raw" / "HI-Small_Trans.csv"
RAW_PATTERNS = PROJECT_ROOT / "data" / "raw" / "HI-Small_Patterns.txt"
FEATURES_FILE = PROJECT_ROOT / "data" / "processed" / "features.parquet"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"

# After 10 Sept 2022 the simulator produces almost no ordinary transactions,
# but labelled laundering transactions continue (see README). Those days would
# give a misleading picture, so the analysis is restricted to 1-10 Sept.
ANALYSIS_START = "2022-09-01"
ANALYSIS_END = "2022-09-10"  # inclusive

# Chronological split (inclusive day ranges).
SPLITS = {
    "train": ("2022-09-01", "2022-09-05"),
    "validation": ("2022-09-06", "2022-09-07"),
    "test": ("2022-09-08", "2022-09-10"),
}

# History windows used for account-behaviour features (in seconds).
WINDOW_24H = 24 * 3600
WINDOW_7D = 7 * 24 * 3600

# Minimum number of earlier outgoing payments before we trust an account's
# "typical amount" (used by the amount-vs-history rule).
MIN_HISTORY_TXNS = 3

# Alert budgets used to set rule thresholds on the validation period:
# the share of all transactions an approach is allowed to flag.
ALERT_BUDGETS = {"restrictive": 0.001, "moderate": 0.01, "sensitive": 0.05}

RANDOM_SEED = 42
