"""Parse HI-Small_Patterns.txt: which laundering typology each labelled
transaction belongs to (fan-out, cycle, ...), according to the simulator."""
import re

import pandas as pd

_BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)")
KEY_COLUMNS = ["timestamp_str", "from_bank", "from_acct", "to_bank", "to_acct", "amount_paid_str"]


def parse_patterns(lines) -> pd.DataFrame:
    """Return one row per transaction listed inside a laundering attempt."""
    rows = []
    typology = None
    attempt = 0
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = _BEGIN.match(line)
        if m:
            typology = m.group(1)
            attempt += 1
            continue
        if line.startswith("END LAUNDERING ATTEMPT"):
            typology = None
            continue
        if typology is None:
            continue
        parts = line.split(",")
        if len(parts) < 11:
            continue
        rows.append(
            {
                "timestamp_str": parts[0],
                "from_bank": parts[1],
                "from_acct": parts[2],
                "to_bank": parts[3],
                "to_acct": parts[4],
                "amount_paid_str": parts[7],
                "typology": typology,
                "attempt_id": attempt,
            }
        )
    return pd.DataFrame(rows, columns=KEY_COLUMNS + ["typology", "attempt_id"])


def read_patterns(path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as fh:
        return parse_patterns(fh)
