"""Download the two HI-Small files from Kaggle into data/raw/.

The IBM AML data is published on Kaggle by its author (Erik Altman):
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
Licence: Community Data License Agreement - Sharing 1.0 (CDLA-Sharing-1.0).

Single public files could be downloaded without a Kaggle account when this
project was built (Nov 2026). If the download fails, download
HI-Small_Trans.csv and HI-Small_Patterns.txt manually from the Kaggle page
(free account needed) and place them in data/raw/.
"""
import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
BASE = ("https://www.kaggle.com/api/v1/datasets/download/"
        "ealtman2019/ibm-transactions-for-anti-money-laundering-aml?file_name=")
FILES = ["HI-Small_Trans.csv", "HI-Small_Patterns.txt"]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = RAW_DIR / name
        if target.exists() and target.stat().st_size > 0:
            print(f"already present: {target}")
            continue
        print(f"downloading {name} ...")
        try:
            urllib.request.urlretrieve(BASE + name, target)
        except Exception as exc:  # network / access problems
            sys.exit(f"Download failed ({exc}). Please download {name} manually (see docstring).")
        print(f"saved {target} ({target.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
