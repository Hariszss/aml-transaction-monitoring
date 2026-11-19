"""Run the whole analysis in order: python run_all.py

Requires Java 17 (JAVA_HOME set) for the PySpark steps.
"""
import subprocess
import sys
from pathlib import Path

STEPS = [
    "00_download_data.py",
    "01_prepare_features.py",
    "02_explore_with_spark.py",
    "03_rule_thresholds.py",
    "04_logistic_regression.py",
    "05_compare_on_test.py",
]

if __name__ == "__main__":
    scripts = Path(__file__).resolve().parent / "scripts"
    for step in STEPS:
        print(f"\n===== {step} =====", flush=True)
        subprocess.run([sys.executable, str(scripts / step)], check=True)
