"""Step 1: download the credit card fraud dataset and prove it's the right file.

Source: OpenML dataset 42175 ("CreditCardFraudDetection"), a mirror of the Kaggle / ULB Machine Learning Group file.
(Dataset 1597 is the same data, but it flags Time as a row ID, so it would arrive without the Time column.)
No account or API key is needed. If OpenML is down, download creditcard.csv from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud by hand, put it in data/raw/, and re-run:
the script then skips the download and just runs the checks.
"""
import sys

import pandas as pd
from sklearn.datasets import fetch_openml

from common import CACHE, COLUMNS, EXPECTED_FRAUD, EXPECTED_ROWS, OPENML_ID, RAW_CSV


def download() -> pd.DataFrame:
    print(f"Downloading OpenML dataset {OPENML_ID} (about 150 MB, can take a few minutes)...")
    bunch = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto", data_home=str(CACHE))
    df = bunch.frame.copy()
    # OpenML copies store the label as '0'/'1' text or 0.0/1.0 numbers; normalise either to int 0/1
    df["Class"] = pd.to_numeric(df["Class"].astype(str).str.strip("'")).round().astype(int)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"Unexpected columns from OpenML; missing {missing}. Got: {list(df.columns)}")
    df = df[COLUMNS]
    df[COLUMNS[:-1]] = df[COLUMNS[:-1]].astype(float)
    df.to_csv(RAW_CSV, index=False)
    return df


def main():
    if RAW_CSV.exists():
        print(f"Using existing {RAW_CSV}")
        df = pd.read_csv(RAW_CSV)
    else:
        df = download()

    checks = [
        (len(df) == EXPECTED_ROWS, f"rows = {len(df):,} (expected {EXPECTED_ROWS:,})"),
        (list(df.columns) == COLUMNS, f"{df.shape[1]} columns: Time, V1-V28, Amount, Class"),
        (int(df["Class"].sum()) == EXPECTED_FRAUD, f"fraud cases = {int(df['Class'].sum())} (expected {EXPECTED_FRAUD})"),
        (set(df["Class"].unique()) == {0, 1}, "Class is 0/1 only"),
        (int(df.isna().sum().sum()) == 0, f"missing values = {int(df.isna().sum().sum())}"),
    ]
    ok = True
    for passed, msg in checks:
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {msg}")

    fraud = df[df["Class"] == 1]
    print(f"\nFraud rate {df['Class'].mean():.4%}  (1 in {len(df) / len(fraud):,.0f})")
    print(f"Time span  {df['Time'].min():,.0f} to {df['Time'].max():,.0f} seconds (~{df['Time'].max() / 3600:.0f} hours)")
    print(f"Amounts    total EUR {df['Amount'].sum():,.2f}; fraud total EUR {fraud['Amount'].sum():,.2f}")
    print(f"Duplicates {df.duplicated().sum():,} exact duplicate rows")
    if not ok:
        sys.exit("\nData checks FAILED - do not continue.")
    print(f"\nData OK -> {RAW_CSV}")


if __name__ == "__main__":
    main()
