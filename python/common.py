"""Shared paths and settings for the Fraud_Detection pipeline (script, notebook and verify.py all import this)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLEAU = ROOT / "tableau"
CACHE = ROOT / "data" / "openml_cache"
RAW_CSV = RAW / "creditcard.csv"

# Dataset identity: ULB / Worldline credit card fraud (Sept 2013), OpenML mirror of the Kaggle file.
# 42175 keeps Time as a normal column (dataset 1597 flags Time as a row ID, so fetch_openml drops it).
OPENML_ID = 42175
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492
V_COLS = [f"V{i}" for i in range(1, 29)]
COLUMNS = ["Time", *V_COLS, "Amount", "Class"]

# Modelling settings (locked in PLAN.md)
SEED = 42
SPLIT = (0.60, 0.20, 0.20)          # train / validation / test, stratified on Class
FP_COST = 10.0                      # € per false alarm (analyst review); a Tableau parameter later
TP_COST = 10.0                      # € to review a correctly flagged fraud
# A missed fraud (false negative) costs its own Amount.

for p in (RAW, PROCESSED, TABLEAU):
    p.mkdir(parents=True, exist_ok=True)
