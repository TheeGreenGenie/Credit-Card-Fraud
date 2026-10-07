"""Independent checks on everything the notebook exported for Tableau.

This recomputes the numbers from the exported CSVs and the raw data (no notebook objects), so a
mistake in the notebook's bookkeeping can't hide. Must end with ALL CHECKS PASSED.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import EXPECTED_FRAUD, EXPECTED_ROWS, FP_COST, PROCESSED, RAW_CSV, TABLEAU, TP_COST

SCORE_COL = {"XGBoost": "XGB_Prob", "Random Forest": "RF_Prob", "Logistic Regression": "LR_Prob",
             "Isolation Forest": "IForest_Score"}
failures = 0


def check(ok, msg):
    global failures
    failures += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {msg}")


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def confusion(y, score, amount, t):
    f = score >= t
    tp, fp, fn = f & (y == 1), f & (y == 0), ~f & (y == 1)
    review = FP_COST * fp.sum() + TP_COST * tp.sum()
    return dict(TP=int(tp.sum()), FP=int(fp.sum()), FN=int(fn.sum()), TN=int((~f & (y == 0)).sum()),
                FraudMissedEUR=amount[fn].sum(), ReviewCost=review, TotalCost=amount[fn].sum() + review)


raw = pd.read_csv(RAW_CSV)
raw.insert(0, "TxnID", np.arange(1, len(raw) + 1))
splits = pd.read_csv(PROCESSED / "splits.csv")
scored = pd.read_csv(TABLEAU / "scored_transactions.csv")
curve = pd.read_csv(TABLEAU / "threshold_curve.csv")
comp = pd.read_csv(TABLEAU / "model_comparison.csv")
imp = pd.read_csv(TABLEAU / "feature_importance.csv")
summ = pd.read_csv(TABLEAU / "class_summary.csv")
roc = pd.read_csv(TABLEAU / "roc_curve.csv")
metrics = json.loads((PROCESSED / "metrics.json").read_text())

print("Raw data")
check(len(raw) == EXPECTED_ROWS and raw.Class.sum() == EXPECTED_FRAUD, f"{len(raw):,} rows, {raw.Class.sum()} frauds")

print("\nSplit (stratified 60/20/20, no overlap)")
check(splits.TxnID.is_unique and len(splits) == len(raw), "every transaction is in exactly one split")
s = splits.merge(raw[["TxnID", "Class"]], on="TxnID").groupby("Split")["Class"].agg(["size", "mean"])
for name, share in (("train", 0.6), ("validation", 0.2), ("test", 0.2)):
    check(abs(s.loc[name, "size"] / len(raw) - share) < 0.001 and abs(s.loc[name, "mean"] - raw.Class.mean()) < 0.0002,
          f"{name}: {s.loc[name, 'size']:,} rows ({s.loc[name, 'size'] / len(raw):.1%}), fraud rate {s.loc[name, 'mean']:.4%}")

print("\nScored test set")
test_ids = set(splits.loc[splits.Split == "test", "TxnID"])
check(set(scored.TxnID) == test_ids and scored.TxnID.is_unique, f"scored_transactions = the {len(test_ids):,} test rows, no train/validation rows")
m = scored.merge(raw[["TxnID", "Class", "Amount"]], on="TxnID")
check((m.TrueLabel == m.Class).all() and np.allclose(m.Amount_x, m.Amount_y), "labels and amounts match the raw file")
check(scored[list(SCORE_COL.values())].stack().between(0, 1).all(), "all scores are between 0 and 1")

y, amount = scored.TrueLabel.to_numpy(), scored.Amount.to_numpy()
test_rows = comp[comp.Stage == "test"].set_index("Model")
val_rows = comp[comp.Stage == "validation"].set_index("Model")
print("\nModel metrics recomputed from scored rows (test)")
for model, col in SCORE_COL.items():
    sc = scored[col].to_numpy()
    r = test_rows.loc[model]
    check(close(average_precision_score(y, sc), r.PR_AUC, 1e-3) and close(roc_auc_score(y, sc), r.ROC_AUC, 1e-3),
          f"{model}: PR-AUC {r.PR_AUC:.4f}, ROC-AUC {r.ROC_AUC:.4f}")
    t = r.Threshold
    check(close(t, val_rows.loc[model, "Threshold"]) and close(t, metrics["thresholds"][model]),
          f"{model}: test uses the cutoff chosen on validation ({t:.3f})")
    c = confusion(y, sc, amount, t)
    same = all(c[k] == r[k] for k in ("TP", "FP", "FN", "TN")) and close(c["TotalCost"], r.TotalCost, 1e-4)
    check(same, f"{model}: TP {c['TP']} FP {c['FP']} FN {c['FN']}, total cost EUR {c['TotalCost']:,.2f}")
    row = curve[(curve.Model == model) & curve.IsChosenThreshold]
    check(len(row) == 1 and int(row.TP.iloc[0]) == c["TP"] and int(row.FP.iloc[0]) == c["FP"],
          f"{model}: threshold_curve row at the chosen cutoff agrees")

print("\nThreshold curve spot checks (drives the Tableau PR and cost curves)")
ok = True
for model, col in SCORE_COL.items():
    for t in (0.05, 0.25, 0.5, 0.75, 0.95):
        row = curve[(curve.Model == model) & np.isclose(curve.Threshold, t)].iloc[0]
        c = confusion(y, scored[col].to_numpy(), amount, t)
        ok &= c["TP"] == row.TP and c["FP"] == row.FP and close(c["TotalCost"], row.TotalCost, 1e-4)
check(ok, "20 model x cutoff points recomputed from scored rows match threshold_curve.csv")
fraud_total = amount[y == 1].sum()
check(np.allclose(curve.TotalCost, curve.FraudMissedEUR + curve.ReviewCost, atol=0.01) and
      np.allclose(curve.NetSavings, fraud_total - curve.TotalCost, atol=0.01),
      f"cost identities hold (total = missed + review; savings = EUR {fraud_total:,.2f} - total)")
check(roc.groupby("Model").apply(lambda d: d.FPR.is_monotonic_increasing and d.TPR.is_monotonic_increasing).all(),
      "ROC curves are monotonic")

print("\nOther extracts")
check(summ.Transactions.sum() == EXPECTED_ROWS and summ.loc[summ.ClassName == "Fraud", "Transactions"].sum() == EXPECTED_FRAUD
      and close(summ.TotalAmount.sum(), raw.Amount.sum(), 1e-6), "class_summary totals match the full dataset")
check(len(imp) == 30 and sorted(imp.Rank) == list(range(1, 31)), "feature_importance has 30 features ranked 1-30")
winner = test_rows.PR_AUC.idxmax()
check(winner == metrics["winner"], f"winner = highest test PR-AUC: {winner}")
study = comp[comp.Stage.str.contains("imbalance")]
check(set(study.Imbalance) == {"No adjustment", "Class weights", "SMOTE", "Undersampling"}, "imbalance study has all 4 techniques")

print("\n" + ("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED"))
sys.exit(1 if failures else 0)
