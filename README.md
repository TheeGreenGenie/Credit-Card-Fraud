# Credit Card Fraud Detection on Highly Imbalanced Data

**Business question:** *Only 1 in 579 card transactions is fraud. How do you catch it without flagging thousands of honest customers, and is it worth the cost of reviewing the alerts?*

I trained and compared models on 284,807 real card transactions and picked the alert cutoff by **money**, not by a default setting. The results are in an interactive **Tableau** workbook with sliders for model, cutoff and review cost.

| | |
|---|---|
| **Data** | ULB / Worldline credit card fraud dataset: 284,807 transactions over 2 days (Sept 2013), 492 frauds (0.17%) |
| **Tools** | Python (pandas, scikit-learn, imbalanced-learn, XGBoost, SHAP), Tableau |
| **Deliverables** | [**Live Tableau dashboard**](https://public.tableau.com/app/profile/solomon.shasanmi/viz/CreditCardFraudAnalysis_17914054926190/Story1) · [Notebook](notebooks/fraud_detection.ipynb) · [Tableau workbook](tableau/Fraud_Detection.twbx) · [Findings report (PDF)](output/Fraud_Detection.pdf) (5 pages) · cost-benefit analysis across every cutoff |

---

## Results (final test set, never seen during training or tuning)

The test set holds 56,962 transactions, **99 of them fraud** (€9,874 at risk). Each model uses the cutoff it chose on the validation set.

| Model | PR-AUC | Frauds caught | False alarms | Precision | Recall | Net savings |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost** | **0.878** | **84 / 99** | **10** | **89%** | **85%** | **€6,426** |
| Random Forest | 0.877 | 81 / 99 | 10 | 89% | 82% | €6,246 |
| Logistic Regression (SMOTE) | 0.766 | 83 / 99 | 30 | 73% | 84% | €6,237 |
| Isolation Forest (no labels) | 0.147 | 24 / 99 | 119 | 17% | 24% | €2,567 |

*Net savings = fraud stopped minus review costs, compared with running no model at all.*

**In plain terms:** XGBoost caught 85% of the fraud, and 9 out of 10 of its alerts were real. It flagged only 10 honest transactions out of 56,863 and saved about **65%** of the money that would otherwise have been lost.

---

## Recommendations

1. **Run XGBoost with a cost-based cutoff of about 0.26** (at a €10 review cost). That's the cheapest setting found on the validation set, and it held up on the test set.
2. **Pick the cutoff by cost, never by the default 0.5.** The right cutoff depends on what a review costs and what a missed fraud costs. The good news: the cost curve has a wide, flat valley. On the test set, every cutoff from 0.10 to 0.90 costs within about **€170** of the cheapest point, and the total fraud at risk is €9,874. So a team that hates false alarms can raise the cutoff without losing much.
3. **Don't count on resampling to fix the imbalance.** SMOTE and undersampling barely improved how well the model separates fraud from normal (see below). They mostly shifted where it draws the line. Mild class weighting plus a tuned cutoff did the job.
4. **Don't read a resampled model's score as a real probability.** SMOTE pushed logistic regression's scores so high that its best cutoff landed at 0.999. Scores from resampled models are rankings, not odds.
5. **Pay extra attention to small overnight purchases.** Fraud runs about **10× the normal rate at 2am**, and about **half of all frauds are under €10**, compared with a third of normal purchases.
6. **Monitor and retrain.** Each month, track precision and recall on reviewed alerts, and retrain when they drift, because fraud patterns change. With labels, keep using the supervised models: the unsupervised Isolation Forest caught only 24% here, so it's a stop-gap for when no labels exist yet.

---

## How I handled the imbalance

With 578 normal transactions for every fraud, a model that calls *everything* normal is **99.83% accurate** and catches nothing. So I made four choices:

**1. Better scorecards.**
- **PR-AUC** is the main metric. It measures how well a model ranks fraud above normal at every cutoff.
- **Precision and recall** are reported at the chosen cutoff.
- **Euros** are the final measure.
- **ROC-AUC** is reported but not trusted: at 0.17% fraud, it scores even the weak Isolation Forest at 0.95.

**2. A fair test of the imbalance fixes.** I held the model fixed (logistic regression) and changed only the imbalance technique. Results on the validation set, at the default 0.5 cutoff:

| Technique | PR-AUC | Precision | Recall |
|---|---:|---:|---:|
| No adjustment | 0.675 | 79% | 63% |
| Class weights | 0.685 | 6% | 88% |
| SMOTE (synthetic fraud examples) | 0.685 | 6% | 90% |
| Random undersampling | 0.638 | 4% | 88% |

PR-AUC barely moves, while precision collapses. **The fixes trade false alarms for catches; they don't make the model better at telling fraud apart.** That's why the cutoff is tuned separately.

**3. No leakage.** SMOTE and undersampling sit inside an `imblearn` pipeline, so they only ever touch **training** data. Validation and test are always real, untouched transactions.

**4. Weights for the tree models.**
- **XGBoost:** 5-fold cross-validation on the training set tested `scale_pos_weight` at the full 578:1 ratio and at its square root (about 24). The square root did as well or slightly better (CV PR-AUC 0.859 vs. 0.857), so full-strength weighting isn't needed.
- **Random Forest:** `class_weight="balanced_subsample"`.

---

## Method in one paragraph

The data is split **60 / 20 / 20** (train / validation / test), stratified so every part keeps the 0.17% fraud rate.
- **Learning and tuning:** models learn on train, and cutoffs are chosen on validation.
- **Scoring:** test is scored **once**, at the end.
- **Features:** V1–V28 (see the caveat below), log of amount, and hour of day written as sine/cosine.
- **Cost model:** a missed fraud costs its full amount; every alert costs €10 of review time. The €10 is an assumption, adjustable in Tableau.
- **Explainability:** **SHAP** explains each XGBoost prediction. V14 was the top reason behind 84 of the 94 alerts.
- **Checks:** `python/verify.py` recomputes every exported number from the raw data and the CSVs, and passes all checks.

---

## Tableau dashboards

**View it online:** [Tableau Public](https://public.tableau.com/app/profile/solomon.shasanmi/viz/CreditCardFraudAnalysis_17914054926190/Story1). No install needed.

Or open [`tableau/Fraud_Detection.twbx`](tableau/Fraud_Detection.twbx) in the free **[Tableau Desktop Public Edition](https://www.tableau.com/products/public/download)**. The data is packaged inside, so no other files are needed. It contains 7 dashboards and a story:

| Dashboard | What it shows |
|---|---|
| How rare is fraud? | 1 in 579, and the barely visible fraud bar |
| Fraud by time and amount | Fraud rate by hour (peak 1.7% at 2am) and amount mix |
| Which model finds fraud best? | Precision-recall and ROC curves, leaderboard |
| Drawing the line | Score histogram with a live cutoff line, confusion grid, precision/recall |
| Why was it flagged? | SHAP drivers, V14 vs V4 scatter |
| What it's worth | Alert queue with filters, € caught / missed / review cost / net savings |
| What it's worth 2 | Cost-vs-cutoff curve that moves with the review-cost slider |

**Interactive controls:** model, cutoff (0.001–0.999) and review cost (€). Static images of every dashboard are in [`tableau/images/`](tableau/images/).

---

## Reproduce it

Requires Python 3.13 (3.11 also works if `shap` won't install on 3.13). Commands are for Windows PowerShell, run from the `Fraud_Detection` folder.

**One-time setup:**

```powershell
py -3.13 -m venv FraudVenv
.\FraudVenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name FraudVenv --display-name "Python (FraudVenv)"
```

**Run the analysis** (with the venv active):

```powershell
python python\01_get_data.py      # downloads the data from OpenML and checks 284,807 rows / 492 frauds
cd notebooks
jupyter nbconvert --to notebook --execute --inplace fraud_detection.ipynb   # or open it and Run All
cd ..
python python\verify.py           # must end with: ALL CHECKS PASSED
```

The notebook writes the six CSV files that Tableau reads into `tableau/`. Results are reproducible, because every random step uses seed 42.

```
Fraud_Detection/
├── notebooks/fraud_detection.ipynb   full analysis, run top to bottom with outputs saved
├── python/                           01_get_data.py, verify.py, common.py (shared settings)
├── tableau/                          Fraud_Detection.twbx, CSV extracts, images/
├── output/                           Fraud_Detection.pdf and Findings_Report.html (5-page plain-language report)
├── data/                             raw/ (git-ignored, ~150 MB), processed/ (splits, metrics.json)
└── requirements.txt                  Python packages
```

---

## Limitations

- **Anonymised features.** V1–V28 are PCA components, released that way to protect cardholders. The model can say *which* signal raised the alarm, not what it means (for example "new device"). With a company's own data, SHAP would name real drivers.
- **Small and old.** The data covers two days in 2013, and the final test contains only 99 frauds. The exact numbers would shift with more data, and fraud patterns change over time.
- **Random split, not time-based.** Because the data spans only 48 hours, I used a stratified random split. In production, I'd validate on later weeks than the model was trained on.
- **Assumed costs.** The €10 review cost is an assumption. Missed-fraud cost uses the transaction amount only, not chargeback fees or lost customers. Both can be adjusted in the dashboard.
- **Duplicates kept.** 1,081 exact duplicate rows were kept to stay comparable with other work on this benchmark.

---

*Data: Machine Learning Group, Université Libre de Bruxelles (ULB), and Worldline, via [OpenML dataset 42175](https://www.openml.org/d/42175), a mirror of the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Original research: Dal Pozzolo et al., "Calibrating Probability with Undersampling for Unbalanced Classification," IEEE SSCI, 2015.*
