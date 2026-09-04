# Fraud Detection & Rare-Event Classification Workbench

An interactive machine-learning workbench for exploring **rare-event classification**, class imbalance and operational decision thresholds.

The project began as a money-laundering detection case study in an online gambling context, using a fully synthetic dataset (25,000 accounts, ~1.5% flagged as suspicious — see [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for the feature set and generation approach). It has since been developed into an interactive Streamlit application that can also be used with other binary classification datasets.

The central question is:

> **How do you detect enough rare events without overwhelming investigators with false alarms?**

Rather than optimising raw accuracy, the workflow focuses on the operational trade-off between **precision, recall, F1 score and alert rate**.

🚀 **Live app: [https://fraud-detection-workbench.streamlit.app](https://fraud-detection-workbench.streamlit.app)**

---

## Key Results: Money Laundering Case Study

For the synthetic gambling AML dataset, XGBoost with class weighting produced approximately:

- **Precision:** 20.0%
- **Recall:** 25.3%
- **F1 score:** 0.224
- **Alert rate:** 1.90%

This means:

- roughly 1 in 5 flagged accounts show genuinely suspicious activity
- approximately 1 in 4 truly suspicious accounts are detected
- around 1.9% of all accounts are sent for review

This illustrates why accuracy alone is a poor metric for rare-event detection — and why, even with real signal in the data (this dataset was validated to have genuine, learnable structure rather than noise), rare-event classification remains a fundamentally difficult, trade-off-driven problem rather than one with a "solved" answer.


---

## Iteration: Does Further Modelling Help?

Four iteration strategies were tested against the XGBoost baseline reported
above, to see whether the remaining ~3 in 4 undetected suspicious accounts
could realistically be recovered.

**1. Better model tuning — helped meaningfully**
A more carefully tuned GradientBoostingClassifier (`n_estimators=300,
max_depth=3, learning_rate=0.05`, class-balanced via `sample_weight`)
improved recall from 25.3% to 48.0% and PR AUC from 0.118 to 0.224 —
nearly double on both — at a comparable precision (19.5% vs. 20.0%). ROC
AUC also improved (0.848 → 0.897). This model is now available in the
workbench alongside XGBoost, Logistic Regression and Decision Tree, and
catches roughly twice as many genuinely suspicious accounts as the
original model, at the cost of a higher alert rate (~3.7% vs. 1.9%).

**2. Ensembling — did not help**
A soft-voting ensemble of RandomForest, GradientBoosting and Logistic
Regression underperformed the best single model (F1 0.255 vs. 0.323),
since weaker constituent models diluted rather than complemented the stronger one.

**3. Cascade / second-look modelling — failed for an informative reason**
A second-stage model trained specifically on the first model's
"uncertain-probability" predictions found essentially no true positives in
that band — indicating the missed cases aren't sitting in a recoverable
middle ground the model is unsure about; they're either confidently (and
wrongly) scored as negative, or the signal for them genuinely isn't in the
data.

**4. Peel-and-retrain — actively hurt performance**
A natural question: after flagging the top-scored accounts for
investigation, does removing them and re-running detection on the
remainder surface a meaningful second wave of suspects? Tested directly:

| Approach | True positives per 25-account batch | Cumulative recall after 2 rounds |
|---|---|---|
| Extend the same original ranked list | 6/25 (24%) | 22.7% |
| Remove "easy" cases, retrain fresh, re-rank remainder | 2/25 (8%) | 17.3% |

Retraining on the remainder performed *worse*, not better. Removing the
most confidently-scored training cases to force the model toward "harder"
examples also removed a disproportionate share of the (already rare)
positive training examples — the positive rate in the retrained subset
dropped from 1.50% to 1.04%, leaving the model with less signal, not more
insight. There is no second, distinct fraud pattern hiding behind the
obvious cases in this dataset — just fewer positive examples to learn from.

**Why there's a hard ceiling regardless of approach**
The synthetic dataset's label is deterministically generated as a weighted
combination of risk features *plus* injected random noise (see
`generate_dataset.py`). Decomposing that construction shows roughly
**two-thirds of the total variance in the label is pure, feature-independent
noise** — meaning a meaningful share of accounts near the decision boundary
are fundamentally unrecoverable from the available features, no matter how
sophisticated the modelling. Iteration can close part of the gap between a
naive model and this ceiling (as tuning did here), but cannot close all of
it — and some intuitive-sounding strategies (ensembling, cascading,
peel-and-retrain) can make things worse rather than better if applied
without first checking whether they address the actual source of the
remaining error.

---

## Interactive Workbench

The Streamlit application allows users to:

- use the bundled money-laundering case study
- upload their own CSV dataset
- load a public CSV from a URL
- select the binary target and positive class
- exclude identifier or non-predictive columns
- explore predictor distributions by class
- compare alternative classifiers
- compare class-imbalance strategies
- inspect precision, recall, F1 and alert workload
- explore confusion matrices, precision-recall curves and ROC curves
- examine predicted-probability separation
- inspect model feature importance
- dynamically tune the classification threshold

For uploaded datasets, text target labels are cleaned automatically to reduce problems caused by accidental leading or trailing whitespace.

---

## Why Rare Events Are Difficult

Rare-event classification creates several challenges:

- the majority class can dominate model training
- standard accuracy can appear extremely high even when no rare events are detected
- false positives create operational workload
- false negatives allow important events to go undetected
- the best classification threshold depends on the operational objective

For the money-laundering case study, a model predicting every account as legitimate would achieve approximately **98.5% accuracy while identifying no suspicious activity at all**.

---

## Modelling Workflow

### Data preparation

- removes identifier/non-predictive fields from modelling
- uses numeric predictors
- handles missing numeric values using median imputation
- removes constant predictors
- performs the train/test split before any resampling
- preserves the natural class distribution in the held-out test set

### Class-imbalance strategies

The workbench compares:

- **Class weighting** — gives greater importance to minority-class observations without changing the training data
- **SMOTE** — creates synthetic minority-class examples within the training set
- **Under-sampling** — reduces the number of majority-class training observations

Resampling is applied only to training data to avoid data leakage.

### Classifiers

The interactive workbench currently includes:

- **XGBoost**
- **Logistic Regression**
- **Decision Tree**

These provide a useful comparison between high-performance ensemble learning, an interpretable linear baseline and a simple non-linear tree model.

---

## Threshold Optimisation

A classifier produces probabilities rather than inherently deciding which observations should become alerts.

Instead of automatically using the conventional **0.5 threshold**, the workbench evaluates candidate thresholds from **0.01 to 0.99**.

The initial threshold is set to the value that maximises:

**F1 = harmonic mean of precision and recall**

This provides a useful starting point because a high F1 requires both reasonable detection of positive cases and reasonable precision.

Users can then move away from that threshold to explore the operational trade-off between:

- precision
- recall
- false positives
- missed positive cases
- investigation workload

For the money-laundering case study, the F1-maximising threshold for XGBoost is approximately **0.65**.

---

## Evaluation

### Confusion Matrix — Money Laundering Case Study


| | **Predicted Legitimate** | **Predicted Suspicious** |
|---|---:|---:|
| **Actual Legitimate** | TN = 4776 | FP = 149 |
| **Actual Suspicious** | FN = 39 | TP = 36 |

![Confusion matrix](assets/images/results.png)

Because the dataset is extremely imbalanced, the normalised confusion matrix contains a very large true-negative proportion. This reflects the underlying class distribution rather than near-perfect overall model performance.

![Model metrics](assets/images/metrics.png)

---

## Key Results: Money Laundering Case Study

These figures are produced directly by the workbench's own evaluation
workflow (see `pipeline.py`) — the same numbers and plots any user sees
when running the built-in case study through the app.

For the synthetic gambling AML dataset, **Gradient Boosting with class
weighting** (via `sample_weight`) gave the strongest result of the models
compared:

- **Precision:** 19.5%
- **Recall:** 48.0%
- **F1 score:** 0.277
- **Alert rate:** ~3.7%
- **ROC AUC:** 0.897
- **PR AUC:** 0.224

For comparison, the original XGBoost model achieved 20.0% precision, 25.3%
recall, F1 = 0.224, ROC AUC 0.848, PR AUC 0.118 at a 1.90% alert rate. Both
are valid operating points — Gradient Boosting roughly **doubles recall for
a comparable precision**, at the cost of a higher alert (investigation)
rate; which is preferable depends on available investigation capacity (see
Threshold Optimisation, below).

This means, using the Gradient Boosting result:

- roughly 1 in 5 flagged accounts show genuinely suspicious activity
- nearly half of all truly suspicious accounts are detected
- around 3.7% of all accounts are sent for review

**Why these numbers are stronger than they first appear:** with a base
rate of only 1.5% suspicious accounts, a model with no discriminative
power would achieve roughly 1.5% precision — pure chance. The observed
19.5% precision represents a **13.3x lift over random selection**. The
ROC AUC of 0.897 indicates the model separates suspicious from legitimate
accounts substantially better than chance (0.5) across all possible
thresholds, independent of any single operating point.

The model's predicted probabilities also separate cleanly between classes:
suspicious accounts have a mean predicted probability of 0.64 (95% CI:
0.59–0.69), while non-suspicious accounts average 0.175 (95% CI tight
around this value) — a substantial, well-separated gap even though
individual-case classification remains genuinely hard at this level of
class imbalance.

Precision and recall are tied to one specific decision threshold; ROC AUC
and PR AUC describe the model's overall discriminative power regardless of
where that threshold is set — useful for judging the underlying model
separately from the specific alert-rate trade-off chosen for this case
study. PR AUC in particular is the more informative of the two for a
problem this imbalanced, since ROC AUC can look deceptively strong even
for weak rare-event classifiers.

This illustrates why accuracy alone is a poor metric for rare-event
detection — and why, even with real signal in the data (this dataset was
validated to have genuine, learnable structure rather than noise),
rare-event classification remains a fundamentally difficult, trade-off-
driven problem rather than one with a "solved" answer.

## Key Insight

**Real-world model performance is about trade-offs, not perfection.**

A useful rare-event classifier must balance:

- catching enough positive cases (**recall**)
- ensuring alerts are meaningful (**precision**)
- balancing both objectives (**F1**)
- keeping the number of cases requiring review operationally realistic (**alert rate**)

The appropriate threshold therefore depends not only on statistical performance, but also on the cost of false positives, false negatives and investigation capacity.

---


## Data

The bundled case study uses a fully synthetic dataset generated for this project — not derived from any real gambling platform, company dataset or interview assignment. See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for the full feature list and generation methodology.

---

## Technologies

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- XGBoost
- imbalanced-learn
- SciPy
- Matplotlib

---

## Future Improvements

- cost-sensitive optimisation using explicit false-positive and false-negative costs
- model calibration
- additional classifiers
- repeated cross-validation for smaller datasets
- richer feature engineering
- deployment as a production scoring pipeline (see the companion [fraud-detection-api](https://github.com/steviecurran/fraud-detection-api) project)
