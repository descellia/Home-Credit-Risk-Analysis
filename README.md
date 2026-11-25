# Home Credit Default Risk Prediction (Indonesia)

Binary classification of loan default risk on the **Home Credit Indonesia** dataset, built as the final project for **Machine Learning**.

The project covers the full tabular ML workflow: multi-table merging, cleaning, encoding, correlation-driven feature extraction with PCA, filter and wrapper feature selection, class-imbalance handling, and a stratified cross-validation benchmark across several classifiers.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1.1-red)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.14.0-purple)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Pipeline](#pipeline)
- [Results](#results)
- [Getting Started](#getting-started)
- [Methodological Notes and Limitations](#methodological-notes-and-limitations)
- [Future Work](#future-work)
- [Team](#team)
- [Acknowledgements](#acknowledgements)

---

## Problem Statement

Lenders need to estimate whether an applicant will have difficulty repaying a loan. The target variable `TARGET` is binary:

| Value | Meaning |
|---|---|
| `1` | Client had payment difficulties (default risk) |
| `0` | All other cases |

The dataset is heavily imbalanced, so accuracy alone is misleading. A model that always predicts `0` already reaches about 92% accuracy. This project therefore uses **ROC-AUC** as the main selection metric and reports **Precision, Recall, F1, and MCC** alongside it.

## Dataset

Source: [Home Credit Indonesia Default Risk (Kaggle)](https://www.kaggle.com/datasets/dwiwahyuef/home-credit-indonesia-default-risk), downloaded with `kagglehub`.

The raw data is not included in this repository because of its size. See [Getting Started](#getting-started) for how to download it.

| File | Rows (raw) | Description |
|---|---:|---|
| `application_train.csv` | 307,511 | Main application table with `TARGET` |
| `application_test.csv` | 48,744 | Applications without labels (not used for evaluation) |
| `bureau.csv` | 1,716,428 | Previous credits from other institutions |
| `bureau_balance.csv` | 27,299,925 | Monthly balance of bureau credits |
| `previous_application.csv` | 1,670,214 | Previous Home Credit applications |
| `POS_CASH_balance.csv` | 10,001,358 | Monthly POS and cash loan snapshots |
| `credit_card_balance.csv` | 3,840,312 | Monthly credit card snapshots |
| `installments_payments.csv` | 13,605,401 | Repayment history |

Class distribution in `application_train.csv`:

| Class | Count | Share |
|---|---:|---:|
| 0 | 282,686 | 91.93% |
| 1 | 24,825 | 8.07% |

## Repository Structure

```
.
├── Home-Credit-Risk.ipynb                                              # Main pipeline: multi-table merge, PCA, RFE, 10-fold CV benchmark, final model
├── IF441-Machine-Learning-Project-Home-Credit-Risk-Final-Project.ipynb  # Course notebook: application table only, per-model RFE, SMOTE/ADASYN/cost-sensitive comparison
├── requirements.txt                                                    # Pinned dependencies
├── .gitignore
└── README.md
```

The two notebooks are separate experiments. **`Home-Credit-Risk.ipynb`** is the more complete pipeline and produces the final model reported below. The **IF441** notebook is the course submission version, which works on the application table only and compares imbalance strategies per model.

## Pipeline

The steps below describe `Home-Credit-Risk.ipynb`.

```
Raw CSVs ─► Deduplicate & left-join ─► Stratified 80/20 split ─► Feature typing
        ─► Numeric: mean impute + MinMax scale ─► Categorical: fill "Missing" + One-Hot (drop first)
        ─► PCA on highly correlated pairs (|r| > 0.8) ─► SelectKBest (ANOVA / chi²)
        ─► RFE (15 features) ─► 4 resamplers × 5 models, 10-fold Stratified CV
        ─► Best combo by ROC-AUC ─► Retrain on full train ─► Evaluate on held-out test
```

### 1. Merging

Each auxiliary table is deduplicated on its join key(s), keeping the first record, then left-joined onto `application_train.csv` via `SK_ID_CURR`, `SK_ID_BUREAU`, and `SK_ID_PREV`.

### 2. Train/test split

Stratified 80/20 split (`random_state=42`) is performed **before** any fitting, so imputers, scalers, encoders, PCA, and selectors are fitted on the training portion only.

### 3. Feature typing

A helper function groups columns into **numeric** (135 after merging), **categorical**, and **boolean** (0/1) features, and drops ID-like columns.

### 4. Cleaning and encoding

- Numeric: mean imputation, then `MinMaxScaler`.
- Categorical: missing values filled with `"Missing"`, then `OneHotEncoder(drop="first")` to avoid the dummy variable trap.
- Boolean: missing values filled with `0`.

### 5. Feature extraction with PCA

Pairs of numeric features with absolute Pearson correlation above 0.8 are each replaced by a single PCA component. This produced 33 combined features, for example `PCA_AMT_GOODS_PRICE_x_AMT_CREDIT_x` (r = 0.99).

### 6. Feature selection

| Stage | Method | Output |
|---|---|---|
| Numeric filter | `SelectKBest(f_classif, k=10)` | 10 features |
| Categorical filter | `SelectKBest(chi2, k=10)` | 10 features |
| Boolean filter | `SelectKBest(chi2, k=10)` | 10 features |
| Wrapper | `RFE(LogisticRegression, n_features_to_select=15)` | 15 final features |

Final 15 features after RFE:

`DAYS_EMPLOYED`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`, `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`, `DAYS_LAST_PHONE_CHANGE`, `NAME_CONTRACT_TYPE_x_Revolving loans`, `CODE_GENDER_M`, `NAME_INCOME_TYPE_Pensioner`, `NAME_EDUCATION_TYPE_Higher education`, `ORGANIZATION_TYPE_XNA`, `FLAG_EMP_PHONE`, `FLAG_DOCUMENT_13`, `FLAG_DOCUMENT_16`

### 7. Imbalance handling and model benchmark

Every combination of resampler and model is evaluated with **10-fold Stratified CV** on the training set. Resampling is applied **inside each fold** to the training portion only, so validation folds keep the original class distribution.

| Resampling | Models |
|---|---|
| SMOTE, ADASYN (oversampling) | Logistic Regression |
| Tomek Links, ENN (undersampling) | Decision Tree |
| | XGBoost |
| | LightGBM |
| | MLP |

The loop saves a checkpoint (`checkpoint.pkl`) after every fold, so a long run can resume after an interruption.

## Results

### Final model (held-out test set)

The combination with the highest fold-level validation ROC-AUC was **ENN + LightGBM**. It was retrained on the full training set and evaluated once on the held-out 20% test set (61,503 rows).

| Metric | Score |
|---|---:|
| ROC-AUC | **0.740** |
| Accuracy | 0.913 |
| Precision (class 1) | 0.371 |
| Recall (class 1) | 0.110 |
| F1 (class 1) | 0.170 |
| MCC | 0.167 |

```
              precision    recall  f1-score   support
           0       0.93      0.98      0.95     56538
           1       0.37      0.11      0.17      4965
    accuracy                           0.91     61503
```

**How to read this:** the model ranks risky applicants reasonably well (ROC-AUC 0.74), but at the default 0.5 threshold it catches only about 11% of defaulters. In a lending context, the decision threshold should be tuned on validation data to fit the business trade-off between missed defaulters and rejected good applicants.

### Imbalance strategy comparison (IF441 notebook, 5-fold CV)

Mean scores across 5 folds, using application-table features only:

| Model | Strategy | ROC-AUC | Recall | F1 | MCC |
|---|---|---:|---:|---:|---:|
| LightGBM | Cost-sensitive | **0.739** | 0.664 | **0.255** | **0.203** |
| Logistic Regression | SMOTE | 0.726 | 0.654 | 0.246 | 0.190 |
| Logistic Regression | Cost-sensitive | 0.726 | 0.653 | 0.246 | 0.189 |
| XGBoost | Cost-sensitive | 0.720 | 0.631 | 0.246 | 0.187 |
| XGBoost | SMOTE | 0.710 | 0.602 | 0.243 | 0.180 |
| LightGBM | SMOTE | 0.702 | 0.534 | 0.242 | 0.172 |
| Random Forest | Cost-sensitive | 0.702 | 0.007 | 0.013 | 0.048 |
| Random Forest | SMOTE | 0.679 | 0.278 | 0.221 | 0.140 |
| Decision Tree | SMOTE | 0.559 | 0.321 | 0.177 | 0.079 |

Key observations:

- **Boosting models and logistic regression outperform single trees.** A single Decision Tree stays close to random ranking (ROC-AUC ≈ 0.53 to 0.56).
- **Class weighting is competitive with synthetic oversampling**, and is simpler and cheaper to run.
- **`EXT_SOURCE_1/2/3`** (external credit scores) are selected by every method, making them the most consistently informative features.
- **High accuracy is not a useful signal here.** Random Forest with class weights reaches 91.9% accuracy while detecting under 1% of defaulters.

## Getting Started

### 1. Clone and set up the environment

```bash
git clone https://github.com/descellia/Home-Credit-Risk-Analysis.git
cd Home-Credit-Risk-Analysis

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download the dataset

You need a Kaggle account and API token (`~/.kaggle/kaggle.json`, or the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables).

```python
import kagglehub
path = kagglehub.dataset_download("dwiwahyuef/home-credit-indonesia-default-risk")
print(path)
```

### 3. Update file paths

The notebooks currently read CSVs from a hard-coded Windows path (`C:/Users/USER/.cache/kagglehub/...`). Replace it with the `path` returned above, for example:

```python
import os
df_train = pd.read_csv(os.path.join(path, "application_train.csv"))
```

### 4. Run the notebooks

```bash
jupyter notebook Home-Credit-Risk.ipynb
```

The full 10-fold benchmark (4 resamplers × 5 models × 10 folds) takes a while on a laptop. Delete `checkpoint.pkl` to start a fresh run.

## Methodological Notes and Limitations

These points matter for interpreting the numbers above:

1. **Deduplication instead of aggregation.** Auxiliary tables keep only the first record per key. This discards most historical information (for example, `bureau.csv` goes from 1.7M to 306K rows). Aggregating with mean, max, count, and similar statistics per `SK_ID_CURR` is the standard approach and would likely improve performance.
2. **Feature selection outside the CV loop.** SelectKBest and RFE are fitted on the full training set before cross-validation. Because of this, CV scores may be slightly optimistic. The held-out test set is not affected, since it was split off first.
3. **Model selection by a single fold.** The best combination is picked from the single highest fold-level ROC-AUC rather than the mean across folds. Selecting by mean (± std) would be more robust.
4. **Default decision threshold.** All threshold-based metrics use 0.5. Precision and recall depend heavily on this choice for imbalanced data.
5. **ROC-AUC in the IF441 baseline cells.** In the single-split baseline cells of the IF441 notebook (Random Forest, Decision Tree, SMOTE variants), `y_prob` is computed from the Logistic Regression model, so the ROC-AUC reported there (0.7345) does not reflect those models. The 5-fold comparison table above uses each model's own probabilities and is not affected.
6. **`application_test.csv` is unlabeled** and was not used for evaluation. All reported test metrics come from the internal 20% split.

## Future Work

- Aggregate auxiliary tables per client instead of deduplicating.
- Wrap preprocessing, selection, resampling, and the model in a single `imblearn.pipeline.Pipeline` so every step is refit inside each fold.
- Tune hyperparameters and the decision threshold on validation folds.
- Add SHAP-based interpretation of the final model.
- Report mean ± std across folds and use precision-recall curves alongside ROC.

## Team

IF441 Machine Learning, Class B, Universitas Multimedia Nusantara:

- Calvin Leo
- Descellia Ng
- Muiz
- Howard
- Theo

## Acknowledgements

Thanks to our lecturer **Dr. David Agustriawan, S.Kom., M.Sc., Ph.D.** for guidance throughout the project, and to the Kaggle dataset author for publishing the data.

The literature review used to find references and the research gap is available [here](https://docs.google.com/spreadsheets/d/1UUFPzZtpT_SAIC7mc16DE4hJyCBK8i_aWLWxOL_yRX4/edit?usp=sharing).
