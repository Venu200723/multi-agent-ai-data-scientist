# Multi-Agent AI Data Scientist — Pipeline Report

**Generated:** 2026-09-18 09:22  
**Source CSV:** `sample.csv`  
**Target:** `churn`  
**Task:** `classification`

## 1. Cleaning Agent

- Shape: `(403, 12)` → `(400, 12)`
- Imputed numeric 'total_charges' with median=1881
- Imputed categorical 'internet_service' with mode='Fiber'
- Removed 3 duplicate rows

## 2. EDA Agent

- Rows/cols: **400** × **12** (numeric=5, categorical=7)
- Missing overall: **0.00%**
- Target profile: `{'type': 'numeric', 'mean': 0.235, 'std': 0.4245298151011003, 'min': 0.0, 'max': 1.0}`
- Top correlations:
  - `tenure_months` ↔ `total_charges`: **0.803**
  - `monthly_charges` ↔ `total_charges`: **0.533**
  - `tenure_months` ↔ `churn`: **-0.248**
  - `monthly_charges` ↔ `churn`: **0.175**
  - `total_charges` ↔ `churn`: **-0.136**
- Figure: `artifacts/eda_histograms.png`
  ![eda_histograms.png](artifacts/eda_histograms.png)
- Figure: `artifacts/eda_correlation.png`
  ![eda_correlation.png](artifacts/eda_correlation.png)

## 3. ML Agent

- Train/test sizes: **300** / **100**
- Holdout metrics:
  - **accuracy**: `0.7800`
  - **f1_weighted**: `0.7080`
  - **roc_auc**: `0.6254`

## 4. Explain Agent

Offline summary — classification model holdout metrics: accuracy=0.780, f1_weighted=0.708, roc_auc=0.625. Strongest drivers by permutation importance: contract (0.022), tenure_months (0.012), internet_service (0.012). Use these features first when debugging predictions or collecting better data.

| Feature | Importance | Std |
|---|---:|---:|
| `contract` | 0.0220 | 0.0098 |
| `tenure_months` | 0.0120 | 0.0098 |
| `internet_service` | 0.0120 | 0.0075 |
| `monthly_charges` | 0.0100 | 0.0089 |
| `total_charges` | 0.0080 | 0.0075 |
| `tech_support` | 0.0040 | 0.0049 |
| `senior_citizen` | 0.0040 | 0.0049 |
| `partner` | 0.0040 | 0.0049 |
| `customer_id` | 0.0000 | 0.0000 |
| `payment_method` | 0.0000 | 0.0110 |

- Figure: `artifacts/feature_importance.png`
  ![feature_importance.png](artifacts/feature_importance.png)

## Agent action logs

### cleaning
- Imputed numeric 'total_charges' with median=1881
- Imputed categorical 'internet_service' with mode='Fiber'
- Removed 3 duplicate rows

### eda
- Computed correlations; top pair=('tenure_months', 'total_charges', 0.8025478914697876)
- Profiled target 'churn'
- Saved histograms → eda_histograms.png
- Saved correlation heatmap → eda_correlation.png

### ml
- Inferred task: classification
- Holdout metrics: {'accuracy': 0.78, 'f1_weighted': 0.708028293545535, 'roc_auc': 0.6253529079616037}

### explain
- Computed permutation importance (accuracy)
- Saved importance plot → feature_importance.png
- No OPENAI_API_KEY — using offline summary

---
_Produced by `multi_agent_ds` — Multi-Agent Autonomous AI Data Scientist._