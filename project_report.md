# Loan Risk Classification Project — Technical Report

## Executive Summary

This project implements a multi-class classification system to predict loan grades (A through G) based on borrower and loan characteristics. The pipeline includes comprehensive feature engineering, traditional machine learning models (Random Forest, KNN, SVM), and deep learning approaches (FFNN, TabNet, TabTransformer, FTTransformer).

**Best Result:** Random Forest achieved **96.5% accuracy** and **0.957 F1 macro** using 3-fold cross-validation.

---

## 1. Dataset Overview

| Attribute | Value |
|-----------|-------|
| Total samples | 148,301 |
| Target variable | `grade` (A, B, C, D, E, F, G) |
| Original features | ~120 columns |
| Final features (after engineering) | 126 |
| Class distribution | Imbalanced (A-C majority, F-G minority ~5%) |

### Class Distribution
- Grade A: ~18%
- Grade B: ~25%
- Grade C: ~25%
- Grade D: ~14%
- Grade E: ~8%
- Grade F: ~5%
- Grade G: ~4%

---

## 2. Data Preprocessing & Feature Engineering

### 2.1 Missing Value Handling

#### High Missing Rate Columns (Dropped)
- **Threshold:** >80% missing values
- **Columns dropped:** 37 columns
- **Rationale:** Columns with >80% missing contain too little signal; imputation would dominate the real data

### 2.2 Collinearity Removal
- **Threshold:** Pearson correlation > 0.95
- **Columns dropped:** ~10 columns
- **Method:** For each pair with r > 0.95, drop one column

### 2.3 Specific Handling


#### Special Handling: `months_since_*` Columns
Four columns with 50-80% missing were **kept** because their missingness is informative:

| Column | Missing % | Handling |
|--------|-----------|----------|
| `months_since_last_delinquency` | 51.3% | Kept + `_ever` flag |
| `months_since_recent_revolving_delinquency` | 67.4% | Kept + `_ever` flag |
| `months_since_last_major_derog` | 74.1% | Kept + `_ever` flag |
| `months_since_recent_bankcard_delinquency` | 77.2% | Kept + `_ever` flag |

**Why:** Missing = "event never happened" = positive credit signal. Each column generates:
1. Original column imputed with `-1` (distinguishable from positive real values)
2. Binary `{col}_ever` flag (1 = had event, 0 = never)

This adds **8 features** (4 original + 4 flags).

#### Total Column Dropped
- `next_payment_date` — operational/administrative data, not predictive of loan grade
- `loan_purpose_category` — duplicate of a feature engineered new feature from loan_
- `loan_title` — after feature engineering
- `borrower_address_zip` — redundant with engineered features



#### Categorical Missing Values
- All remaining categorical columns with missing values → imputed with literal `"missing"` category
- This allows the model to learn if missingness itself is predictive



### 2.3 Feature Transformations

#### Hidden Numeric Columns
Two columns stored as strings were converted to numeric:

| Column | Original Format | Transformation |
|--------|-----------------|----------------|
| `loan_contract_term_months` | "36 months", "60 months" | Extract integer, impute mode |
| `borrower_profile_employment_length` | "< 1 year", "2 years", "10+ years" | Extract integer (0-10), impute median |

#### Date Columns — Cyclical Encoding
Four date columns were transformed to capture temporal patterns:

**Original columns -> Dropped after creating derived features:**
- `loan_issue_date`
- `credit_history_earliest_line`
- `last_payment_date`
- `last_credit_pull_date`

**Generated features (per date column):**
| Feature | Description |
|---------|-------------|
| `{col}_year` | Year as integer |
| `{col}_month_sin` | sin(2π × month / 12) — cyclical encoding |
| `{col}_month_cos` | cos(2π × month / 12) — cyclical encoding |
| `{col}_quarter` | Quarter (1-4) |
| `{col}_days_since_ref` | Days since reference date (2020-01-01) |

**Additional derived feature:**
- `credit_history_length_months` = (loan_issue_date - credit_history_earliest_line) / 30

#### Loan Status — Ordinal Risk Mapping
`loan_status_current_code` was converted from categorical to ordinal based on risk level:

| Status | Risk Score |
|--------|------------|
| Fully Paid | 0 |
| Current | 1 |
| In Grace Period | 2 |
| Late (16-30 days) | 3 |
| Late (31-120 days) | 4 |
| Default / Charged Off | 5 |

### 2.4 Geographic Feature Engineering

#### State Features
From `borrower_address_state`:
| Feature | Description |
|---------|-------------|
| `state_region` | Categorical: Northeast, Southeast, Midwest, South Central, Mountain, Pacific |
| `state_no_income_tax` | Binary: 1 if state has no income tax (AK, FL, NV, SD, TX, WA, WY) |
| `state_log_frequency` | log(1 + state_count) — captures state population density |
| `borrower_address_state` | Kept for target encoding (high cardinality: 51 values) |

#### ZIP Code Features
From `borrower_address_zip`:
| Feature | Description |
|---------|-------------|
| `zip3_prefix` | First 3 digits (for target encoding) |
| `zip_region` | Region based on first digit (0-9 → geographic regions) |
| `zip_log_frequency` | log(1 + zip3_count) |

### Dropped feature:
- `borrower_address_zip` — redundant with engineered features

### 2.5 Loan Title Processing (NLP-lite)
The free-text `loan_title` field was processed using pattern matching:

**Semantic Categories (12 categories):**
- debt_consolidation, credit_card, home_improvement, major_purchase
- medical, car, business, moving, home_buying, vacation, wedding, education
- other, other_uncategorized, missing

**Generated features:**
| Feature | Description |
|---------|-------------|
| `loan_category` | Semantic category (categorical) |
| `loan_title_frequency` | Raw frequency count in dataset |
| `loan_title_log_frequency` | log(1 + frequency) |
| `loan_title_is_custom` | Binary: 1 if frequency < 10 (custom/rare title) |
| `loan_title_word_count` | Number of words in title |
| `loan_title_has_numbers` | Binary: 1 if title contains digits |

### 2.6 Dropped Features
- `loan_title`
- `loan_purpose_category` — redundant with engineered `loan_category`

### 2.7 Skewness Analysis
Numeric features were analyzed for skewness:
| Category | Criteria | Count | Treatment |
|----------|----------|-------|-----------|
| Normal distribution | \|skew\| ≤ 1 | 35 | Scaler |
| Safe for log transform | \|skew\| > 1, no negatives | 71 | log1p + Scaler |
| Contains negatives | \|skew\| > 1, has negatives | 7 | PowerTransformer (Yeo-Johnson) +  Scaler |

---

## 3. Categorical Encoding Strategy

### ML Pipeline (Random Forest, KNN, SVM)
| Cardinality | Encoding | Columns |
|-------------|----------|---------|
| 2 values | OrdinalEncoder | Binary flags |
| 3-51 values | OneHotEncoder | state_region, zip_region, loan_category, etc. |
| >51 values | TargetEncoder | borrower_address_state, zip3_prefix |

### DL Pipeline (FFNN, TabNet, TabTransformer)
- All categoricals → OrdinalEncoder (integers for embedding layers)
- Unknown categories → mapped to index 0 (via +1 shift)
- Cardinalities stored for embedding dimension sizing

---

## 4. Train/Test Split Strategy

### ML Models (RF, KNN, SVM)
- **Method:** 3-fold Stratified Cross-Validation on full dataset
- **Rationale:** Maximizes training data, provides variance estimates
- **Metrics:** Reported as mean ± std across folds

### DL Models (FFNN, TabNet, TabTransformer, FTTransformer)
- **Method:** Fixed 80/10/10 stratified split (train/val/test)
- **Rationale:** DL needs explicit validation set for early stopping
- **Split sizes:**
  - Train: 118,639 samples (80%)
  - Validation: 14,831 samples (10%)
  - Test: 14,831 samples (10%)

All splits use **stratification** to maintain class distribution.

---

## 5. Class Imbalance Handling

### Approach: Class Weights
All models use **balanced class weights** computed as:
```
weight[c] = n_samples / (n_classes × n_samples_in_class[c])
```

### Implementation by Model
| Model | Parameter |
|-------|-----------|
| Random Forest | `class_weight="balanced_subsample"` |
| SVM | `class_weight="balanced"` |
| KNN | N/A (distance-based, no weights) |
| FFNN | `CrossEntropyLoss(weight=class_weights)` |
| TabNet | `weights={class: weight}` in `.fit()` |
| TabTransformer | `CrossEntropyLoss(weight=class_weights)` |

---

## 6. Model Architectures & Hyperparameters

### 6.1 Random Forest
```python
n_estimators = 500
max_depth = None (unlimited)
min_samples_split = 8
min_samples_leaf = 1
max_features = 0.3 (30% of features per split)
class_weight = "balanced_subsample"
```

### 6.2 K-Nearest Neighbors
```python
n_neighbors = 30
weights = "distance"
p = 2 (Euclidean distance)
# Preprocessing:
PCA(n_components=40)  # Dimensionality reduction
StandardScaler()
```

### 6.3 Support Vector Machine
```python
kernel = "rbf"
C = 100
gamma = "scale"
class_weight = "balanced"
# Preprocessing:
StandardScaler()
```

### 6.4 Feed-Forward Neural Network (FFNN)
```python
Architecture:
  Input(126) → Linear(256) → LayerNorm → LeakyReLU → Dropout(0.3)
            → Linear(128) → LayerNorm → LeakyReLU → Dropout(0.3)
            → Linear(64)  → LayerNorm → LeakyReLU → Dropout(0.05)
            → Linear(7)   → Softmax

Training:
  optimizer = AdamW(lr=3e-4, weight_decay=1e-4)
  scheduler = ReduceLROnPlateau(patience=5, factor=0.3)
  batch_size = 256
  max_epochs = 300
  early_stopping_patience = 25
```

### 6.5 TabNet
```python
Architecture:
  n_d = 64 (decision layer width)
  n_a = 64 (attention layer width)
  n_steps = 5 (sequential attention steps)
  gamma = 1.5 (coefficient for feature reusage)
  n_independent = 2
  n_shared = 2
  cat_emb_dim = 8 (categorical embedding dimension)

Training:
  optimizer = Adam(lr=0.02)
  scheduler = StepLR(step_size=50, gamma=0.9)
  lambda_sparse = 1e-4 (sparsity regularization)
  batch_size = 1024
  virtual_batch_size = 128 (Ghost Batch Normalization)
  max_epochs = 200
  patience = 20
```

### 6.6 TabTransformer
```python
Architecture:
  dim = 32 (embedding dimension)
  depth = 3 (transformer layers)
  heads = 4 (attention heads)
  attn_dropout = 0.3
  ff_dropout = 0.3
  mlp_hidden_mults = (4, 2)
  
Note: Attention only on categorical features (13 columns).
      Numerical features (113 columns) concatenated before MLP.

Training:
  optimizer = AdamW(lr=3e-4, weight_decay=5e-3)
  scheduler = CosineAnnealingLR(T_max=100)
  warmup_epochs = 5
  max_grad_norm = 1.0 (gradient clipping)
  batch_size = 1024
  max_epochs = 100
  patience = 30
```

### 6.7 FTTransformer (Feature Tokenizer Transformer)
```python
Architecture:
  dim = 64 (embedding dimension)
  depth = 4 (transformer layers)
  heads = 8 (attention heads)
  attn_dropout = 0.2
  ff_dropout = 0.2
  
Note: Unlike TabTransformer, applies attention to ALL features
      (both categorical and numerical → 118 total tokens).

Training:
  optimizer = AdamW(lr=1e-4, weight_decay=1e-3)
  scheduler = CosineAnnealingLR(T_max=100)
  warmup_epochs = 5
  max_grad_norm = 1.0
  batch_size = 512
  max_epochs = 100
  patience = 25
```

---

## 7. Training Techniques

### Early Stopping
- Monitor validation loss (not accuracy)
- Patience: 20-30 epochs depending on model
- Restore best weights after training

### Learning Rate Scheduling
| Model | Scheduler |
|-------|-----------|
| FFNN | ReduceLROnPlateau (reactive) |
| TabNet | StepLR (fixed schedule) |
| Transformers | CosineAnnealingLR (smooth decay) + linear warmup |

### Gradient Clipping (Transformers only)
- `max_grad_norm = 1.0`
- Prevents training instability from attention layers

### Warmup (Transformers only)
- 5 epochs of linear LR warmup (0.01 → 1.0 of target LR)
- Stabilizes early training before full learning rate

---

## 8. Results Summary

### Model Comparison

| Model | Accuracy | Balanced Acc | F1 Macro | Notes |
|-------|----------|--------------|----------|-------|
| **Random Forest** | **0.9652** | **0.9550** | **0.9568** | CV, best overall |
| TabNet | 0.9284 | 0.9016 | 0.9000 | Test set |
| FFNN | 0.9086 | 0.8704 | 0.8629 | Test set |
| TabTransformer | 0.8602 | 0.8036 | 0.7912 | Test set |
| SVM | 0.6404 | 0.5974 | 0.6008 | CV |
| KNN | 0.4184 | 0.3118 | 0.3212 | CV |

### Per-Class Performance (TabNet — Best DL Model)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Grade A | 0.94 | 0.98 | 0.96 | 2,654 |
| Grade B | 0.97 | 0.92 | 0.94 | 3,774 |
| Grade C | 0.94 | 0.96 | 0.95 | 3,708 |
| Grade D | 0.92 | 0.93 | 0.93 | 2,126 |
| Grade E | 0.90 | 0.88 | 0.89 | 1,208 |
| Grade F | 0.82 | 0.76 | 0.79 | 753 |
| Grade G | 0.81 | 0.88 | 0.85 | 608 |

---

## 9. Key Findings & Insights

### Why Random Forest Won
1. **Tree-based models excel on tabular data** with axis-aligned decision boundaries
2. **Feature engineering captured the signal well** — 126 features after engineering
3. **Handles mixed feature types naturally** (numeric + categorical)
4. **Robust to outliers** and doesn't require feature scaling

### Why Transformers Underperformed
1. **TabTransformer limitation:** Only 13/126 features are categorical; attention operates on just 11% of data
2. **FTTransformer helped but not enough:** Full attention on all features, but still overfits
3. **Transformers need more data** — 118K samples is modest for transformer architectures
4. **Tabular data lacks the structure** that transformers exploit in sequences/images

### Why KNN Failed
1. **Curse of dimensionality:** 126 features is too high-dimensional
2. **Even with PCA (40 components):** Distance metrics become unreliable
3. **No feature weighting:** All dimensions weighted equally

### Class Imbalance Observations
- All models struggle most with Grade F (lowest F1 scores)
- Grade G has better recall than F, possibly due to more extreme characteristics
- Class weights help but don't fully solve the imbalance

---

## 10. Architecture Decisions Tested & Rejected

### Embedding-based FFNN
- **Tested:** Separate embedding layers for categoricals + concatenate with numerics + MLP
- **Result:** 4-5% worse than simple FFNN
- **Reason:** Only 13/126 features are categorical; embedding overhead not worth it

### Higher TabTransformer Depth
- **Tested:** depth=6 (original paper recommendation)
- **Result:** Severe overfitting, training instability
- **Reason:** 118K samples insufficient for deep transformer; depth=3 optimal

### ReduceLROnPlateau for Transformers
- **Tested:** Reactive LR reduction on validation plateau
- **Result:** LR stayed too high too long, then crashed
- **Reason:** CosineAnnealingLR with warmup provides smoother, more stable training

---

## 11. File Structure

```
project/
├── data/
│   └── train.csv
├── src/
│   ├── config.py          # Paths, seed, device configuration
│   ├── data_processing.py # Feature engineering, preprocessors
│   ├── models.py          # DL models, training loop, evaluation
│   ├── ml_pipeline.py     # RF, KNN, SVM grid search wrappers
│   └── train.ipynb        # Main notebook
├── models/                # Saved model weights (.pkl)
└── images/                # Training curves, confusion matrices
```

---

## 12. Reproducibility

### Random Seed
- Global seed: 42
- Applied to: NumPy, PyTorch, scikit-learn, train_test_split

### Environment
- Python 3.10+
- PyTorch 2.0+
- scikit-learn 1.3+
- pytorch-tabnet
- tab-transformer-pytorch

### Hardware
- Developed on: MacBook Pro M1 (MPS backend)
- Also tested on: Google Colab (CUDA)

---

## 13. Future Improvements

1. **Ensemble methods:** Combine RF + TabNet predictions
2. **Feature selection:** Use TabNet's feature importances to prune low-importance features
3. **Hyperparameter optimization:** Bayesian optimization instead of grid search
4. **More data:** Transformers would benefit from larger dataset
5. **Cost-sensitive learning:** Different misclassification costs for different grade pairs
6. **Gradient boosting:** XGBoost/LightGBM likely competitive with RF

---

## 14. Conclusion

Random Forest with comprehensive feature engineering achieved the best results (96.5% accuracy) on this loan classification task. Deep learning approaches, while powerful, were limited by:
- Dataset size (118K samples)
- Feature distribution (mostly numerical, few categoricals)
- Class imbalance (despite weighting)

The extensive feature engineering — particularly date cyclical encoding, geographic region mapping, loan title NLP processing, and the informative missingness handling for `months_since_*` columns — was crucial for achieving high performance across all models.

For production deployment, Random Forest is recommended due to its superior accuracy, interpretability (feature importances), and fast inference time.
