# Project Overview: Loan Risk Classification

This document summarizes the pipeline architecture, data processing decisions, and model configurations used in this project.

## 1. Data Processing
**Module:** `data_processing.py`

### 1.1. Filtering & Drops
* **High Missingness:** Dropped columns with **> 80% missing values**.
* **Operational Leakage:** Dropped `next_payment_date` (operational field not predictive at origination).
* **Redundant/Processed Drops:**
    * `loan_purpose_category` (redundant with processed loan title).
    * `loan_title` (replaced by engineered features).
    * `borrower_address_zip` (replaced by regional/frequency features).
    * Original Date columns (replaced by cyclical/duration features).
* **Collinearity:** Dropped numeric columns with Pearson correlation **> 0.95**.
* **Duplicates:** Dropped exact duplicate rows.

### 1.2. Feature Engineering

#### A. Temporal Features (Dates)
* **Columns Processed:** `loan_issue_date`, `credit_history_earliest_line`, `last_payment_date`, `last_credit_pull_date`.
* **New Features Created:**
    * **Credit History Length:** `loan_issue_date` - `credit_history_earliest_line` (in months).
    * **Cyclical Encoding:** Sine and Cosine of the month (`_month_sin`, `_month_cos`) to capture seasonality.
    * **Duration:** `days_since_ref` (Reference: 2020-01-01).
    * **Calendar:** Extracted Year and Quarter.

#### B. "Months Since" Handling
* **Logic:** Missing values in columns starting with `months_since_` indicate the event (e.g., last delinquency) **never happened** (good signal).
* **Action:**
    1. Created binary flags `*_ever` (1 = event occurred, 0 = never).
    2. Imputed original column NaNs with **-1** to preserve the "never" signal numerically.

#### C. Text & Categorical Cleaning
* **`loan_contract_term_months`:** Cleaned " months" suffix, converted to numeric, imputed NaNs with **Mode**.
* **`borrower_profile_employment_length`:**
    * Mapped `< 1 year` → 0, `10+ years` → 10.
    * Extracted years via Regex, converted to float, imputed NaNs with **Median**.
* **`borrower_housing_ownership_status`:** Consolidating noise ("none", "any") into "other"; NaNs filled with "other".
* **`borrower_income_verification_status`:** NaNs filled with "not verified".

#### D. Loan Status Ordinal Encoding
* **Feature:** `loan_status_current_code`.
* **Logic:** Mapped descriptive status strings to an **Ordinal Risk Score (0–5)**:
    * 0: Fully Paid
    * 1: Current
    * ...
    * 5: Charged Off / Default
* **Imputation:** Missing values filled with the **Median** risk score.

#### E. Geographic Engineering
* **State (`borrower_address_state`):**
    * **Region Mapping:** Mapped states to macro-regions (Northeast, Midwest, South, West) → `state_region`.
    * **Economic Flag:** Created `state_no_income_tax` binary flag for specific states (TX, FL, WA, etc.).
    * **Frequency:** Added `state_log_frequency` (log count of state occurrences).
* **Zip Code (`borrower_address_zip`):**
    * **Granularity Reduction:** Extracted first 3 digits → `zip3_prefix`.
    * **Region Mapping:** Mapped first digit to US region (e.g., 0-1 = Northeast) → `zip_region`.
    * **Frequency:** Added `zip_log_frequency`.

#### F. Loan Title Processing (NLP)
* **Processor:** Custom `LoanTitleProcessor` class.
* **Categorization:** Mapped noisy user input (5000+ unique titles) to ~12 standard categories (e.g., "consolidation", "credit_card", "medical") → `loan_category`.
* **Metadata Features:**
    * `loan_title_word_count`: Number of words in title.
    * `loan_title_has_numbers`: Binary flag.
    * `loan_title_is_custom`: Flag for rare titles (freq < 10).

### 1.3. Imputation Strategy
* **Categorical:** All remaining categorical NaNs filled with explicit **"missing"** token.
* **Numeric:** Standard median imputation (unless specific logic applied above).
---

## 2. Machine Learning Pipeline
**Module:** `ml_pipeline.py`

### Preprocessing (Sklearn Pipeline)
* **Scaler:** `RobustScaler` (Chosen for resilience to outliers).
* **Categorical Encoding:**
    * **One-Hot Encoding:** Low cardinality columns (≤ 51 unique values).
    * **Target Encoding:** High cardinality columns (e.g., `zip3_prefix`, `state`).
    * **Ordinal Encoding:** Binary columns.
* **Transformations:**
    * `Yeo-Johnson`: For negative/skewed numeric features.
    * `Log1p`: For strictly positive skewed features.

### Models & Strategy
All models trained using **GridSearchCV** with `cv=3` and `refit='f1_macro'`.
1. **Random Forest:**
    * Uses `class_weight='balanced_subsample'`.
    * Tuned: `n_estimators`, `max_depth`, `min_samples_leaf`.
2. **KNN (K-Nearest Neighbors):**
    * **Pipeline:** Includes **PCA** (dimensionality reduction) → KNN.
    * Tuned: `n_components` (PCA), `n_neighbors`, `weights` (distance vs uniform).
3. **SVM (Support Vector Machine):**
    * Uses `class_weight='balanced'`.
    * Tuned: `C` (Regularization), `Kernel` (RBF vs Linear).

---

## 3. Deep Learning Pipeline
**Module:** `models.py`

### Preprocessing (Distinct from ML)
* **Scaling:** `StandardScaler` applied to numerical features.
* **Encoding:** All categoricals converted to **Integers** (Ordinal Encoding) for Embedding layers.
* **Handling Unknowns:** Categorical indices shifted (+1) to reserve Index 0 for "Unknown/Missing".

### Architectures
1. **FFNN (Feed-Forward):**
    * Block structure: `Linear` → `LayerNorm` → `LeakyReLU` → `Dropout`.
2. **TabNet:**
    * Uses **Attentive Transformers** for feature selection.
    * Key Params: `sparsemax` masking, `virtual_batch_size` for Ghost Batch Norm.
3. **TabTransformer:**
    * **Transformer Encoder** applied to Categorical Embeddings.
    * **MLP** applied to Numerical features.
4. **FT-Transformer:**
    * **Feature Tokenizer:** Converts *both* Numerical and Categorical features into embeddings.
    * Applies Self-Attention over the entire feature set.

### Training Configuration
* **Optimizer:** `AdamW` (Weight Decay for regularization).
* **Loss Function:** `CrossEntropyLoss` with **Class Weights** to handle imbalance.
* **Scheduler:** `CosineAnnealingLR` (Smooth decay) or `ReduceLROnPlateau`.
* **Stability:** Gradient Clipping (`max_grad_norm=1.0`) applied to prevent divergence.

---

## 4. Evaluation Standards
**Module:** `config.py`

* **Primary Metric:** `F1-Macro` (Prioritized due to class imbalance).
* **Global Seed:** `42` set for Numpy, Pandas, PyTorch, and CUDA to ensure reproducibility.