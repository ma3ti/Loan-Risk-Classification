"""
data_processing.py
-------------------
All feature engineering and preprocessing logic.

Public API used from the notebook:
    load_data(path)                -> DataFrame
    feature_engineer(X, y)         -> X_engineered, column_lists_dict
    build_ml_preprocessor(cols)    -> preprocessor, global_scaler
    build_dl_preprocessor(cols)    -> dl_preprocessor
    prepare_ml_data(path)          -> X, y, preprocessor, global_scaler
    prepare_dl_data(path, ...)     -> dict with splits, encoders, metadata
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from config import SEED, TRAIN_CSV

# load dataset and remove duplicates rows if present
def load_data(path: str = TRAIN_CSV) -> pd.DataFrame:
    """Load CSV, drop exact-duplicate rows, return DataFrame."""
    df = pd.read_csv(path)
    n_dup = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"Loaded {len(df)} rows  (dropped {n_dup} duplicates)")
    return df


# helpers used inside feature_engineer
_REGIONS = {
    "northeast": ["ct", "me", "ma", "nh", "ri", "vt", "nj", "ny", "pa"],
    "midwest": ["il", "in", "mi", "oh", "wi", "ia", "ks", "mn", "mo",
                 "ne", "nd", "sd"],
    "south": ["de", "fl", "ga", "md", "nc", "sc", "va", "wv", "dc",
              "al", "ky", "ms", "tn", "ar", "la", "ok", "tx"],
    "west": ["az", "co", "id", "mt", "nv", "nm", "ut", "wy", "ak",
             "ca", "hi", "or", "wa"],
}

_NO_INCOME_TAX_STATES = {"ak", "fl", "nv", "sd", "tn", "tx", "wa", "wy"}

_STATUS_MAP = {
    "does not meet the credit policy. status:fully paid": "fully paid",
    "does not meet the credit policy. status:charged off": "charged off",
    "default": "charged off",
}

_RISK_ORDER = {
    "fully paid": 0,
    "current": 1,
    "in grace period": 2,
    "late (16-30 days)": 3,
    "late (31-120 days)": 4,
    "charged off": 5,
}

def _get_state_region(state: str) -> str:
    s = str(state).lower()
    for region, states in _REGIONS.items():
        if s in states:
            return region
    return "unknown"

def _get_zip_region(zip_code) -> str:
    try:
        first = int(str(zip_code)[0])
        if first in (0, 1):
            return "northeast"
        elif first in (2, 3):
            return "southeast"
        elif first in (4, 5):
            return "midwest"
        elif first in (6, 7):
            return "south_central"
        elif first == 8:
            return "mountain"
        elif first == 9:
            return "pacific"
    except Exception:
        return "unknown"
    return "unknown"

class LoanTitleProcessor(BaseEstimator, TransformerMixin):
    """Semantic grouping + frequency features for loan_title."""

    CATEGORY_PATTERNS = {
        "debt_consolidation": ["debt consolidation", "consolidation",
                               "consolidate", "debt consol", "consol",
                               "payoff", "pay off"],
        "credit_card": ["credit card", "cc ", "card"],
        "home_improvement": ["home improvement", "home repair",
                             "renovation", "remodel"],
        "major_purchase": ["major purchase", "large purchase"],
        "medical": ["medical", "health", "dental", "hospital"],
        "car": ["car", "auto", "vehicle", "motorcycle", "bike"],
        "business": ["business", "startup", "small business"],
        "moving": ["moving", "relocation", "relocate"],
        "home_buying": ["home buying", "house", "mortgage"],
        "vacation": ["vacation", "travel", "trip"],
        "wedding": ["wedding", "marriage"],
        "education": ["education", "student", "tuition", "school"],
    }

    def __init__(self):
        self.value_counts_ = None

    def fit(self, X, y=None):
        if "loan_title" in X.columns:
            titles = X["loan_title"].fillna("missing").str.lower().str.strip()
            self.value_counts_ = titles.value_counts()
        return self

    def _categorize(self, title):
        if pd.isna(title) or title == "":
            return "missing"
        title = str(title).lower().strip()
        if title == "other":
            return "other"
        for cat, patterns in self.CATEGORY_PATTERNS.items():
            for p in patterns:
                if p in title:
                    return cat
        return "other_uncategorized"

    def transform(self, X):
        X = X.copy()
        if "loan_title" not in X.columns:
            return X
        titles = X["loan_title"].fillna("missing").str.lower().str.strip()
        X["loan_category"] = titles.apply(self._categorize)
        X["loan_title_frequency"] = titles.map(self.value_counts_).fillna(0)
        X["loan_title_log_frequency"] = np.log1p(X["loan_title_frequency"])
        X["loan_title_is_custom"] = (X["loan_title_frequency"] < 10).astype(int)
        X["loan_title_word_count"] = titles.str.split().str.len()
        X["loan_title_has_numbers"] = (
            titles.str.contains(r"\d", regex=True).astype(int)
        )
        X.drop(columns=["loan_title"], inplace=True)
        return X

# Feature engineering function
def feature_engineer(X: pd.DataFrame, y=None):
    """
    Apply ALL feature-engineering steps from the notebook.

    Parameters
    ----------
    X : DataFrame   –  raw features (grade already dropped)
    y : array-like  –  target (unused here, kept for pipeline compat)

    Returns
    -------
    X : DataFrame   –  engineered features
    col_lists : dict with keys
        "normal_dist", "safe_log_cols", "negative_cols",
        "ordinal_cols", "one_hot_cols", "target_encode_cols"
    """
    X = X.copy()

    # Drop cols with > 80% missing
    miss = X.isna().mean()
    drop_high_miss = miss[miss > 0.80].index.tolist()
    X.drop(columns=drop_high_miss, inplace=True, errors="ignore")
    print(f"Dropped {len(drop_high_miss)} cols with >80% missing")

    # Drop next_payment_date feature (not useful for modeling)
    if "next_payment_date" in X.columns:
        X.drop(columns=["next_payment_date"], inplace=True)
        print("Dropped next_payment_date (operational column)")

    # Handle specific "months_since_*" columns
    # These cols have a meaningful missingness: missing = "never happened" (good credit).
    # Create binary flag + impute with -1 to preserve this signal.
    months_since_special = [
        "months_since_last_delinquency",
        "months_since_recent_revolving_delinquency",
        "months_since_last_major_derog",
        "months_since_recent_bankcard_delinquency",
    ]
    processed_months_since = []
    for col in months_since_special:
        if col in X.columns:
            X[f"{col}_ever"] = X[col].notna().astype("float64")
            X[col] = X[col].fillna(-1)
            processed_months_since.append(col)
    if processed_months_since:
        print(f"Processed {len(processed_months_since)} 'months_since_*' cols (added _ever flags, imputed -1)")

    # Impute remaining categorical cols with "missing"
    # This ensures categoricals with moderate missingness get a proper category instead of causing issues downstream.
    cat_cols_early = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols_early:
        if X[col].isna().any():
            X[col] = X[col].fillna("missing")
    if cat_cols_early:
        n_imputed = sum(1 for c in cat_cols_early if "missing" in X[c].values)
        print(f"Imputed 'missing' category in {n_imputed} categorical cols")

    # Drop collinear (r > 0.95) numeric columns
    num_df = X.select_dtypes(include=[np.number])
    corr = num_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    collinear_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    X.drop(columns=collinear_drop, inplace=True, errors="ignore")
    print(f"Dropped {len(collinear_drop)} collinear cols")

    # Convert hidden-numeric columns
    if "loan_contract_term_months" in X.columns:
        X["loan_contract_term_months"] = pd.to_numeric(
            X["loan_contract_term_months"].str.replace(" months", ""),
            errors="coerce",
        )
        mode_val = X["loan_contract_term_months"].mode()[0]
        X["loan_contract_term_months"].fillna(mode_val, inplace=True)

    if "borrower_profile_employment_length" in X.columns:
        X["borrower_profile_employment_length"] = (
            X["borrower_profile_employment_length"]
            .replace({"< 1 year": "0 years", "10+ years": "10 years"})
        )
        X["borrower_profile_employment_length"] = (
            X["borrower_profile_employment_length"]
            .str.extract(r"(\d+)").astype(float)
        )
        med = X["borrower_profile_employment_length"].median()
        X["borrower_profile_employment_length"].fillna(med, inplace=True)

    # Cyclical date encoding + derived features
    date_cols = [
        "loan_issue_date",
        "credit_history_earliest_line",
        "last_payment_date",
        "last_credit_pull_date",
    ]
    for col in date_cols:
        if col in X.columns:
            X[col] = pd.to_datetime(X[col], format="%b-%Y", errors="coerce")

    if {"loan_issue_date", "credit_history_earliest_line"}.issubset(X.columns):
        X["credit_history_length_months"] = (
            (X["loan_issue_date"] - X["credit_history_earliest_line"]).dt.days / 30
        )

    ref_date = pd.Timestamp("2020-01-01")
    for col in date_cols:
        if col in X.columns:
            X[f"{col}_year"] = X[col].dt.year
            month = X[col].dt.month
            X[f"{col}_month_sin"] = np.sin(2 * np.pi * month / 12)
            X[f"{col}_month_cos"] = np.cos(2 * np.pi * month / 12)
            X[f"{col}_quarter"] = X[col].dt.quarter
            X[f"{col}_days_since_ref"] = (X[col] - ref_date).dt.days
            X.drop(columns=[col], inplace=True)

    # Categorical analysis
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    ordinal_cols = []
    one_hot_cols = []
    target_encode_cols = []

    for col in cat_cols:
        n = X[col].nunique()
        if n == 2:
            ordinal_cols.append(col)
        elif n <= 51:
            one_hot_cols.append(col)

    # borrower_housing_ownership_status
    if "borrower_housing_ownership_status" in X.columns:
        X["borrower_housing_ownership_status"] = (
            X["borrower_housing_ownership_status"]
            .replace({"none": "other", "any": "other"})
            .fillna("other")
        )

    # borrower_income_verification_status
    if "borrower_income_verification_status" in X.columns:
        X["borrower_income_verification_status"].fillna(
            "not verified", inplace=True
        )

    # loan_status_current_code  (ordinal mapping)
    if "loan_status_current_code" in X.columns:
        X["loan_status_current_code"] = (
            X["loan_status_current_code"].replace(_STATUS_MAP)
        )
        X["loan_status_current_code"] = (
            X["loan_status_current_code"].map(_RISK_ORDER)
        )
        med_status = X["loan_status_current_code"].median()
        X["loan_status_current_code"].fillna(med_status, inplace=True)
        if "loan_status_current_code" in one_hot_cols:
            one_hot_cols.remove("loan_status_current_code")

    # borrower_address_state -> region features
    if "borrower_address_state" in X.columns:
        X["borrower_address_state"].fillna("unknown", inplace=True)
        X["state_region"] = X["borrower_address_state"].apply(_get_state_region)
        X["state_no_income_tax"] = (
            X["borrower_address_state"]
            .str.lower()
            .isin(_NO_INCOME_TAX_STATES)
            .astype("float64")
        )
        state_counts = X["borrower_address_state"].value_counts()
        X["state_log_frequency"] = np.log1p(
            X["borrower_address_state"].map(state_counts)
        )
        target_encode_cols.append("borrower_address_state")
        if "borrower_address_state" in one_hot_cols:
            one_hot_cols.remove("borrower_address_state")
        one_hot_cols.append("state_region")

    # borrower_address_zip -> region + frequency
    if "borrower_address_zip" in X.columns:
        X["borrower_address_zip"].fillna("unknown", inplace=True)
        X["zip3_prefix"] = X["borrower_address_zip"].str[:3]
        X["zip_region"] = X["borrower_address_zip"].apply(_get_zip_region)
        zip_counts = X["zip3_prefix"].value_counts()
        X["zip_log_frequency"] = np.log1p(X["zip3_prefix"].map(zip_counts))
        X.drop(columns=["borrower_address_zip"], inplace=True)
        one_hot_cols.append("zip_region")
        target_encode_cols.append("zip3_prefix")

    # loan_title processing
    if "loan_title" in X.columns:
        proc = LoanTitleProcessor()
        proc.fit(X)
        X = proc.transform(X)
        # loan_category goes to one-hot
        one_hot_cols.append("loan_category")

    # Drop loan_purpose_category
    if "loan_purpose_category" in X.columns:
        X.drop(columns=["loan_purpose_category"], inplace=True)
        if "loan_purpose_category" in one_hot_cols:
            one_hot_cols.remove("loan_purpose_category")

    # Convert remaining Int64 -> float64
    int_cols = X.select_dtypes(include=["int64"]).columns.tolist()
    X[int_cols] = X[int_cols].astype("float64")

    # Skewness analysis on numeric cols
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    skewness = X[numeric_cols].apply(lambda c: c.skew())
    highly_skewed = skewness[skewness.abs() > 1].index.tolist()
    normal_dist = skewness[skewness.abs() <= 1].index.tolist()
    negative_cols = [c for c in highly_skewed if (X[c] < 0).any()]
    safe_log_cols = [c for c in highly_skewed if c not in negative_cols]

    # Clean up column lists (ensure they still exist)
    existing = set(X.columns)
    ordinal_cols = [c for c in ordinal_cols if c in existing]
    one_hot_cols = [c for c in one_hot_cols if c in existing]
    target_encode_cols = [c for c in target_encode_cols if c in existing]
    normal_dist = [c for c in normal_dist if c in existing]
    safe_log_cols = [c for c in safe_log_cols if c in existing]
    negative_cols = [c for c in negative_cols if c in existing]

    col_lists = {
        "normal_dist": normal_dist,
        "safe_log_cols": safe_log_cols,
        "negative_cols": negative_cols,
        "ordinal_cols": ordinal_cols,
        "one_hot_cols": one_hot_cols,
        "target_encode_cols": target_encode_cols,
    }

    print(f"Feature engineering done  ->  {X.shape[1]} features")
    print(f"  normal_dist      : {len(normal_dist)}")
    print(f"  safe_log_cols    : {len(safe_log_cols)}")
    print(f"  negative_cols    : {len(negative_cols)}")
    print(f"  ordinal_cols     : {len(ordinal_cols)}")
    print(f"  one_hot_cols     : {len(one_hot_cols)}")
    print(f"  target_encode_cols: {len(target_encode_cols)}")

    return X, col_lists


# ML preprocessor for RF, KNN and SVM models
# def build_ml_preprocessor(col_lists: dict, seed: int = SEED):
#     """
#     Build the ColumnTransformer + global scaler used by pipelines.

#     Returns
#     -------
#     preprocessor   : ColumnTransformer
#     global_scaler  : ColumnTransformer  (RobustScaler on all non-OHE cols)
#     """
#     normal_transformer = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#     ])

#     log_transformer = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("log", FunctionTransformer(
#             np.log1p, validate=False, feature_names_out="one-to-one"
#         )),
#     ])

#     power_transformer_pipe = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
#     ])

#     target_transformer = Pipeline([
#         ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
#         ("target_enc", TargetEncoder(random_state=seed)),
#     ])

#     ordinal_transformer = Pipeline([
#         ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
#         ("encoder", OrdinalEncoder(
#             handle_unknown="use_encoded_value", unknown_value=-1
#         )),
#     ])

#     one_hot_transformer = Pipeline([
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
#     ])

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("normal", normal_transformer, col_lists["normal_dist"]),
#             ("skewed", log_transformer, col_lists["safe_log_cols"]),
#             ("power", power_transformer_pipe, col_lists["negative_cols"]),
#             ("target_enc", target_transformer, col_lists["target_encode_cols"]),
#             ("ordinal", ordinal_transformer, col_lists["ordinal_cols"]),
#             ("one_hot", one_hot_transformer, col_lists["one_hot_cols"]),
#         ],
#         n_jobs=-1,
#         verbose_feature_names_out=False,
#     )

#     n_scalable = (
#         len(col_lists["normal_dist"])
#         + len(col_lists["safe_log_cols"])
#         + len(col_lists["negative_cols"])
#         + len(col_lists["ordinal_cols"])
#         + len(col_lists["target_encode_cols"])
#     )
#     global_scaler = ColumnTransformer(
#         transformers=[
#             ("scaler_op", RobustScaler(), slice(0, n_scalable)),
#         ],
#         remainder="passthrough",
#     )

#     return preprocessor#, global_scaler

# DL preprocessor: ordinal cats, StandardScaler, no one-hot
def build_dl_preprocessor(X_reference: pd.DataFrame):
    """
    Build a DL-specific ColumnTransformer.

    * Numeric cols  -> median impute -> log1p(clip) -> StandardScaler
    * Categorical cols -> "missing" impute -> OrdinalEncoder  (integer output)

    Parameters
    ----------
    X_reference : DataFrame (e.g. X_train) used only to detect column dtypes.

    Returns
    -------
    dl_preprocessor : ColumnTransformer
    numerical_cols  : list[str]
    categorical_cols: list[str]
    """
    numerical_cols = X_reference.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_reference.select_dtypes(include=["object", "category"]).columns.tolist()
    if "grade" in categorical_cols:
        categorical_cols.remove("grade")

    dl_numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(
            func=lambda X: np.log1p(np.clip(X, 0, None)),
            validate=False,
        )),
        ("scaler", StandardScaler()),
    ])

    dl_categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )),
    ])

    dl_preprocessor = ColumnTransformer([
        ("num", dl_numeric_transformer, numerical_cols),
        ("cat", dl_categorical_transformer, categorical_cols),
    ], remainder="drop")

    return dl_preprocessor, numerical_cols, categorical_cols


#Full ML data preparation pipeline
# def prepare_ml_data_test(path: str = TRAIN_CSV, seed: int = SEED):
#     """
#     Full pipeline for traditional ML:
#       load -> separate X/y -> feature_engineer -> build preprocessor

#     Returns
#     -------
#     X, y, col_lists, preprocessor, global_scaler
#     """
#     df = load_data(path)
#     X = df.drop(columns=["grade"])
#     y = df["grade"]

#     le = LabelEncoder()
#     y = le.fit_transform(y)

#     X, col_lists = feature_engineer(X)
#     preprocessor, global_scaler = build_ml_preprocessor(col_lists, seed)

#     return X, y, col_lists, preprocessor, global_scaler, le


# Full DL data preparation pipeline
def prepare_dl_data(
    path: str = TRAIN_CSV,
    val_size: float = 0.10,
    test_size: float = 0.10,
    seed: int = SEED,
):
    """
    Full pipeline for DL models:
      load -> split -> feature_engineer (fit on train) -> DL preprocess

    Returns
    -------
    dict with keys:
        X_train, X_val, X_test          – numpy arrays (processed)
        y_train, y_val, y_test           – numpy int arrays
        label_encoder                    – fitted LabelEncoder
        dl_preprocessor                  – fitted ColumnTransformer
        numerical_cols, categorical_cols – column name lists
        cat_cardinalities                – list[int]  (for TabNet/TabTransformer)
        num_cat_cols                     – number of categorical columns
        num_num_cols                     – number of numerical columns
        cat_indices                      – list of cat column indices in processed array
        cont_mean_std                    – (n_num, 2) array of mean/std (for TabTransformer)
    """
    df = load_data(path)
    X_full = df.drop(columns=["grade"])
    y_full = df["grade"]

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_full)

    # separate test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_full, y_encoded,
        test_size=test_size,
        stratify=y_encoded,
        random_state=seed,
    )

    # separate val
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val,
        stratify=y_temp,
        random_state=seed,
    )

    print(f"Train: {len(X_train)} ({len(X_train)/len(X_full)*100:.1f}%)")
    print(f"Val  : {len(X_val)} ({len(X_val)/len(X_full)*100:.1f}%)")
    print(f"Test : {len(X_test)} ({len(X_test)/len(X_full)*100:.1f}%)")

    # Feature engineering fit statistics on TRAIN only
    # We need to apply the same transforms to val/test without re-fitting
    X_train, col_lists = feature_engineer(X_train)
    X_val = apply_feature_engineer_transform(X_val, X_train)
    X_test = apply_feature_engineer_transform(X_test, X_train)

    # Build DL preprocessor from train columns
    dl_preprocessor, numerical_cols, categorical_cols = build_dl_preprocessor(X_train)

    # Fit on train, transform all splits
    X_train_proc = dl_preprocessor.fit_transform(X_train)
    X_val_proc = dl_preprocessor.transform(X_val)
    X_test_proc = dl_preprocessor.transform(X_test)

    # Handle sparse output
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_val_proc = X_val_proc.toarray()
        X_test_proc = X_test_proc.toarray()

    # ★ CRITICAL FIX: Shift categorical values by +1 ★
    # OrdinalEncoder outputs: known categories as 0,1,2,..., unknown as -1.
    # By shifting +1, we get: unknown → 0, categories → 1,2,3,...
    # This prevents collision between "unknown" and "first category" in embeddings.
    n_num = len(numerical_cols)
    n_cat = len(categorical_cols)
    if n_cat > 0:
        print(f"Shifting {n_cat} categorical columns by +1 (unknown=-1 → 0, categories → 1+)")
        X_train_proc[:, n_num:] += 1
        X_val_proc[:, n_num:] += 1
        X_test_proc[:, n_num:] += 1

    # Compute metadata for TabNet / TabTransformer
    cat_indices = list(range(n_num, n_num + n_cat))

    # Cardinalities: max value + 1 (since we already shifted, just need +1 for 0-index)
    cat_cardinalities = []
    for i in range(n_cat):
        col_idx = n_num + i
        max_val = max(
            X_train_proc[:, col_idx].max(),
            X_val_proc[:, col_idx].max(),
            X_test_proc[:, col_idx].max(),
        )
        cat_cardinalities.append(int(max_val) + 1)

    # Continuous mean/std for TabTransformer (computed from train only)
    cont_mean = X_train_proc[:, :n_num].mean(axis=0)
    cont_std = X_train_proc[:, :n_num].std(axis=0)
    cont_std[cont_std == 0] = 1.0  # avoid division by zero
    cont_mean_std = np.stack([cont_mean, cont_std], axis=1)  # shape (n_num, 2)

    # Ordered feature names: [numerical_cols..., categorical_cols...]
    # This matches the column order in the processed arrays.
    feature_names = numerical_cols + categorical_cols

    return {
        "X_train": X_train_proc.astype(np.float32),
        "X_val": X_val_proc.astype(np.float32),
        "X_test": X_test_proc.astype(np.float32),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_encoder": le,
        "dl_preprocessor": dl_preprocessor,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "feature_names": feature_names,
        "num_num_cols": n_num,
        "num_cat_cols": n_cat,
        "cat_indices": cat_indices,
        "cat_cardinalities": cat_cardinalities,
        "cont_mean_std": cont_mean_std,
        "col_lists": col_lists,
        "num_classes": len(le.classes_),
    }


def apply_feature_engineer_transform(X_new: pd.DataFrame, X_train_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering to val/test that was applied to train.

    This re-runs the deterministic transforms (date encoding, status mapping, etc.)
    and aligns columns to match X_train_ref. Frequency-based features are
    re-computed per-split (acceptable for log-frequency features since they're
    just used as rough signals, and this avoids complex fit/transform state).
    """
    X, _ = feature_engineer(X_new)

    # Align columns: add missing cols as 0, drop extra cols
    missing = set(X_train_ref.columns) - set(X.columns)
    for col in missing:
        X[col] = 0
    X = X[X_train_ref.columns]
    return X


# Convenience: LabelEncoder fit_transform
def encode_target(y):
    """Convenience: LabelEncoder fit_transform."""
    le = LabelEncoder()
    return le.fit_transform(y), le







# --------------------- TEST

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, seed=42):
        self.seed = seed
        self.medians_ = {}
        self.modes_ = {}
        self.freq_counts_ = {}
        self.cols_to_drop_ = []
        self.expected_columns_ = [] # expected cols (used to check dataframe structure)
        
        # Internal Logic Lists
        self.skewed_log_cols_ = []
        self.skewed_power_cols_ = []
        self.target_enc_cols_ = []
        
        # Transformers
        self.power_transformer_ = PowerTransformer(method='yeo-johnson', standardize=False)
        self.target_encoder_ = TargetEncoder(random_state=seed)
        self.title_processor = LoanTitleProcessor()

    def fit(self, X, y=None):
        X = X.copy()

        self.expected_columns_ = X.columns.tolist()
        
        # Drop > 80% Missing
        miss = X.isna().mean()
        high_miss = miss[miss > 0.80].index.tolist()
        self.cols_to_drop_.extend(high_miss)
        
        # Drop Collinear Columns (> 0.95)
        # We temporarily drop high_miss cols to calculate correlation safely
        X_temp = X.drop(columns=high_miss, errors='ignore')
        num_df = X_temp.select_dtypes(include=[np.number])
        if not num_df.empty:
            corr = num_df.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            collinear_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
            self.cols_to_drop_.extend(collinear_drop)

        # Fit Title Processor
        self.title_processor.fit(X)

        # Learn Statistics (Medians, Modes, Skew)
        # Remove columns we plan to drop so we don't waste time processing them
        X_temp = X.drop(columns=self.cols_to_drop_, errors='ignore')

        # Clean "Hidden Numerics" JUST to calculate Median correctly
        # (We repeat this logic in transform, but we need it here to get the right median)
        if "loan_contract_term_months" in X_temp.columns:
            X_temp["loan_contract_term_months"] = pd.to_numeric(X_temp["loan_contract_term_months"].str.replace(" months", ""), errors="coerce")
        if "borrower_profile_employment_length" in X_temp.columns:
            X_temp["borrower_profile_employment_length"] = X_temp["borrower_profile_employment_length"].replace({"< 1 year": "0 years", "10+ years": "10 years"}).str.extract(r"(\d+)").astype(float)

        # Numeric Stats
        for col in X_temp.select_dtypes(include=[np.number]).columns:
            self.medians_[col] = X_temp[col].median()
            
            skew = X_temp[col].skew()
            if abs(skew) > 1:
                if (X_temp[col] < 0).any():
                    self.skewed_power_cols_.append(col)
                else:
                    self.skewed_log_cols_.append(col)

        # Categorical Stats
        for col in X_temp.select_dtypes(include=["object", "category"]).columns:
            self.modes_[col] = X_temp[col].mode()[0] if not X_temp[col].mode().empty else "missing"
            
            # Frequencies
            if col in ["borrower_address_zip", "borrower_address_state"]:
                self.freq_counts_[col] = X_temp[col].value_counts()

            # Target Encoding Rule (> 51 categories)
            if X_temp[col].nunique() > 51:
                self.target_enc_cols_.append(col)

        # Fit Internal Transformers
        # Power Transformer (Impute median first)
        if self.skewed_power_cols_:
            X_pow = X_temp[self.skewed_power_cols_].fillna(X_temp[self.skewed_power_cols_].median())
            self.power_transformer_.fit(X_pow)
            
        # Target Encoder (Impute missing first)
        if self.target_enc_cols_ and y is not None:
            X_tar = X_temp[self.target_enc_cols_].fillna("missing")
            self.target_encoder_.fit(X_tar, y)

        return self


    def transform(self, X):
        
        missing_cols = set(self.expected_columns_) - set(X.columns) - set(self.cols_to_drop_)
        if missing_cols:
            raise ValueError(f"The test dataset is missing required columns: {missing_cols}")

        X = X.copy()
        
        # Drop bad columns
        X.drop(columns=self.cols_to_drop_, errors="ignore", inplace=True)
        if "next_payment_date" in X.columns:
            X.drop(columns=["next_payment_date"], inplace=True)

        # Specific 'Months Since' Handling
        special_cols = ["months_since_last_delinquency", "months_since_recent_revolving_delinquency",
                        "months_since_last_major_derog", "months_since_recent_bankcard_delinquency"]
        for col in special_cols:
            if col in X.columns:
                X[f"{col}_ever"] = X[col].notna().astype("float64")
                X[col] = X[col].fillna(-1)

        # Convert Hidden Numerics (to ensure we can apply medians correctly and have them as numerics for transforms)
        if "loan_contract_term_months" in X.columns:
            X["loan_contract_term_months"] = pd.to_numeric(X["loan_contract_term_months"].str.replace(" months", ""), errors="coerce")
        if "borrower_profile_employment_length" in X.columns:
            X["borrower_profile_employment_length"] = X["borrower_profile_employment_length"].replace({"< 1 year": "0 years", "10+ years": "10 years"}).str.extract(r"(\d+)").astype(float)

        # Impute Numerics (using Learned Medians)
        for col, val in self.medians_.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)

        # Impute Categoricals (using Learned Modes)
        for col, val in self.modes_.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)

        # Collect new columns in a dictionary first to avoid fragmentation warnings from multiple insertions
        new_cols = {}

        # Dates & Regions
        date_cols = ["loan_issue_date", "credit_history_earliest_line", "last_payment_date", "last_credit_pull_date"]
        # Convert to datetime first (in-place is fine here)
        for col in date_cols:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col], format="%b-%Y", errors="coerce")
        
        if {"loan_issue_date", "credit_history_earliest_line"}.issubset(X.columns):
            new_cols["credit_history_length_months"] = ((X["loan_issue_date"] - X["credit_history_earliest_line"]).dt.days / 30)

        ref_date = pd.Timestamp("2020-01-01")
        cols_to_drop_dates = []
        
        for col in date_cols:
            if col in X.columns:
                new_cols[f"{col}_year"] = X[col].dt.year
                month = X[col].dt.month
                new_cols[f"{col}_month_sin"] = np.sin(2 * np.pi * month / 12)
                new_cols[f"{col}_month_cos"] = np.cos(2 * np.pi * month / 12)
                new_cols[f"{col}_quarter"] = X[col].dt.quarter
                new_cols[f"{col}_days_since_ref"] = (X[col] - ref_date).dt.days
                cols_to_drop_dates.append(col)
        
        # Region Mapping
        if "borrower_address_state" in X.columns:
             new_cols["state_region"] = X["borrower_address_state"].apply(_get_state_region)
             new_cols["state_no_income_tax"] = X["borrower_address_state"].str.lower().isin(_NO_INCOME_TAX_STATES).astype("float64")

        if "borrower_address_zip" in X.columns:
            new_cols["zip3_prefix"] = X["borrower_address_zip"].astype(str).str[:3]
            new_cols["zip_region"] = X["borrower_address_zip"].apply(_get_zip_region)

        # Apply Frequency Encoding
        if "borrower_address_state" in X.columns and "borrower_address_state" in self.freq_counts_:
            new_cols["state_log_frequency"] = np.log1p(X["borrower_address_state"].map(self.freq_counts_["borrower_address_state"]).fillna(0))
        
        # --- MERGE ALL NEW COLUMNS AT ONCE ---
        if new_cols:
            # This solves the fragmentation warning
            X = pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)

        # Now safe to drop old columns
        X.drop(columns=cols_to_drop_dates, inplace=True, errors='ignore')
        if "borrower_address_zip" in X.columns:
            X.drop(columns=["borrower_address_zip"], inplace=True)

        # Loan Title (Transformer handles its own internals)
        X = self.title_processor.transform(X)

        # Advanced Transforms (Log, Power, Target)
        # Log
        for col in self.skewed_log_cols_:
            if col in X.columns:
                X[col] = np.log1p(X[col].clip(lower=0))
        
        # Power
        if self.skewed_power_cols_:
            valid = [c for c in self.skewed_power_cols_ if c in X.columns]
            if valid:
                X[valid] = self.power_transformer_.transform(X[valid])
                
        # Target
        if self.target_enc_cols_:
            valid = [c for c in self.target_enc_cols_ if c in X.columns]
            if valid:
                X[valid] = self.target_encoder_.transform(X[valid])

        return X


def build_dynamic_preprocessor():
    """
    Returns the ColumnTransformer that scales/encodes the OUTPUT of FeatureEngineer.
    """
    return ColumnTransformer(
        transformers=[
            # Scale all numerics (RobustScaler handles outliers better)
            ("scaler", RobustScaler(), make_column_selector(dtype_include=np.number)),
            
            # OneHotEncode low-cardinality strings (high-cardinality ones are already TargetEncoded in FeatureEngineer)
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), 
             make_column_selector(dtype_include=[object, "category"])),
        ],
        verbose_feature_names_out=False,
    )

def prepare_ml_data(path: str = TRAIN_CSV):
    """
    Perform a first cleaning of the data and return unfitted feature engineer and preprocessor.
    
    Parameters
    ----------
    path : str
        Path to the CSV data file.
    
    Returns
    -------
    X : DataFrame
        Raw features (grade column dropped and dropped duplicates rows).
    y : array-like
        Encoded target variable.
    feature_engineer_instance : FeatureEngineer
        An unfitted instance of the FeatureEngineer class.
    preprocessor_instance : ColumnTransformer
        An unfitted instance of the dynamic preprocessor.
    label_encoder_instance : LabelEncoder
        An unfitted instance of LabelEncoder for the target variable.    
    """
    # 1. Load Data
    df = pd.read_csv(path)
    
    # 2. Basic hygiene (Drop duplicates is safe)
    df.drop_duplicates(inplace=True)
    
    # 3. Split X and y
    X = df.drop(columns=["grade"])
    y = df["grade"]
    
    # 4. Encode Target (Safe to do globally for LabelEncoding)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    print(f"Data Loaded: {X.shape}. Returned raw X, y and unfitted preprocessors.")
    
    # Return RAW data + The Class Instance + The Preprocessor Instance
    return X, y, FeatureEngineer(), build_dynamic_preprocessor(), le