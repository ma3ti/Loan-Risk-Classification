"""
data_processing.py
-------------------
All feature engineering and preprocessing logic.

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


# helpers used inside feature_engineer classes
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


def prepare_dl_data(path: str = TRAIN_CSV, val_size: float = 0.10, test_size: float = 0.10, seed: int = SEED) -> dict:
    df = load_data(path)
    X_full = df.drop(columns=["grade"])
    y_full = df["grade"]

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_full)

    # separate test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_full, 
        y_encoded,
        test_size=test_size,
        stratify=y_encoded,
        random_state=seed,
    )

    # separate val
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, 
        y_temp,
        test_size=relative_val,
        stratify=y_temp,
        random_state=seed,
    )

    print(f"Train: {len(X_train)} ({len(X_train)/len(X_full)*100:.1f}%)")
    print(f"Val  : {len(X_val)} ({len(X_val)/len(X_full)*100:.1f}%)")
    print(f"Test : {len(X_test)} ({len(X_test)/len(X_full)*100:.1f}%)")

    numerical_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_full.select_dtypes(include=["object", "category"]).columns.tolist()

    n_num = len(numerical_cols)
    n_cat = len(categorical_cols)
    feature_names = numerical_cols + categorical_cols

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_encoder": le,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "feature_names": feature_names,
        "num_num_cols": n_num,
        "num_cat_cols": n_cat,
        "num_classes": len(le.classes_),
    }   




class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
        Custom Feature Engineer and Preprocessor for traditional ML models (LR, RF, SVM, KNN).
        Combines feature engineering and scaling/encoding into one transformer.
    """
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
    # Load Data
    df = pd.read_csv(path)
    
    # Basic hygiene (Drop duplicates is safe)
    df.drop_duplicates(inplace=True)
    
    # Split X and y
    X = df.drop(columns=["grade"])
    y = df["grade"]
    
    # Encode Target (Safe to do globally for LabelEncoding)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    print(f"Data Loaded: {X.shape}. Returned raw X, y and unfitted preprocessors.")
    
    # Return RAW data + The FeatureEngineer Instance + The Preprocessor Instance
    return X, y, FeatureEngineer(), build_dynamic_preprocessor(), le


# Helper function in dl models processing data
def _safe_log1p(X):
    """
    Applies log1p safely by clipping negative values to 0 first.
    Required because 'months_since' columns have -1 for missing values.
    """
    return np.log1p(np.clip(X, 0, None))



class FFNNPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for Deep Learning models (FFNNs).
    Combines feature engineering and scaling/encoding into one transformer.
    """
    def __init__(self):
        # Lists & State
        self.cols_to_drop_ = []
        self.expected_columns_ = []
        self.medians_ = {}
        self.modes_ = {} # Stores modes for categorical/term cols
        
        # Learned Frequency Maps
        self.state_counts_ = None
        self.zip_counts_ = None
        
        # Sub-Processors
        self.loan_title_proc_ = LoanTitleProcessor() 
        
        # Inner Transformers
        self.dl_preprocessor_ = None
        self.num_cols_count_ = 0
        self.cat_cols_count_ = 0

    def fit(self, X, y=None):
        X = X.copy()
        self.expected_columns_ = X.columns.tolist()
        
        # Missing > 80%
        miss = X.isna().mean()
        self.cols_to_drop_ = miss[miss > 0.80].index.tolist()
        
        # Collinear > 0.95 (Numeric only)
        num_df = X.select_dtypes(include=[np.number])
        if not num_df.empty:
            corr = num_df.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            self.cols_to_drop_ += [c for c in upper.columns if any(upper[c] > 0.95)]
            
        X_temp = X.drop(columns=self.cols_to_drop_, errors='ignore')

        # Hidden Numeric: Loan Term (Learn Mode)
        if "loan_contract_term_months" in X_temp.columns:
            # Clean temporarily to find mode
            clean_term = pd.to_numeric(
                X_temp["loan_contract_term_months"].str.replace(" months", ""), 
                errors='coerce'
            )
            self.modes_["loan_contract_term_months"] = clean_term.mode()[0]

        # Hidden Numeric: Emp Length (Learn Median)
        if "borrower_profile_employment_length" in X_temp.columns:
            series = X_temp["borrower_profile_employment_length"].replace(
                {"< 1 year": "0 years", "10+ years": "10 years"}
            ).str.extract(r"(\d+)").astype(float)
            self.medians_["borrower_profile_employment_length"] = series.median()[0]

        # Standard Numerics (Learn Median)
        numeric_cols = X_temp.select_dtypes(include=[np.number]).columns
        for c in numeric_cols:
            self.medians_[c] = X_temp[c].median()

        # Frequencies (State & Zip)
        if "borrower_address_state" in X_temp.columns:
            self.state_counts_ = X_temp["borrower_address_state"].value_counts()
            
        if "borrower_address_zip" in X_temp.columns:
            zip3 = X_temp["borrower_address_zip"].astype(str).str[:3]
            self.zip_counts_ = zip3.value_counts()
            
        # Loan Title NLP
        if "loan_title" in X_temp.columns:
            self.loan_title_proc_.fit(X_temp)


        # We transform X first so the ColumnTransformer sees the final engineered columns
        X_eng = self._feature_engineer_transform(X)
        
        dl_numeric_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()
        dl_cat_cols = X_eng.select_dtypes(include=["object", "category"]).columns.tolist()
        
        self.num_cols_count_ = len(dl_numeric_cols)
        self.cat_cols_count_ = len(dl_cat_cols)
        
        # Pipeline: Impute -> Log -> Scale
        dl_numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(func=_safe_log1p, validate=False)), 
            ("scaler", StandardScaler()),
        ])

        # Pipeline: Impute -> Ordinal Encode
        dl_categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ordinal", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )),
        ])

        self.dl_preprocessor_ = ColumnTransformer([
            ("num", dl_numeric_transformer, dl_numeric_cols),
            ("cat", dl_categorical_transformer, dl_cat_cols),
        ])
        
        self.dl_preprocessor_.fit(X_eng)
        return self

    def transform(self, X):
        # Re-run all engineering steps
        X_eng = self._feature_engineer_transform(X)
        
        # Scale/Encode to Matrix
        X_trans = self.dl_preprocessor_.transform(X_eng)
        
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
            
        # Shift Categoricals (+1) for Embeddings
        # (Converts -1 to 0, 0 to 1, etc.) to reserve 0 for unknown 
        if self.cat_cols_count_ > 0:
            X_trans[:, self.num_cols_count_:] += 1
            
        return X_trans.astype(np.float32)

    def _feature_engineer_transform(self, X):
        X = X.copy()
        
        # Drop Columns
        X.drop(columns=self.cols_to_drop_, errors='ignore', inplace=True)
        if "next_payment_date" in X.columns:
            X.drop(columns=["next_payment_date"], inplace=True)
        if "loan_purpose_category" in X.columns:
            X.drop(columns=["loan_purpose_category"], inplace=True)

        # Months Since cols kept + add "_ever" binary feature 
        months_cols = [
            "months_since_last_delinquency", 
            "months_since_recent_revolving_delinquency",
            "months_since_last_major_derog",
            "months_since_recent_bankcard_delinquency"
        ]
        for col in months_cols:
            if col in X.columns:
                X[f"{col}_ever"] = X[col].notna().astype(float)
                X[col] = X[col].fillna(-1) 

        # Loan Contract Term (Clean & Impute)
        if "loan_contract_term_months" in X.columns:
            X["loan_contract_term_months"] = pd.to_numeric(
                X["loan_contract_term_months"].str.replace(" months", ""),
                errors="coerce"
            )
            if "loan_contract_term_months" in self.modes_:
                X["loan_contract_term_months"] = X["loan_contract_term_months"].fillna(
                    self.modes_["loan_contract_term_months"]
                )

        # Employment Length (Clean & Impute)
        if "borrower_profile_employment_length" in X.columns:
            X["borrower_profile_employment_length"] = X["borrower_profile_employment_length"].replace(
                {"< 1 year": "0 years", "10+ years": "10 years"}
            ).str.extract(r"(\d+)").astype(float)
            if "borrower_profile_employment_length" in self.medians_:
                 X["borrower_profile_employment_length"] = X["borrower_profile_employment_length"].fillna(
                     self.medians_["borrower_profile_employment_length"]
                 )

        # Dates & Cyclical Features
        date_cols = ["loan_issue_date", "credit_history_earliest_line", "last_payment_date", "last_credit_pull_date"]
        ref_date = pd.Timestamp("2020-01-01")
        
        # Credit History Length
        if {"loan_issue_date", "credit_history_earliest_line"}.issubset(X.columns):
            d1 = pd.to_datetime(X["loan_issue_date"], format="%b-%Y", errors='coerce')
            d2 = pd.to_datetime(X["credit_history_earliest_line"], format="%b-%Y", errors='coerce')
            # Created new feature
            X["credit_history_length_months"] = (d1 - d2).dt.days / 30

        # Create Cyclical Features & Drop Originals
        for col in date_cols:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col], format="%b-%Y", errors='coerce')
                X[f"{col}_year"] = X[col].dt.year
                month = X[col].dt.month
                X[f"{col}_month_sin"] = np.sin(2 * np.pi * month / 12)
                X[f"{col}_month_cos"] = np.cos(2 * np.pi * month / 12)
                X[f"{col}_quarter"] = X[col].dt.quarter
                X[f"{col}_days_since_ref"] = (X[col] - ref_date).dt.days
                X.drop(columns=[col], inplace=True)

        # Loan Title NLP
        if "loan_title" in X.columns:
            X = self.loan_title_proc_.transform(X)

        # State Features (Region, Tax, Freq)
        if "borrower_address_state" in X.columns:
            X["borrower_address_state"] = X["borrower_address_state"].fillna("unknown")
            X["state_region"] = X["borrower_address_state"].apply(_get_state_region)
            X["state_no_income_tax"] = X["borrower_address_state"].str.lower().isin(_NO_INCOME_TAX_STATES).astype(float)
            if self.state_counts_ is not None:
                X["state_log_frequency"] = np.log1p(X["borrower_address_state"].map(self.state_counts_))

        # Zip Features (Region, Freq)
        if "borrower_address_zip" in X.columns:
            X["borrower_address_zip"] = X["borrower_address_zip"].fillna("unknown")
            zip3 = X["borrower_address_zip"].astype(str).str[:3]
            X["zip3_prefix"] = zip3 
            X["zip_region"] = X["borrower_address_zip"].apply(_get_zip_region)
            if self.zip_counts_ is not None:
                X["zip_log_frequency"] = np.log1p(zip3.map(self.zip_counts_))
            X.drop(columns=["borrower_address_zip"], inplace=True)

        # Loan Status Mapping
        if "loan_status_current_code" in X.columns:
             # Map to risk ordinal order to integers
             X["loan_status_current_code"] = X["loan_status_current_code"].replace(_STATUS_MAP).map(_RISK_ORDER)
             if "loan_status_current_code" in self.medians_:
                 X["loan_status_current_code"] = X["loan_status_current_code"].fillna(self.medians_["loan_status_current_code"])

        # Housing & Income Verification
        if "borrower_housing_ownership_status" in X.columns:
            X["borrower_housing_ownership_status"] = X["borrower_housing_ownership_status"].replace(
                {"none": "other", "any": "other"}
            ).fillna("other")
            
        if "borrower_income_verification_status" in X.columns:
             X["borrower_income_verification_status"] = X["borrower_income_verification_status"].fillna("not verified")
             
        # Final Clean-up for Inner Transformer
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if X[col].isna().any():
                X[col] = X[col].fillna("missing")
        
        int_cols = X.select_dtypes(include=["int64"]).columns
        X[int_cols] = X[int_cols].astype(float)

        return X


    def get_metadata(self):
        """
        Returns metadata needed for TabNet and TabTransformer.
        Must be called AFTER fit().
        """
        if self.dl_preprocessor_ is None:
            raise ValueError("Preprocessor is not fitted yet.")

        # Get counts
        n_num = self.num_cols_count_
        n_cat = self.cat_cols_count_
        
        cat_indices = list(range(n_num, n_num + n_cat))
        
        # Get Cardinalities (Vocab Size)
        # access the fitted OrdinalEncoder to see how many categories it learned.
        # add +1 because we shift everything by 1 to reserve index 0 for "Unknown"
        cat_cardinalities = []
        ordinal_encoder = self.dl_preprocessor_.named_transformers_["cat"].named_steps["ordinal"]
        for categories in ordinal_encoder.categories_:
            cat_cardinalities.append(len(categories) + 1)
            
        return {
            "num_num_cols": n_num,
            "num_cat_cols": n_cat,
            "cat_indices": cat_indices,
            "cat_cardinalities": cat_cardinalities,
            # Since I use StandardScaler, mean is 0 and std is 1 for the model input
            "cont_mean_std": np.array([[0.0, 1.0]] * n_num) 
        }
