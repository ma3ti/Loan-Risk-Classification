"""
ml_pipeline.py
--------------
Sklearn-based ML model training: Random Forest, KNN, SVM.

Each function builds a Pipeline, runs GridSearchCV on full data,
and returns the fitted GridSearchCV object.

Saving and printing are done separately from the notebook so you
can re-run them without re-training.

Public API:
    train_random_forest(X, y, preprocessor, ...)  -> GridSearchCV
    train_knn(X, y, preprocessor, global_scaler, ...)  -> GridSearchCV
    train_svm(X, y, preprocessor, global_scaler, ...)  -> GridSearchCV
    save_model(model, name)                        -> path
    print_grid_results(grid_search, model_name)    -> None
    plot_rf_feature_importances(gs, top_k)         -> None
"""

import os
import pickle
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC

from config import MODEL_DIR, SCORING_METRICS, SEED


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES  (called from the notebook, not inside train_ functions)
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model, name: str) -> str:
    """Pickle a model to MODEL_DIR.  Call from the notebook after training."""
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")
    return path


def print_grid_results(gs: GridSearchCV, name: str) -> None:
    """Pretty-print GridSearchCV results for the best estimator."""
    idx = gs.best_index_
    r = gs.cv_results_

    print(f"\n{'='*50}")
    print(f"  {name} — Best GridSearch Results")
    print(f"{'='*50}")
    print(f"  Best Params: {gs.best_params_}")
    print()
    for metric in SCORING_METRICS:
        mean = r[f"mean_test_{metric}"][idx]
        std = r[f"std_test_{metric}"][idx]
        print(f"  {metric:25s}: {mean:.4f} ± {std:.4f}")
    print()


def plot_rf_feature_importances(gs: GridSearchCV, top_k: int = 20, save_path: str = None):
    """
    Plot top-k feature importances from the best Random Forest estimator.

    Parameters
    ----------
    gs    : fitted GridSearchCV whose best_estimator_ is a Pipeline
            containing steps named 'preprocessor' and 'classifier'.
    top_k : how many features to show.
    save_path : if set, save the figure to this path.
    """
    import matplotlib.pyplot as plt

    best_pipe = gs.best_estimator_
    rf = best_pipe.named_steps["classifier"]
    preprocessor = best_pipe.named_steps["preprocessor"]
    importances = rf.feature_importances_

    # Get feature names from the fitted preprocessor
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    # Sort and take top_k
    top_k = min(top_k, len(importances))
    indices = np.argsort(importances)[-top_k:]
    names = [feature_names[i] for i in indices]
    values = importances[indices]

    plt.figure(figsize=(10, max(6, top_k * 0.35)))
    plt.barh(range(top_k), values, color="steelblue")
    plt.yticks(range(top_k), names)
    plt.title(f"Random Forest — Top {top_k} Feature Importances")
    plt.xlabel("Importance (Gini)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════

def train_random_forest(
    X,
    y,
    preprocessor,
    param_grid: Optional[dict] = None,
    cv: int = 3,
    seed: int = SEED,
) -> GridSearchCV:
    """
    Train Random Forest with GridSearchCV.

    Returns the fitted GridSearchCV.  Save / print from the notebook.
    """
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )),
    ])

    if param_grid is None:
        param_grid = {
            "classifier__n_estimators": [300, 500],
            "classifier__max_depth": [None],
            "classifier__min_samples_split": [8, 10],
            "classifier__min_samples_leaf": [1, 2],
            "classifier__max_features": ["sqrt", 0.3],
        }

    gs = GridSearchCV(
        pipeline, param_grid,
        cv=cv,
        scoring=SCORING_METRICS,
        refit="f1_macro",
        n_jobs=-1,
    )

    print("Starting Random Forest GridSearch...")
    gs.fit(X, y)
    print("Random Forest GridSearch complete.")
    return gs


# ══════════════════════════════════════════════════════════════════════════════
# KNN
# ══════════════════════════════════════════════════════════════════════════════

def train_knn(
    X,
    y,
    preprocessor,
    global_scaler,
    param_grid: Optional[dict] = None,
    cv: int = 3,
    seed: int = SEED,
) -> GridSearchCV:
    """Train KNN with GridSearchCV (includes PCA + scaler)."""
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", global_scaler),
        ("pca", PCA(random_state=seed)),
        ("classifier", KNeighborsClassifier(n_jobs=-1)),
    ])

    if param_grid is None:
        param_grid = {
            "pca__n_components": [20, 40, None],
            "scaler__scaler_op": [RobustScaler(), StandardScaler(), MinMaxScaler()],
            "classifier__n_neighbors": [30, 50, 100],
            "classifier__weights": ["distance"],
            "classifier__p": [1, 2],
        }

    gs = GridSearchCV(
        pipeline, param_grid,
        cv=cv,
        scoring=SCORING_METRICS,
        refit="f1_macro",
        n_jobs=-1,
    )

    print("Starting KNN GridSearch...")
    gs.fit(X, y)
    print("KNN GridSearch complete.")
    return gs


# ══════════════════════════════════════════════════════════════════════════════
# SVM
# ══════════════════════════════════════════════════════════════════════════════

def train_svm(
    X,
    y,
    preprocessor,
    global_scaler,
    param_grid: Optional[dict] = None,
    cv: int = 3,
    seed: int = SEED,
) -> GridSearchCV:
    """Train SVM with GridSearchCV."""
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", global_scaler),
        ("classifier", SVC(
            class_weight="balanced",
            random_state=seed,
            max_iter=1000,
            cache_size=6000,
        )),
    ])

    if param_grid is None:
        param_grid = {
            "scaler__scaler_op": [RobustScaler(), StandardScaler(), MinMaxScaler()],
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__kernel": ["linear", "poly", "rbf"],
            "classifier__gamma": ["scale"],
        }

    gs = GridSearchCV(
        pipeline, param_grid,
        cv=cv,
        scoring=SCORING_METRICS,
        refit="f1_macro",
        n_jobs=-1,
    )

    print("Starting SVM GridSearch...")
    gs.fit(X, y)
    print("SVM GridSearch complete.")
    return gs