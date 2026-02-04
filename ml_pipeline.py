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
    save_split_model(fitted_gs, model_name)        -> None
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
from sklearn.linear_model import LogisticRegression


from config import MODEL_DIR, SCORING_METRICS, SEED
#from data_processing import feature_engineer, FeatureEngineer


# UTILITIES FOR SAVING, PRINTING, PLOTTING
def save_model(model, name: str) -> str:
    """Pickle a model to MODEL_DIR.  Call from the notebook after training."""
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")
    return path

def save_split_model(fitted_gs, name: str):
    """
    Splits a fitted Pipeline into two parts and saves them:
    1. Preprocessor (All steps EXCEPT the last one) -> models/{name}_preprocessor.pkl
    2. Classifier (The LAST step) -> models/{name}_classifier.pkl
    
    This works for:
    - RF:  [Engineer, Prep] + [RF]
    - SVM:  [Engineer, Prep] + [SVM]
    - KNN: [Engineer, Prep, PCA] + [KNN]
    """
    os.makedirs("models", exist_ok=True)
    
    # Get the best estimator (The Full Pipeline)
    full_pipeline = fitted_gs.best_estimator_
    
    # Preprocessor: for KNN, this includes Engineer + Prep + PCA
    preprocessor_chain = Pipeline(full_pipeline.steps[:-1])
    path_prep = f"models/{name}_preprocessor.pkl"
    with open(path_prep, "wb") as f:
        pickle.dump(preprocessor_chain, f)
        
    # Classifier
    classifier_model = full_pipeline.steps[-1][1]
    path_clf = f"models/{name}_classifier.pkl"
    with open(path_clf, "wb") as f:
        pickle.dump(classifier_model, f)
        
    print(f"[{name.upper()}] Saved Successfully:")
    print(f"  Preprocessor: {path_prep}")
    print(f"  Classifier:   {path_clf}")


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



# RANDOM FOREST training
def train_random_forest(X, y, feature_engineer, preprocessor, param_grid=None, cv=3, seed=SEED) -> GridSearchCV:
    """ Train Random Forest with GridSearchCV.
    
    Parameters
    ----------
    X : feature DataFrame
    y : target Series
    feature_engineer : FeatureEngineer instance
    preprocessor : preprocessor instance
    param_grid : dict
        GridSearchCV param grid. If None, use default.
    cv : int
        Number of CV folds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object.
    """    

    pipeline = Pipeline([
        ("engineer", feature_engineer),
        ("preprocessor", preprocessor), 
        ("classifier", RandomForestClassifier(
            random_state=seed, 
            class_weight="balanced_subsample",
            n_jobs=-1
        ))
    ])

    if param_grid is None:
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, 20],
        }

    gs = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring=SCORING_METRICS,
        refit="f1_macro", 
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting RF GridSearch...")
    gs.fit(X, y)
    print("RF GridSearch complete.")
    
    return gs


# KNN training
def train_knn(X, y, feature_engineer, preprocessor, param_grid=None, cv=3, seed=SEED) -> GridSearchCV:
    """Train KNN with GridSearchCV 
    
    Parameters
    ----------
    X : feature DataFrame
    y : target Series
    feature_engineer : FeatureEngineer instance
    preprocessor : preprocessor instance
    param_grid : dict
        GridSearchCV param grid. If None, use default.
    cv : int
        Number of CV folds.
    seed : int
        Random seed for reproducibility.    
    
    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object.
    """


    pipeline = Pipeline([
        ("engineer", feature_engineer), 
        ("preprocessor", preprocessor),     
        ("pca", PCA(random_state=seed)),
        ("classifier", KNeighborsClassifier(
            n_jobs=-1
        ))
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
        pipeline, 
        param_grid,
        cv=cv,
        scoring=SCORING_METRICS,
        refit="f1_macro",
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting KNN GridSearch...")
    gs.fit(X, y)
    print("KNN GridSearch complete.")
    
    return gs

# SVM training 
def train_svm(X, y, feature_engineer, preprocessor, param_grid=None, cv=3, seed=SEED) -> GridSearchCV:
    """Train SVM with GridSearchCV.
    
    Parameters
    ----------
    X : feature DataFrame
    y : target Series
    feature_engineer : FeatureEngineer instance
    preprocessor : preprocessor instance
    param_grid : dict
        GridSearchCV param grid. If None, use default.
    cv : int
        Number of CV folds.
    seed : int
        Random seed for reproducibility.    
    
    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object.
    """
    
    pipeline = Pipeline([
        ("engineer", feature_engineer),
        ("preprocessor", preprocessor),        
        ("classifier", SVC(
            class_weight="balanced",
            random_state=seed,
            max_iter=1000,
            cache_size=6000,
        ))
    ])

    if param_grid is None:
        param_grid = {
            "scaler__scaler_op": [RobustScaler(), StandardScaler(), MinMaxScaler()],
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__kernel": ["linear", "poly", "rbf"],
            "classifier__gamma": ["scale"],
        }

    gs = GridSearchCV(
        pipeline, 
        param_grid,
        cv=cv,
        scoring=SCORING_METRICS,
        refit="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    print("Starting SVM GridSearch...")
    gs.fit(X, y)
    print("SVM GridSearch complete.")
    
    return gs



def train_lr(X, y, feature_engineer, preprocessor, param_grid=None, cv=3, seed=SEED) -> GridSearchCV:
    """Train Logistic Regression with GridSearchCV.
    
    Parameters
    ----------
    X : feature DataFrame
    y : target Series
    feature_engineer : FeatureEngineer instance
    preprocessor : preprocessor instance
    param_grid : dict
        GridSearchCV param grid. If None, use default.
    cv : int
        Number of CV folds.
    seed : int
        Random seed for reproducibility.    
    
    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object.
    """
    
    pipeline = Pipeline([
        ("engineer", feature_engineer),
        ("preprocessor", preprocessor),        
        ("classifier", LogisticRegression(
            class_weight="balanced",
            random_state=seed,
            max_iter=2000, 
            n_jobs=-1    
        ))
    ])

    if param_grid is None:
        param_grid = {
            "preprocessor__scaler": [StandardScaler(), RobustScaler(), MinMaxScaler()],
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__solver": ["lbfgs", "saga"],
        }

    gs = GridSearchCV(
        pipeline, 
        param_grid,
        cv=cv,
        scoring=SCORING_METRICS,
        refit="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    print("Starting Logistic Regression GridSearch...")
    gs.fit(X, y)
    print("Logistic Regression GridSearch complete.")
    
    return gs