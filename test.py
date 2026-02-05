"""
test.py
-------
Inference Module.
Supports: 'lr','rf', 'knn', 'svm', 'ff', 'tb', 'tf'.
"""

import pickle
import pandas as pd
import numpy as np
import os
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Imports necessary for pickle to reconstruct custom classes
# (Ensure these files are in the same folder)
import config
import data_processing 
import models


def getName():
    """Returns the author's name."""
    return "Matteo Fontana"


def preprocess(dataset: pd.DataFrame, clfName: str):
    """
    Loads preprocessor. 
    Returns:
        X_processed: The transformed feature matrix.
        y_true: The true encoded labels (or None if 'grade' was missing).
    """
    # 1. Handle Target (y)
    y_true = None
    if "grade" in dataset.columns:
        # Map letters A-G to integers 0-6 to match model output
        grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
        # Filter only valid rows if necessary, or assume clean data
        y_true = dataset["grade"].map(grade_map).values
        dataset = dataset.drop(columns=["grade"])
    
    # 2. Resolve Preprocessor Path
    path = f"models/{clfName}_preprocessor.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor for method '{clfName}' not found at {path}")

    # 3. Load & Transform (X)
    with open(path, "rb") as f:
        preprocessor = pickle.load(f)
    
    X_processed = preprocessor.transform(dataset)
    
    # Return tuple (X, y) so predict() can score it
    return X_processed, y_true


def load(clfName: str):
    """
    Loads {clfName}_classifier.pkl.
    """
    path = f"models/{clfName}_classifier.pkl"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Classifier for method '{clfName}' not found at {path}")
        
    print(f"Loading classifier from {path}...")
    
    with open(path, "rb") as f:
        if clfName in ['ff', 'tb', 'tf']:
            try:
                clf = pickle.load(f)
            except Exception:
                # Fix for GPU->CPU or PyTorch 2.6 security
                f.seek(0)
                clf = torch.load(f, map_location='cpu', weights_only=False)
            
            if isinstance(clf, torch.nn.Module):
                clf.to('cpu')
                clf.eval()
        else:
            clf = pickle.load(f) 
            
    return clf


def predict(dataset_processed, model):
    """
    Predicts and evaluates.
    Input:
        dataset_processed: Tuple (X, y) OR just X (if no labels available)
        model: Loaded model
    Returns:
        dict: {'acc': float, 'bacc': float, 'f1': float, 'predictions': array}
    """
    # 1. Unpack Input
    # Check if input is a tuple (X, y) from our new preprocess
    if isinstance(dataset_processed, tuple):
        X, y_true = dataset_processed
    else:
        X = dataset_processed
        y_true = None

    # 2. Generate Predictions (Indices 0-6)
    # -------------------------------------
    # PyTorch
    if isinstance(model, torch.nn.Module):
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X
            
        with torch.no_grad():
            logits = model(X_tensor)
            y_pred_idx = torch.argmax(logits, dim=1).numpy()
    # Scikit-Learn
    else:
        y_pred_idx = model.predict(X)
    
    # Map indices back to labels (A-G) for the final output
    labels_map = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    try:
        y_pred_labels = labels_map[y_pred_idx]
    except IndexError:
        y_pred_labels = y_pred_idx # Fallback

    # 3. Calculate Metrics (If y_true is available)
    # ---------------------------------------------
    metrics = {
        'acc': -1.0,
        'bacc': -1.0,
        'f1': -1.0,
        'predictions': y_pred_labels # Keeping predictions accessible
    }

    if y_true is not None:
        # Ensure y_true has no NaNs (metrics will crash otherwise)
        # We calculate metrics using the INDICES (0-6), not the Letters
        # because y_true was mapped to 0-6 in preprocess
        try:
            metrics['acc'] = accuracy_score(y_true, y_pred_idx)
            metrics['bacc'] = balanced_accuracy_score(y_true, y_pred_idx)
            metrics['f1'] = f1_score(y_true, y_pred_idx, average='weighted')
        except Exception as e:
            print(f"Warning: Metric calculation failed ({e})")

    return metrics