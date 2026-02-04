"""
test.py
-------
Inference Module.
Supports: 'lr','rf', 'knn', 'svm', 'ffnn', 'tabnet', 'tabtransformer'.
"""

import pickle
import pandas as pd
import numpy as np
import os
import torch

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
    Loads the appropriate preprocessor and transforms the data.
    
    Mapping:
      - 'lr', 'rf', 'knn', 'svm' -> load models/{clfName}_preprocessor.pkl
      - 'ffnn', 'tabnet', 'tabtransformer' -> load models/ffnn_preprocessor.pkl
    """
    # Drop target if present
    if "grade" in dataset.columns:
        dataset = dataset.drop(columns=["grade"])
    
    # Resolve Preprocessor Path
    # Deep Learning models share the same preprocessor pipeline
    #dl_models = ['ffnn', 'tabnet', 'tabtransformer']
    
    # if clfName in dl_models:
    #     path = "models/ffnn_preprocessor.pkl"
    # else:
    #    path = f"models/{clfName}_preprocessor.pkl"

    path = f"models/{clfName}_preprocessor.pkl"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor for method '{clfName}' not found at {path}")

    # Load & Transform
    with open(path, "rb") as f:
        preprocessor = pickle.load(f)
    X_processed = preprocessor.transform(dataset)
    return X_processed


def load(clfName: str):
    """
    Loads {clfName}_classifier.pkl.
    """
    path = f"models/{clfName}_classifier.pkl"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Classifier for method '{clfName}' not found at {path}")
        
    print(f"Loading classifier from {path}...")
    with open(path, "rb") as f:
        # Load the model
        # map_location='cpu' ensures it works even if trained on GPU
        if clfName in ['ffnn', 'tabnet', 'tabtransformer']:
            clf = pickle.load(f)
            if isinstance(clf, torch.nn.Module):
                clf.to('cpu')
                clf.eval()
        else:
            clf = pickle.load(f) 
    return clf


def predict(dataset_processed, model):
    """
    Predicts using the processed matrix.
    Handles both Scikit-Learn (predict) and PyTorch (forward pass).
    """
    # PyTorch Models
    if isinstance(model, torch.nn.Module):
        # Ensure input is a float32 tensor
        if not isinstance(dataset_processed, torch.Tensor):
            X_tensor = torch.tensor(dataset_processed, dtype=torch.float32)
        else:
            X_tensor = dataset_processed
            
        with torch.no_grad():
            # Model inference
            logits = model(X_tensor)
            # Get class indices
            y_pred_idx = torch.argmax(logits, dim=1).numpy()   
    # Scikit-Learn Models
    else:
        y_pred_idx = model.predict(dataset_processed)
    
    # Map indices back to labels
    labels = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    try:
        y_pred_labels = labels[y_pred_idx]
    except IndexError:
        return y_pred_idx # Fallback
    return y_pred_labels