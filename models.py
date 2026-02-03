"""
models.py
---------
All DL model definitions, datasets, training loops, and evaluation utilities.

Public API:
    LoanDataset                   – PyTorch Dataset for tabular data
    TabularDataset                – PyTorch Dataset that splits num/cat (for TabTransformer/FTTransformer)
    LoanClassifierFFNN            – Feed-forward neural network
    build_tabnet_classifier(...)  – Configure & return TabNetClassifier
    build_tab_transformer(...)    – Configure & return TabTransformer from library
    build_ft_transformer(...)     – Configure & return FTTransformer (attention on ALL features)
    train_model(...)              – Generic training loop with early stopping
    evaluate_model(...)           – Compute metrics on a loader
    plot_losses(...)              – Plot training vs validation loss
    plot_feature_importances(...) – Plot feature importances with actual names
"""

import os
from typing import Dict, List, Optional, Tuple

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import DEVICE, MODEL_DIR, SEED

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASETS
# ══════════════════════════════════════════════════════════════════════════════

class LoanDataset(Dataset):
    """Generic tabular dataset: all features as a single float tensor."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.num_features = X.shape[1]
        self.num_classes = len(np.unique(y))

        # Validate labels are in [0, num_classes) — an out-of-range label
        # causes a silent CUDA kernel error that surfaces later as
        # "device-side assert triggered" (often inside fix_random or
        # an unrelated call).
        y_min, y_max = int(self.y.min()), int(self.y.max())
        assert y_min >= 0 and y_max < self.num_classes, (
            f"Labels out of range: min={y_min}, max={y_max}, "
            f"num_classes={self.num_classes}.  "
            f"Check your LabelEncoder / target encoding."
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TabularDataset(Dataset):
    """
    Dataset that returns (x_num, x_cat, y) as separate tensors.
    Used by TabTransformer which needs categorical inputs as LongTensor.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, n_num: int):
        """
        Parameters
        ----------
        X     : processed array where columns [0:n_num] are numeric,
                columns [n_num:] are ordinal-encoded categoricals.
        y     : integer-encoded target.
        n_num : number of numeric columns (first n_num cols of X).
        """
        self.x_num = torch.FloatTensor(X[:, :n_num])
        # Categoricals: ensure non-negative integers
        cat_raw = X[:, n_num:]
        #cat_raw = cat_raw + 1  # Shift: -1 becomes 0 (index for unknown), 0 becomes 1, etc.
        self.x_cat = torch.LongTensor(cat_raw.astype(np.int64))
        self.y = torch.LongTensor(y)
        self.num_features = X.shape[1]
        self.n_num = n_num
        self.n_cat = X.shape[1] - n_num
        self.num_classes = len(np.unique(y))

        # Same validation as LoanDataset — prevents CUDA device-side asserts
        y_min, y_max = int(self.y.min()), int(self.y.max())
        assert y_min >= 0 and y_max < self.num_classes, (
            f"Labels out of range: min={y_min}, max={y_max}, "
            f"num_classes={self.num_classes}.  "
            f"Check your LabelEncoder / target encoding."
        )

        # Validate categoricals are within cardinality bounds
        for i in range(self.n_cat):
            col_max = int(self.x_cat[:, i].max())
            if col_max < 0:
                raise ValueError(
                    f"Categorical column {i} has negative values after clipping. "
                    f"Check preprocessing."
                )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FFNN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class LoanClassifierFFNN(nn.Module):
    """
    Feed-Forward Neural Network for tabular classification.

    Architecture: Linear -> LayerNorm -> LeakyReLU -> Dropout  (repeated)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: List[int] = [256, 128, 64],
        dropout_rates: List[float] = [0.3, 0.05],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Dropout: first rate for early layers, second for later layers
        d1 = dropout_rates[0] if len(dropout_rates) > 0 else 0.3
        d2 = dropout_rates[1] if len(dropout_rates) > 1 else d1

        self.fc1 = nn.Linear(input_dim, hidden_units[0])
        self.ln1 = nn.LayerNorm(hidden_units[0])
        self.drop1 = nn.Dropout(d1)

        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.ln2 = nn.LayerNorm(hidden_units[1])
        self.drop2 = nn.Dropout(d1)

        self.fc3 = nn.Linear(hidden_units[1], hidden_units[1])
        self.ln3 = nn.LayerNorm(hidden_units[1])
        self.drop3 = nn.Dropout(d1)

        self.fc4 = nn.Linear(hidden_units[1], hidden_units[2])
        self.ln4 = nn.LayerNorm(hidden_units[2])
        self.drop4 = nn.Dropout(d2)

        self.out = nn.Linear(hidden_units[2], output_dim)

    def forward(self, x):
        x = self.drop1(F.leaky_relu(self.ln1(self.fc1(x))))
        x = self.drop2(F.leaky_relu(self.ln2(self.fc2(x))))
        x = self.drop3(F.leaky_relu(self.ln3(self.fc3(x))))
        x = self.drop4(F.leaky_relu(self.ln4(self.fc4(x))))
        return self.out(x)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  TABNET BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_tabnet_classifier(
    cat_indices: List[int],
    cat_cardinalities: List[int],
    device: torch.device = DEVICE,
    seed: int = SEED,
    # Architecture
    n_d: int = 64,
    n_a: int = 64,
    n_steps: int = 5,
    gamma: float = 1.5,
    n_independent: int = 2,
    n_shared: int = 2,
    cat_emb_dim: int = 8,
    # Optimization
    lr: float = 0.02,
    scheduler_step_size: int = 50,
    scheduler_gamma: float = 0.9,
    # Regularization
    lambda_sparse: float = 1e-4,
    momentum: float = 0.3,
    mask_type: str = "sparsemax",
):
    """
    Build and return a configured (but untrained) TabNetClassifier.

    Fit it later with:
        tabnet_model.fit(X_train=..., y_train=..., eval_set=..., ...)
    """
    from pytorch_tabnet.tab_model import TabNetClassifier

    device_name = str(device)

    model = TabNetClassifier(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        n_independent=n_independent,
        n_shared=n_shared,
        cat_idxs=cat_indices,
        cat_dims=cat_cardinalities,
        cat_emb_dim=cat_emb_dim,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=lr),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=scheduler_step_size, gamma=scheduler_gamma),
        lambda_sparse=lambda_sparse,
        momentum=momentum,
        mask_type=mask_type,
        device_name=device_name,
        seed=seed,
    )

    return model


def train_tabnet(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 1024,
    virtual_batch_size: int = 128,
    class_weights: Optional[Dict] = None,
):
    """
    Fit a TabNetClassifier and return it.

    Parameters
    ----------
    class_weights : dict {class_label: weight} for imbalanced data, or None.
    """
    # Compute class weights if not provided
    if class_weights is None:
        cw = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights = {i: w for i, w in enumerate(cw)}

    model.fit(
        X_train=X_train.astype(np.float32),
        y_train=y_train,
        eval_set=[(X_val.astype(np.float32), y_val)],
        eval_metric=["logloss", "accuracy"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
        num_workers=0,
        drop_last=False,
        weights=class_weights,
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TAB TRANSFORMER BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_tab_transformer(
    cat_cardinalities: Tuple[int, ...],
    num_continuous: int,
    cont_mean_std: np.ndarray,
    num_classes: int = 7,
    # Architecture
    dim: int = 32,
    depth: int = 6,
    heads: int = 8,
    attn_dropout: float = 0.1,
    ff_dropout: float = 0.1,
    mlp_hidden_mults: Tuple[int, ...] = (4, 2),
    mlp_act=None,
    device: torch.device = DEVICE,
):
    """
    Build a TabTransformer using the `tab_transformer_pytorch` library.

    Returns the model moved to `device`.
    """
    from tab_transformer_pytorch import TabTransformer

    if mlp_act is None:
        mlp_act = nn.ReLU()

    cont_mean_std_tensor = torch.FloatTensor(cont_mean_std).to(device)

    model = TabTransformer(
        categories=tuple(cat_cardinalities),
        num_continuous=num_continuous,
        dim=dim,
        dim_out=num_classes,
        depth=depth,
        heads=heads,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        mlp_hidden_mults=mlp_hidden_mults,
        mlp_act=mlp_act,
        continuous_mean_std=cont_mean_std_tensor,
    ).to(device)

    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4b. FT-TRANSFORMER BUILDER (attention over ALL features)
# ══════════════════════════════════════════════════════════════════════════════

def build_ft_transformer(
    cat_cardinalities: Tuple[int, ...],
    num_continuous: int,
    num_classes: int = 7,
    # Architecture
    dim: int = 64,
    depth: int = 4,
    heads: int = 8,
    attn_dropout: float = 0.1,
    ff_dropout: float = 0.1,
    device: torch.device = DEVICE,
):
    """
    Build an FTTransformer (Feature Tokenizer Transformer) using the
    `tab_transformer_pytorch` library.

    Unlike TabTransformer, FTTransformer tokenizes ALL features (numerical
    AND categorical) into embeddings, then applies self-attention across
    all of them.  This is much better suited for datasets with many
    numerical features.

    Same forward interface as TabTransformer: model(x_categ, x_numer) -> logits.

    Returns the model moved to `device`.
    """
    from tab_transformer_pytorch import FTTransformer

    model = FTTransformer(
        categories=tuple(cat_cardinalities),
        num_continuous=num_continuous,
        dim=dim,
        dim_out=num_classes,
        depth=depth,
        heads=heads,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
    ).to(device)

    return model


# ══════════════════════════════════════════════════════════════════════════════
# 5.  GENERIC TRAINING LOOP (for FFNN, TabTransformer & FTTransformer)
# ══════════════════════════════════════════════════════════════════════════════

def _run_epoch_standard(model, loader, criterion, optimizer, device, is_train=True, max_grad_norm=None):
    """One epoch for models that take a single (X, y) input."""
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    ctx = torch.no_grad() if not is_train else torch.enable_grad()

    with ctx:
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            if is_train:
                loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            total_loss += loss.item()

    return total_loss / len(loader)


def _run_epoch_tab_transformer(model, loader, criterion, optimizer, device, is_train=True, max_grad_norm=None):
    """One epoch for TabTransformer that takes (x_num, x_cat, y)."""
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    ctx = torch.no_grad() if not is_train else torch.enable_grad()

    with ctx:
        for x_num, x_cat, targets in loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            targets = targets.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(x_cat, x_num)
            loss = criterion(outputs, targets)
            if is_train:
                loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            total_loss += loss.item()

    return total_loss / len(loader)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 300,
    patience: int = 25,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.3,
    class_weights: Optional[np.ndarray] = None,
    device: torch.device = DEVICE,
    save_name: str = "best_model.pkl",
    model_type: str = "standard",  # "standard" or "tab_transformer"
    max_grad_norm: Optional[float] = None,  # gradient clipping (e.g. 1.0)
    use_cosine_scheduler: bool = False,  # True = CosineAnnealingLR
    warmup_epochs: int = 0,  # linear warmup
):
    """
    Train a PyTorch model with early stopping and LR scheduling.

    Parameters
    ----------
    model_type : "standard"  – model(X)  -> logits  (FFNN)
                 "tab_transformer" – model(x_cat, x_num) -> logits
    class_weights : array of per-class weights, or None.
    max_grad_norm : if set, clip gradient norms to this value each step.
    use_cosine_scheduler : if True, use CosineAnnealingLR (smooth decay over
                           num_epochs). Otherwise use ReduceLROnPlateau.
    warmup_epochs : linearly ramp LR from ~0 to learning_rate over this
                    many epochs before handing off to the main scheduler.

    Returns
    -------
    model, train_losses, val_losses
    """
    # 1. Setup Criterion
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    # 2. Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    if use_cosine_scheduler:
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6
        )
    else:
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_factor,
            patience=scheduler_patience, min_lr=1e-6,
        )

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_epochs,
        )

    # 3. Helper to run one epoch
    run_epoch = (
        _run_epoch_tab_transformer if model_type == "tab_transformer"
        else _run_epoch_standard
    )

    best_val_loss = float("inf")
    counter = 0
    train_losses, val_losses = [], []
    save_path = os.path.join(MODEL_DIR, save_name)

    # 4. Training Loop
    for ep in range(1, num_epochs + 1):
        t_loss = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True, max_grad_norm=max_grad_norm)
        v_loss = run_epoch(model, val_loader, criterion, None, device, is_train=False)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        # Scheduler Step
        if ep <= warmup_epochs and warmup_epochs > 0:
            warmup_scheduler.step()
        elif use_cosine_scheduler:
            main_scheduler.step()
        else:
            main_scheduler.step(v_loss)

        # 5. Save Logic (The important part!)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            counter = 0
            
            # --- CRITICAL STEP FOR PICKLE ---
            # 1. Move model to CPU (so pickle is portable)
            model.cpu()
            
            # 2. Pickle the object
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
            
            # 3. Move back to Device (to continue training loop)
            model.to(device)
            
        else:
            counter += 1

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {ep:3d}/{num_epochs} | "
            f"Train {t_loss:.4f} | Val {v_loss:.4f} | "
            f"LR {lr_now:.2e} | "
            f"{'★ best' if counter == 0 else f'no improve {counter}/{patience}'}"
        )

        if counter >= patience:
            print(f"Early stopping at epoch {ep}")
            break

    # 6. Load Best Model (using Standard Pickle)
    with open(save_path, "rb") as f:
        model = pickle.load(f)
    
    model.to(device)
    print(f"Loaded best model (val loss = {best_val_loss:.4f}) from {save_path}")

    return model, train_losses, val_losses


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(model, loader, device=DEVICE, model_type="standard"):
    """
    Run inference on a DataLoader.

    Returns
    -------
    y_true, y_pred, y_logits  (all numpy arrays)
    """
    model.eval()
    all_true, all_pred, all_logits = [], [], []

    for batch in loader:
        if model_type == "tab_transformer":
            x_num, x_cat, targets = batch
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            outputs = model(x_cat, x_num)
        else:
            data, targets = batch
            data = data.to(device)
            outputs = model(data)

        _, predicted = torch.max(outputs, 1)
        all_true.append(targets.cpu().numpy())
        all_pred.append(predicted.cpu().numpy())
        all_logits.append(outputs.cpu().numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_logits),
    )


def evaluate_model(
    model,
    loader,
    device=DEVICE,
    class_names=None,
    model_type="standard",
    set_name="Test",
):
    """
    Evaluate a model and print classification report.

    Returns
    -------
    dict with accuracy, balanced_accuracy, f1_weighted, f1_macro
    """
    y_true, y_pred, _ = predict(model, loader, device, model_type)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted")
    f1_m = f1_score(y_true, y_pred, average="macro")

    print(f"\n{'='*50}")
    print(f"  {set_name} Set Evaluation")
    print(f"{'='*50}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Balanced Accuracy : {bal_acc:.4f}")
    print(f"  F1 (weighted)     : {f1_w:.4f}")
    print(f"  F1 (macro)        : {f1_m:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=class_names))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_losses(train_losses, val_losses, title="Training vs Validation Loss", save_path=None):
    """Plot train/val loss curves."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix", save_path=None):
    """Plot a confusion matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_feature_importances(
    importances: np.ndarray,
    feature_names: list,
    top_k: int = 20,
    title: str = "Feature Importances",
    save_path: str = None,
):
    """
    Plot top-k feature importances with actual feature names.

    Parameters
    ----------
    importances   : 1-D array of importance scores (one per feature).
    feature_names : list of feature name strings, same length as importances.
    top_k         : how many features to show.
    title         : plot title.
    save_path     : if set, save the figure to this path.
    """
    import matplotlib.pyplot as plt

    assert len(importances) == len(feature_names), (
        f"importances ({len(importances)}) and feature_names "
        f"({len(feature_names)}) must have the same length"
    )

    top_k = min(top_k, len(importances))
    indices = np.argsort(importances)[-top_k:]
    names = [feature_names[i] for i in indices]
    values = importances[indices]

    plt.figure(figsize=(10, max(6, top_k * 0.35)))
    plt.barh(range(top_k), values, color="steelblue")
    plt.yticks(range(top_k), names)
    plt.title(f"{title} (top {top_k})")
    plt.xlabel("Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 8.  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def get_class_weights(y_train: np.ndarray) -> np.ndarray:
    """Compute balanced class weights for CrossEntropyLoss."""
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    return cw.astype(np.float32)


def get_class_weights_dict(y_train: np.ndarray) -> dict:
    """Compute balanced class weights as a dict {label: weight} for TabNet."""
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    return {i: w for i, w in enumerate(cw)}


def create_loaders(
    train_dataset,
    val_dataset,
    test_dataset=None,
    batch_size: int = 256,
):
    """Create DataLoaders from Datasets."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader