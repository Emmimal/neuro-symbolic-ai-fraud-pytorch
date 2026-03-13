"""
data.py  —  Data loading and preprocessing

Dataset: Kaggle Credit Card Fraud Detection
Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place creditcard.csv in the same directory as app.py.

Split: 70 / 15 / 15  stratified by Class label.
Features: V1–V28 + Amount + Time  (30 features, StandardScaler-normalized)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def load_data(csv_path="creditcard.csv") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"\ncreditcard.csv not found at '{csv_path}'.\n"
            "Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        )
    # Try UTF-8 first, fall back to latin-1 (handles all single-byte encodings)
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    print(f"Loaded {len(df):,} transactions | "
          f"Fraud: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    return df


def make_splits(df, seed=42, val_size=0.15, test_size=0.15):
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Class"].values.astype(np.float32)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=val_size + test_size,
        random_state=seed, stratify=y)

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=test_size / (val_size + test_size),
        random_state=seed, stratify=y_tmp)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    print(f"Seed {seed} | Train: {len(X_train):,} (fraud: {int(y_train.sum())}) | "
          f"Val: {len(X_val):,} | Test: {len(X_test):,} | "
          f"pos_weight: {pos_weight:.1f}")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            feature_cols, pos_weight, scaler)


def make_loaders(X_train, X_val, X_test,
                 y_train, y_val, y_test,
                 batch_size=2048):
    def _ds(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return _ds(X_train, y_train, True), _ds(X_val, y_val, False), _ds(X_test, y_test, False)
