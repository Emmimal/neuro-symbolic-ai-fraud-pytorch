"""
sweep.py  —  Hyperparameter sweep (optional — run before app.py)

Experiment 1: λ_consist × λ_sparse grid  (seed=42)
Experiment 2: n_rules ablation  (seed=42, best lambdas)

Saves best_params.json which app.py reads automatically.

Usage:
    python sweep.py   # ~1-2 hours on CPU
    python app.py     # uses saved params
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data         import load_data, make_splits, make_loaders
from models       import HybridRuleLearner
from train        import train_rule_learner
from evaluate     import evaluate_model, find_best_threshold, compute_detection_metrics, compute_rule_metrics
from extract_rules import extract_rules

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

SEED         = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TOTAL_EPOCHS = 80
PATIENCE     = 15
N_THRESHOLDS = 3
BATCH_SIZE   = 2048

# Fix A: sweep around the stronger sparsity range
LAMBDA_GRID = [
    (0.3, 0.3), (0.3, 0.5), (0.3, 0.7),
    (0.5, 0.3), (0.5, 0.5), (0.5, 0.7),
    (0.1, 0.5),
]

N_RULES_OPTIONS = [4, 6, 8]  # Fix C: smaller n_rules


def setup_data(seed):
    df = load_data("creditcard.csv")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_names, pos_weight, scaler) = make_splits(df, seed=seed)
    train_loader, val_loader, test_loader = make_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, BATCH_SIZE)
    return dict(X_train=X_train, X_val=X_val, X_test=X_test,
                y_train=y_train, y_val=y_val, y_test=y_test,
                feature_names=feature_names, pos_weight=pos_weight,
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)


def run_lambda_sweep():
    print("\n" + "="*60)
    print("  EXPERIMENT 1: LAMBDA SWEEP  (seed=42, n_rules=4)")
    print("="*60)
    data = setup_data(SEED)
    rows = []
    torch.manual_seed(SEED); np.random.seed(SEED)

    for lc, ls in LAMBDA_GRID:
        print(f"\n  λ_consist={lc}, λ_sparse={ls}")
        model = HybridRuleLearner(len(data["feature_names"]), N_THRESHOLDS, n_rules=4)
        history = train_rule_learner(
            model, data["train_loader"], data["val_loader"],
            pos_weight=data["pos_weight"], device=DEVICE,
            total_epochs=TOTAL_EPOCHS, patience=PATIENCE,
            lambda_consist=lc, lambda_sparse=ls, verbose=False)

        vp, _, _, _, yv = evaluate_model(model, data["val_loader"], DEVICE, 0.1)
        vt = find_best_threshold(data["y_val"], vp)
        tp, _, _, _, yt = evaluate_model(model, data["test_loader"], DEVICE, 0.1)
        det = compute_detection_metrics(yt, tp, vt)

        X_t = torch.tensor(data["X_test"], dtype=torch.float32).to(DEVICE)
        rm = compute_rule_metrics(model, X_t, data["y_test"], weight_threshold=0.65)

        extracted = extract_rules(model, data["feature_names"], data["X_train"],
                                  weight_threshold=0.65, min_confidence=0.15)
        avg_cond = np.mean([r["n_conditions"] for r in extracted]) if extracted else 0.0

        row = {"lc": lc, "ls": ls,
               "val_pr_auc": history["best_val_pr_auc"],
               "test_f1": det["f1"], "test_pr_auc": det["pr_auc"],
               "fidelity": rm["fidelity"], "coverage": rm["coverage"],
               "avg_conditions": round(avg_cond, 2), "n_active": len(extracted)}
        rows.append(row)
        print(f"    Val PR-AUC={row['val_pr_auc']:.4f} | F1={row['test_f1']:.3f} | "
              f"Fidelity={row['fidelity']:.3f} | AvgCond={row['avg_conditions']:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv("results/sweep_lambda.csv", index=False)
    print(f"\nSaved → results/sweep_lambda.csv")

    readable = df[df["avg_conditions"] < 8]
    if len(readable) == 0: readable = df
    best = readable.loc[readable["val_pr_auc"].idxmax()]
    print(f"★ Best: λ_c={best['lc']}, λ_s={best['ls']} "
          f"(PR-AUC={best['val_pr_auc']:.4f}, avg_cond={best['avg_conditions']:.1f})")
    return df, float(best["lc"]), float(best["ls"])


def run_nrules_ablation(lc=0.3, ls=0.5):
    print("\n" + "="*60)
    print(f"  EXPERIMENT 2: n_rules ABLATION  (λ_c={lc}, λ_s={ls})")
    print("="*60)
    data = setup_data(SEED)
    rows = []
    torch.manual_seed(SEED); np.random.seed(SEED)

    for n_rules in N_RULES_OPTIONS:
        print(f"\n  n_rules={n_rules}")
        model = HybridRuleLearner(len(data["feature_names"]), N_THRESHOLDS, n_rules=n_rules)
        history = train_rule_learner(
            model, data["train_loader"], data["val_loader"],
            pos_weight=data["pos_weight"], device=DEVICE,
            total_epochs=TOTAL_EPOCHS, patience=PATIENCE,
            lambda_consist=lc, lambda_sparse=ls, verbose=False)

        vp, _, _, _, yv = evaluate_model(model, data["val_loader"], DEVICE, 0.1)
        vt = find_best_threshold(data["y_val"], vp)
        tp, _, _, _, yt = evaluate_model(model, data["test_loader"], DEVICE, 0.1)
        det = compute_detection_metrics(yt, tp, vt)
        X_t = torch.tensor(data["X_test"], dtype=torch.float32).to(DEVICE)
        rm = compute_rule_metrics(model, X_t, data["y_test"], weight_threshold=0.65)
        extracted = extract_rules(model, data["feature_names"], data["X_train"],
                                  weight_threshold=0.65, min_confidence=0.15)
        avg_cond = np.mean([r["n_conditions"] for r in extracted]) if extracted else 0.0

        row = {"n_rules": n_rules, "val_pr_auc": history["best_val_pr_auc"],
               "f1": det["f1"], "fidelity": rm["fidelity"],
               "coverage": rm["coverage"], "avg_conditions": round(avg_cond, 2)}
        rows.append(row)
        print(f"    PR-AUC={row['val_pr_auc']:.4f} | F1={row['f1']:.3f} | "
              f"Coverage={row['coverage']:.3f} | AvgCond={row['avg_conditions']:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv("results/sweep_nrules.csv", index=False)
    print(f"Saved → results/sweep_nrules.csv")
    return df


if __name__ == "__main__":
    df_lambda, best_lc, best_ls = run_lambda_sweep()
    df_nrules = run_nrules_ablation(best_lc, best_ls)

    best_params = {"lambda_consist": best_lc, "lambda_sparse": best_ls, "n_rules": 4}
    with open("results/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved → results/best_params.json")
    print(f"Now run: python app.py")
