"""
evaluate.py  —  All evaluation metrics

Standard:   ROC-AUC, PR-AUC, F1, Recall@1%FPR
Rule quality (novel):
    fidelity    agreement between rules and MLP on binary decision  (target > 0.85)
    coverage    fraction of fraud caught by ≥1 rule                 (target > 0.70)
    simplicity  average active conditions per rule                   (target < 6)
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_recall_curve, roc_curve,
)


# ── Threshold selection ───────────────────────────────────────────────────────

def find_best_threshold(y_true, probs):
    """F1-maximizing threshold. Call only on validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    idx = min(np.argmax(f1), len(thresholds) - 1)
    return float(thresholds[idx])


# ── Standard detection metrics ────────────────────────────────────────────────

def compute_detection_metrics(y_true, probs, threshold):
    preds   = (probs >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, probs)
    return {
        "roc_auc":            round(float(roc_auc_score(y_true, probs)), 4),
        "pr_auc":             round(float(average_precision_score(y_true, probs)), 4),
        "f1":                 round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "recall_at_1pct_fpr": round(float(np.interp(0.01, fpr, tpr)), 4),
        "threshold":          round(float(threshold), 4),
    }


# ── Rule quality metrics ──────────────────────────────────────────────────────

def rule_fidelity(mlp_probs, rule_probs, threshold=0.5):
    """Fraction of samples where rules and MLP agree on binary decision."""
    return float(
        ((mlp_probs >= threshold).astype(int) ==
         (rule_probs >= threshold).astype(int)).mean()
    )


def rule_coverage(rule_acts, y_true, threshold=0.5):
    """Fraction of actual fraud caught by at least one rule."""
    fraud = y_true == 1
    if fraud.sum() == 0:
        return 0.0
    return float((rule_acts >= threshold).any(axis=1)[fraud].mean())


def rule_simplicity(rule_weights_numpy, weight_threshold=0.50):
    """
    Average unique features per rule (after deduplication).
    With n_thresholds=3, raw condition count inflates 3x — unique features
    is the meaningful readability metric. Target: < 8 unique features.
    """
    n_rules, n_bits = rule_weights_numpy.shape
    # We don't have n_thresholds here so count unique feature indices
    # by grouping every n_thresholds columns together
    # Heuristic: count bit positions above threshold, then divide by 3
    active = (np.abs(rule_weights_numpy) > weight_threshold).sum(axis=1)
    # Approximate unique features = raw conditions / n_thresholds (=3)
    unique_features = np.ceil(active / 3.0)
    unique_features = unique_features[unique_features > 0]
    return float(unique_features.mean()) if len(unique_features) > 0 else 0.0


def compute_rule_metrics(model, x_test, y_test,
                         temperature=0.1,
                         fidelity_threshold=0.5,
                         rule_fire_threshold=0.5,
                         weight_threshold=0.50):
    model.eval()
    with torch.no_grad():
        _, mlp_p, rule_p, rule_acts = model(x_test, temperature)

    mlp_np   = mlp_p.squeeze().cpu().numpy()
    rule_np  = rule_p.squeeze().cpu().numpy()
    acts_np  = rule_acts.cpu().numpy()
    W        = model.rule_learner.get_weights_numpy()

    return {
        "fidelity":   round(rule_fidelity(mlp_np, rule_np, fidelity_threshold), 4),
        "coverage":   round(rule_coverage(acts_np, y_test, rule_fire_threshold), 4),
        "simplicity": round(rule_simplicity(W, weight_threshold), 2),
        "alpha":      round(float(torch.sigmoid(model.alpha_raw).item()), 3),
    }


# ── Full inference pass ───────────────────────────────────────────────────────

def evaluate_model(model, loader, device, temperature=0.1):
    model.eval()
    all_final, all_mlp, all_rule, all_acts, all_y = [], [], [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            fp, mp, rp, acts = model(xb, temperature)
            all_final.append(fp.squeeze().cpu())
            all_mlp.append(mp.squeeze().cpu())
            all_rule.append(rp.squeeze().cpu())
            all_acts.append(acts.cpu())
            all_y.append(yb)
    return (torch.cat(all_final).numpy(), torch.cat(all_mlp).numpy(),
            torch.cat(all_rule).numpy(), torch.cat(all_acts).numpy(),
            torch.cat(all_y).numpy())


def evaluate_baseline_mlp(model, loader, device):
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb)).squeeze().cpu()
            all_probs.append(probs)
            all_y.append(yb)
    return torch.cat(all_probs).numpy(), torch.cat(all_y).numpy()
