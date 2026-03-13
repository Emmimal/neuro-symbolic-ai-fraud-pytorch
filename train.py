"""
train.py  —  Training loops

HybridRuleLearner training:
  - Temperature annealing τ: 5.0 → 0.1 over total_epochs
  - Early stopping on val PR-AUC with minimum epoch guard
    (rules cannot crystallize before τ drops — guard = 70% of total_epochs)
  - Gradient clipping for stability on imbalanced data

Baseline MLP training: standard BCE + pos_weight only.
"""

import copy
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import average_precision_score

from models import HybridRuleLearner, MLP, get_temperature
from losses import combined_loss
from evaluate import evaluate_model, evaluate_baseline_mlp


def train_rule_learner(
    model, train_loader, val_loader,
    pos_weight, device,
    total_epochs=80, patience=15,
    lr=1e-3, weight_decay=1e-4,
    lambda_consist=0.3,
    lambda_sparse=0.5,    # Fix A: strong sparsity
    lambda_conf=0.01,     # Fix D: confidence sparsity
    tau_start=5.0, tau_end=0.1,
    verbose=True,
):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_prauc, best_state, best_epoch = -1.0, None, 0
    no_improve = 0
    # Fix: early stopping cannot fire before rules have time to crystallize
    min_epochs = max(30, int(total_epochs * 0.70))

    history = {k: [] for k in
               ["train_loss", "l_bce", "l_consist", "l_sparse", "val_pr_auc", "tau", "alpha"]}

    for epoch in range(total_epochs):
        tau = get_temperature(epoch, total_epochs, tau_start, tau_end)
        model.train()
        epoch_loss = epoch_bce = epoch_con = epoch_spa = 0.0
        n = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            final_p, mlp_p, rule_p, _ = model(xb, tau)

            loss, l_bce, l_con, l_spa, l_csp = combined_loss(
                final_p, mlp_p, rule_p, yb,
                model.rule_learner.rule_weights,
                model.rule_learner.rule_confidence,
                pos_weight,
                lambda_consist=lambda_consist,
                lambda_sparse=lambda_sparse,
                lambda_conf=lambda_conf,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            epoch_bce  += l_bce.item()
            epoch_con  += l_con.item() if isinstance(l_con, torch.Tensor) else float(l_con)
            epoch_spa  += l_spa.item()
            n += 1

        # Validation
        val_probs, _, _, _, y_val = evaluate_model(
            model, val_loader, device, temperature=min(tau, 0.5))
        val_prauc = float(average_precision_score(y_val, val_probs))
        alpha = float(torch.sigmoid(model.alpha_raw).item())

        history["train_loss"].append(round(epoch_loss / n, 6))
        history["l_bce"].append(round(epoch_bce / n, 6))
        history["l_consist"].append(round(epoch_con / n, 6))
        history["l_sparse"].append(round(epoch_spa / n, 6))
        history["val_pr_auc"].append(round(val_prauc, 4))
        history["tau"].append(round(tau, 4))
        history["alpha"].append(round(alpha, 3))

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | τ={tau:.3f} | α={alpha:.3f} | "
                  f"loss={epoch_loss/n:.4f} | val_PR-AUC={val_prauc:.4f}")

        if val_prauc > best_prauc:
            best_prauc, best_state, best_epoch = val_prauc, copy.deepcopy(model.state_dict()), epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience and epoch >= min_epochs:
                if verbose:
                    print(f"  Early stop at epoch {epoch} | "
                          f"Best epoch: {best_epoch} | Best val PR-AUC: {best_prauc:.4f}")
                break

    model.load_state_dict(best_state)
    history["best_epoch"]      = best_epoch
    history["best_val_pr_auc"] = best_prauc
    return history


def train_baseline_mlp(
    model, train_loader, val_loader,
    pos_weight, device,
    total_epochs=80, patience=15,
    lr=1e-3, weight_decay=1e-4,
    verbose=True,
):
    import torch.nn.functional as F
    model.to(device)
    pw_t      = torch.tensor(pos_weight, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw_t)
    opt       = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_prauc, best_state, best_epoch = -1.0, None, 0
    no_improve = 0
    history = {"train_loss": [], "val_pr_auc": []}

    for epoch in range(total_epochs):
        model.train()
        epoch_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb).squeeze(), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n += 1

        probs, y_val = evaluate_baseline_mlp(model, val_loader, device)
        val_prauc    = float(average_precision_score(y_val, probs))

        history["train_loss"].append(round(epoch_loss / n, 6))
        history["val_pr_auc"].append(round(val_prauc, 4))

        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | loss={epoch_loss/n:.4f} | "
                  f"val_PR-AUC={val_prauc:.4f}")

        if val_prauc > best_prauc:
            best_prauc, best_state, best_epoch = val_prauc, copy.deepcopy(model.state_dict()), epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch} | "
                          f"Best: {best_epoch} | PR-AUC: {best_prauc:.4f}")
                break

    model.load_state_dict(best_state)
    history["best_epoch"]      = best_epoch
    history["best_val_pr_auc"] = best_prauc
    return history
