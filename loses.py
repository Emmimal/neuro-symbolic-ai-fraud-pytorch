"""
losses.py  —  Four-part loss function

    L_total = L_BCE
            + λ_consist · L_consistency
            + λ_sparse  · L_sparsity
            + λ_conf    · L_confidence

L_BCE         weighted binary cross-entropy  (pos_weight ≈ 578)
L_consistency MSE(rule_prob, mlp_prob) on confident MLP predictions only
L_sparsity    L1 on raw rule weights  → forces few active conditions per rule
L_confidence  L1 on confidence logits → kills low-confidence noise rules
"""

import torch
import torch.nn.functional as F


def bce_loss(probs, labels, pos_weight):
    probs   = probs.squeeze()
    labels  = labels.float()
    weights = torch.where(labels == 1,
                          torch.full_like(labels, pos_weight),
                          torch.ones_like(labels))
    return F.binary_cross_entropy(probs, labels, weight=weights)


def consistency_loss(rule_prob, mlp_prob, confidence_threshold=0.7):
    """MSE between rule and MLP only where MLP is confident (> 0.7 or < 0.3)."""
    rp   = rule_prob.squeeze()
    mp   = mlp_prob.squeeze().detach()   # detach — MLP is teacher
    mask = (mp > confidence_threshold) | (mp < 1.0 - confidence_threshold)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=rule_prob.device)
    return F.mse_loss(rp[mask], mp[mask])


def sparsity_loss(rule_weights):
    """L1 on raw (pre-tanh) weights — strong gradient toward zero."""
    return rule_weights.abs().mean()


def confidence_sparsity_loss(rule_confidence):
    """L1 on confidence logits — kills low-confidence noise rules."""
    return rule_confidence.abs().mean()


def combined_loss(
    final_prob,
    mlp_prob,
    rule_prob,
    labels,
    rule_weights,
    rule_confidence,
    pos_weight,
    lambda_consist = 0.3,
    lambda_sparse  = 0.5,   # Fix A: strong sparsity
    lambda_conf    = 0.01,  # Fix D: confidence sparsity
):
    """
    Returns (total, l_bce, l_con, l_spa, l_csp).
    Only `total` should be backpropagated.
    """
    l_bce = bce_loss(final_prob, labels, pos_weight)
    l_con = consistency_loss(rule_prob, mlp_prob)
    l_spa = sparsity_loss(rule_weights)
    l_csp = confidence_sparsity_loss(rule_confidence)

    total = l_bce + lambda_consist * l_con + lambda_sparse * l_spa + lambda_conf * l_csp
    return total, l_bce, l_con, l_spa, l_csp
