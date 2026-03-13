"""
extract_rules.py  —  Post-training IF-THEN rule extraction

Converts trained rule weights into human-readable fraud rules:

    Rule 0 [conf=0.72]: IF V4 > +0.8σ AND V14 < -1.0σ  THEN Fraud  (2 conditions)

Fix B: weight_threshold=0.65  (only strong weights become conditions)
Fix E: min_confidence=0.15    (filter noise rules)
"""

import json
import numpy as np
from typing import List, Dict


def extract_rules(
    model,
    feature_names: List[str],
    x_train: np.ndarray,
    weight_threshold: float = 0.50,   # captures |w| ~ 0.55–0.80 after tanh
    min_confidence:   float = 0.12,   # filter noise rules
    temperature:      float = 0.1,
) -> List[Dict]:
    """
    Extract crisp IF-THEN rules from a trained HybridRuleLearner.

    Returns list of rule dicts sorted by confidence (descending).
    """
    model.eval()
    W  = model.rule_learner.get_weights_numpy()    # [R, F*T]
    T  = model.discretizer.get_thresholds_numpy()  # [F, T]
    C  = model.rule_learner.get_confidence_numpy()  # [R]

    n_features   = len(feature_names)
    n_thresholds = model.discretizer.n_thresholds
    feat_mean    = x_train.mean(axis=0)
    feat_std     = x_train.std(axis=0)

    rules = []
    for r_idx in range(W.shape[0]):
        conf = float(C[r_idx])
        if conf < min_confidence:
            continue

        conditions = []
        for f_idx in range(n_features):
            for t_idx in range(n_thresholds):
                w = float(W[r_idx, f_idx * n_thresholds + t_idx])
                if abs(w) <= weight_threshold:
                    continue

                raw_thresh = float(T[f_idx, t_idx])
                if feat_std[f_idx] > 1e-8:
                    sigma = (raw_thresh - feat_mean[f_idx]) / feat_std[f_idx]
                    thresh_str = f"{raw_thresh:.3f} ({sigma:+.1f}σ)"
                else:
                    thresh_str = f"{raw_thresh:.3f}"

                conditions.append({
                    "feature":       feature_names[f_idx],
                    "direction":     ">" if w > 0 else "<",
                    "raw_threshold": raw_thresh,
                    "threshold_str": thresh_str,
                    "weight":        round(w, 4),
                })

        if not conditions:
            continue

        # ── Deduplicate: per (feature, direction) keep only tightest threshold ──
        # n_thresholds=3 means same feature can appear 3x with near-identical cutpoints.
        # Keep the most conservative one: highest threshold for ">", lowest for "<".
        dedup: dict = {}
        for cond in conditions:
            key = (cond["feature"], cond["direction"])
            if key not in dedup:
                dedup[key] = cond
            else:
                prev = dedup[key]["raw_threshold"]
                cur  = cond["raw_threshold"]
                # For ">" keep the highest cutpoint (hardest to satisfy)
                # For "<" keep the lowest cutpoint (hardest to satisfy)
                if cond["direction"] == ">" and cur > prev:
                    dedup[key] = cond
                elif cond["direction"] == "<" and cur < prev:
                    dedup[key] = cond
        conditions = list(dedup.values())

        cond_strs = [f"{c['feature']} {c['direction']} {c['threshold_str']}"
                     for c in conditions]
        rule_str = (f"Rule {r_idx} [conf={conf:.2f}]: "
                    f"IF {' AND '.join(cond_strs)} THEN Fraud")

        rules.append({
            "rule_id":      r_idx,
            "confidence":   round(conf, 4),
            "n_conditions": len(conditions),
            "conditions":   conditions,
            "rule_str":     rule_str,
        })

    rules.sort(key=lambda r: r["confidence"], reverse=True)
    return rules


def print_rules(rules: List[Dict], max_rules: int = 6) -> None:
    sep = "━" * 62
    print(f"\n{sep}")
    print(f"  LEARNED FRAUD RULES  ({len(rules)} active rules)")
    print(sep)
    for rule in rules[:max_rules]:
        conds = " AND ".join(
            f"{c['feature']} {c['direction']} {c['threshold_str']}"
            for c in rule["conditions"]
        )
        print(f"\n  Rule {rule['rule_id']} [conf={rule['confidence']:.2f}]")
        print(f"    IF   {conds}")
        print(f"    THEN Fraud")
        print(f"    ({rule['n_conditions']} conditions)")
    print(f"\n{sep}\n")


def save_rules_json(rules: List[Dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(rules, f, indent=2)
    print(f"Rules saved → {path}")


def get_top_features(rules: List[Dict], top_n: int = 5) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for rule in rules:
        for cond in rule["conditions"]:
            scores[cond["feature"]] = (
                scores.get(cond["feature"], 0.0)
                + rule["confidence"] * abs(cond["weight"])
            )
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n])


def compare_rules_across_seeds(rules_by_seed: dict, top_n: int = 5) -> None:
    print("\n" + "=" * 50)
    print("  TOP FEATURES BY SEED (weighted by rule confidence)")
    print("=" * 50)
    all_features: Dict[str, list] = {}
    for seed, rules in rules_by_seed.items():
        feat_scores = get_top_features(rules, top_n)
        print(f"\nSeed {seed}:")
        for feat, score in feat_scores.items():
            print(f"  {feat:12s}  {score:.4f}")
            all_features.setdefault(feat, []).append(score)

    n_seeds = len(rules_by_seed)
    print("\n" + "─" * 50)
    print("  CROSS-SEED FEATURE CONSISTENCY")
    print("─" * 50)
    print(f"  {'Feature':<14} {'Appears':>8} {'Mean Score':>12}")
    for feat, scores in sorted(all_features.items(),
                                key=lambda x: -len(x[1])):
        print(f"  {feat:<14} {len(scores):>6}/{n_seeds:<1}"
              f"  {np.mean(scores):>10.4f}")
    print()
