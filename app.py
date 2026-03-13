"""
app.py  —  Main 5-seed experiment runner
Article: "How a Neural Network Learned Its Own Fraud Rules"

ALL FIXES APPLIED:
  A: λ_sparse = 0.5        (was 0.1)  — strong weight sparsity
  B: weight_threshold = 0.65 (was 0.3) — only strong weights = active condition
  C: n_rules = 4            (was 8)   — fewer rules, more concentrated
  D: λ_conf = 0.01          (new)     — L1 on confidence logits kills noise rules
  E: min_confidence = 0.15  (was 0.05) — filter low-confidence rules at extraction

Usage:
    1. Place creditcard.csv in this directory
    2. pip install torch scikit-learn pandas numpy matplotlib seaborn
    3. python app.py
"""

import json
import os
import time
import numpy as np
import pandas as pd
import torch

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

from data         import load_data, make_splits, make_loaders
from models       import HybridRuleLearner, MLP
from train        import train_rule_learner, train_baseline_mlp
from evaluate     import (evaluate_model, evaluate_baseline_mlp,
                          find_best_threshold, compute_detection_metrics,
                          compute_rule_metrics)
from extract_rules import (extract_rules, print_rules, save_rules_json,
                           compare_rules_across_seeds)
from figures      import (fig1_architecture, fig2_annealing, fig3_discretization,
                          fig4_multiseed, fig5_fidelity, fig6_rules,
                          fig7_score_distributions)

# ── Hyperparameters (all fixes baked in) ─────────────────────────────────────

SEEDS        = [42, 0, 7, 123, 2024]   # full 5-seed run
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 2048
TOTAL_EPOCHS = 80
PATIENCE     = 15

N_THRESHOLDS   = 3
N_RULES        = 4      # concentrate rules
LAMBDA_CONSIST = 0.3
LAMBDA_SPARSE  = 0.25   # lighter → lets good weights reach |w| ~ 0.55–0.80
LAMBDA_CONF    = 0.01
WEIGHT_THRESH  = 0.50   # lower → captures weights in 0.50–0.80 range
MIN_CONF       = 0.12

# Override with sweep results if available
_p_path = "results/best_params.json"
if os.path.exists(_p_path):
    with open(_p_path) as f:
        _p = json.load(f)
    LAMBDA_CONSIST = _p.get("lambda_consist", LAMBDA_CONSIST)
    LAMBDA_SPARSE  = _p.get("lambda_sparse",  LAMBDA_SPARSE)
    N_RULES        = _p.get("n_rules",         N_RULES)
    print(f"Loaded sweep params: λ_c={LAMBDA_CONSIST}, "
          f"λ_s={LAMBDA_SPARSE}, n_rules={N_RULES}")
else:
    print(f"Using fixed params: λ_consist={LAMBDA_CONSIST}, "
          f"λ_sparse={LAMBDA_SPARSE}, n_rules={N_RULES}")

print(f"\nDevice: {DEVICE}")
print(f"Seeds:  {SEEDS}")
print(f"n_rules={N_RULES}, n_thresholds={N_THRESHOLDS}")
print(f"Epochs={TOTAL_EPOCHS}, Patience={PATIENCE}, Batch={BATCH_SIZE}\n")

# ── Load data ─────────────────────────────────────────────────────────────────

df = load_data("creditcard.csv")

# ── Accumulators ─────────────────────────────────────────────────────────────

results = {
    "pure_neural":  {"f1": [], "pr_auc": [], "roc_auc": [], "recall_at_1pct": []},
    "rule_learner": {"f1": [], "pr_auc": [], "roc_auc": [], "recall_at_1pct": []},
}
rule_metrics_all  = {"fidelity": [], "coverage": [], "simplicity": [], "alpha": []}
seed_histories    = {}
rules_by_seed     = {}
score_dist_seed42 = {}

# ── Main loop ─────────────────────────────────────────────────────────────────

for seed in SEEDS:
    t0 = time.time()
    print("=" * 65)
    print(f"  SEED {seed}")
    print("=" * 65)

    torch.manual_seed(seed)
    np.random.seed(seed)

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_names, pos_weight, scaler) = make_splits(df, seed=seed)

    train_loader, val_loader, test_loader = make_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, BATCH_SIZE)

    # ── 1. Baseline MLP ───────────────────────────────────────────────────────
    print(f"\n  [1/2] Training baseline MLP (seed={seed})...")
    mlp = MLP(input_dim=len(feature_names))
    train_baseline_mlp(
        mlp, train_loader, val_loader,
        pos_weight=pos_weight, device=DEVICE,
        total_epochs=TOTAL_EPOCHS, patience=PATIENCE, verbose=True)

    val_p, y_v = evaluate_baseline_mlp(mlp, val_loader, DEVICE)
    thresh_mlp  = find_best_threshold(y_val, val_p)
    test_p, y_t = evaluate_baseline_mlp(mlp, test_loader, DEVICE)
    det_mlp     = compute_detection_metrics(y_t, test_p, thresh_mlp)

    print(f"\n  Baseline MLP — ROC-AUC={det_mlp['roc_auc']:.4f} | "
          f"PR-AUC={det_mlp['pr_auc']:.4f} | F1={det_mlp['f1']:.4f}")

    for k in ["f1", "pr_auc", "roc_auc"]:
        results["pure_neural"][k].append(det_mlp[k])
    results["pure_neural"]["recall_at_1pct"].append(det_mlp["recall_at_1pct_fpr"])

    # ── 2. Hybrid Rule Learner ────────────────────────────────────────────────
    print(f"\n  [2/2] Training HybridRuleLearner (seed={seed})...")
    rule_model = HybridRuleLearner(
        input_dim=len(feature_names),
        n_thresholds=N_THRESHOLDS,
        n_rules=N_RULES)

    history = train_rule_learner(
        rule_model, train_loader, val_loader,
        pos_weight=pos_weight, device=DEVICE,
        total_epochs=TOTAL_EPOCHS, patience=PATIENCE,
        lambda_consist=LAMBDA_CONSIST,
        lambda_sparse=LAMBDA_SPARSE,
        lambda_conf=LAMBDA_CONF,
        verbose=True)

    val_probs, _, _, _, _ = evaluate_model(rule_model, val_loader, DEVICE, 0.1)
    thresh_rl = find_best_threshold(y_val, val_probs)

    test_probs, mlp_p, rule_p, rule_acts, y_test_np = evaluate_model(
        rule_model, test_loader, DEVICE, 0.1)
    det_rl = compute_detection_metrics(y_test_np, test_probs, thresh_rl)

    print(f"\n  Rule Learner  — ROC-AUC={det_rl['roc_auc']:.4f} | "
          f"PR-AUC={det_rl['pr_auc']:.4f} | F1={det_rl['f1']:.4f}")

    for k in ["f1", "pr_auc", "roc_auc"]:
        results["rule_learner"][k].append(det_rl[k])
    results["rule_learner"]["recall_at_1pct"].append(det_rl["recall_at_1pct_fpr"])

    # Rule quality
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    rm = compute_rule_metrics(rule_model, X_test_t, y_test_np,
                              weight_threshold=WEIGHT_THRESH)
    for k in ["fidelity", "coverage", "simplicity", "alpha"]:
        rule_metrics_all[k].append(rm[k])

    print(f"  Rule metrics  — Fidelity={rm['fidelity']:.4f} | "
          f"Coverage={rm['coverage']:.4f} | "
          f"Simplicity={rm['simplicity']:.2f} | α={rm['alpha']:.3f}")

    # Extract rules
    rules = extract_rules(rule_model, feature_names, X_train,
                          weight_threshold=WEIGHT_THRESH,
                          min_confidence=MIN_CONF)
    print_rules(rules, max_rules=6)
    save_rules_json(rules, f"results/rules_seed{seed}.json")
    rules_by_seed[seed] = rules

    seed_histories[seed] = {
        "tau":        history["tau"],
        "val_pr_auc": history["val_pr_auc"],
        "fidelity_per_epoch": [],
    }

    if seed == 42:
        score_dist_seed42 = {
            "probs":  test_probs.tolist(),
            "y_true": y_test_np.tolist(),
        }
        fig6_rules(rules, seed=42)
        fig7_score_distributions(test_probs, y_test_np, seed=42)

    print(f"\n  Seed {seed} complete in {time.time()-t0:.1f}s\n")


# ── Final results ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  FINAL RESULTS — 5 SEEDS")
print("=" * 65)

print(f"\n  {'Seed':>6} | {'NN F1':>7} | {'RL F1':>7} | "
      f"{'NN ROC':>8} | {'RL ROC':>8} | {'Fidelity':>9} | {'Coverage':>9}")
print("  " + "-" * 70)
for i, seed in enumerate(SEEDS):
    print(f"  {seed:>6} | "
          f"{results['pure_neural']['f1'][i]:>7.3f} | "
          f"{results['rule_learner']['f1'][i]:>7.3f} | "
          f"{results['pure_neural']['roc_auc'][i]:>8.4f} | "
          f"{results['rule_learner']['roc_auc'][i]:>8.4f} | "
          f"{rule_metrics_all['fidelity'][i]:>9.4f} | "
          f"{rule_metrics_all['coverage'][i]:>9.4f}")

rows_summary = []
for model_key, label in [("pure_neural", "Pure Neural (Article 1)"),
                          ("rule_learner", "Rule Learner (This article)")]:
    r = results[model_key]
    row = {
        "Model":    label,
        "F1":       f"{np.mean(r['f1']):.3f} ± {np.std(r['f1']):.3f}",
        "PR-AUC":   f"{np.mean(r['pr_auc']):.3f} ± {np.std(r['pr_auc']):.3f}",
        "ROC-AUC":  f"{np.mean(r['roc_auc']):.3f} ± {np.std(r['roc_auc']):.3f}",
    }
    rows_summary.append(row)
    print(f"\n  {label}:")
    for k in ["F1", "PR-AUC", "ROC-AUC"]:
        print(f"    {k:8s} = {row[k]}")

print("\n  RULE QUALITY (mean ± std across 5 seeds):")
for metric in ["fidelity", "coverage", "simplicity", "alpha"]:
    v = rule_metrics_all[metric]
    print(f"    {metric:12s} = {np.mean(v):.4f} ± {np.std(v):.4f}")

# Cross-seed analysis
print("\n" + "=" * 65)
print("  CROSS-SEED RULE ANALYSIS")
print("=" * 65)
compare_rules_across_seeds(rules_by_seed, top_n=5)

# ── Save results ──────────────────────────────────────────────────────────────

output = {
    "detection":         {"pure_neural": results["pure_neural"],
                          "rule_learner": results["rule_learner"]},
    "rule_quality":      rule_metrics_all,
    "histories":         {str(k): v for k, v in seed_histories.items()},
    "rules_seed42":      rules_by_seed.get(42, []),
    "score_dist_seed42": score_dist_seed42,
    "config": {
        "seeds": SEEDS, "n_rules": N_RULES, "n_thresholds": N_THRESHOLDS,
        "lambda_consist": LAMBDA_CONSIST, "lambda_sparse": LAMBDA_SPARSE,
        "lambda_conf": LAMBDA_CONF, "weight_threshold": WEIGHT_THRESH,
        "min_confidence": MIN_CONF, "total_epochs": TOTAL_EPOCHS,
    }
}

with open("results/five_seed_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("Saved → results/five_seed_results.json")

pd.DataFrame(rows_summary).to_csv("results/summary_table.csv", index=False)

pd.DataFrame({
    "Seed":       SEEDS,
    "Fidelity":   rule_metrics_all["fidelity"],
    "Coverage":   rule_metrics_all["coverage"],
    "Simplicity": rule_metrics_all["simplicity"],
    "Alpha":      rule_metrics_all["alpha"],
}).to_csv("results/rule_quality.csv", index=False)
print("Saved → results/summary_table.csv, results/rule_quality.csv")

# ── Generate figures ──────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  GENERATING FIGURES")
print("=" * 65)

fig1_architecture()
fig2_annealing()
fig3_discretization()
fig4_multiseed(output["detection"])
fig5_fidelity({str(k): v for k, v in seed_histories.items()})

print("\n" + "=" * 65)
print("  ALL DONE")
print("=" * 65)
print(f"\nKey numbers for the article:")
for metric, label in [("f1", "F1"), ("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")]:
    nn = results["pure_neural"][metric]
    rl = results["rule_learner"][metric]
    print(f"  {label:8s}: Neural={np.mean(nn):.3f}±{np.std(nn):.3f}  |  "
          f"RuleLearner={np.mean(rl):.3f}±{np.std(rl):.3f}")
for m in ["fidelity", "coverage", "simplicity"]:
    v = rule_metrics_all[m]
    print(f"  {m:12s}: {np.mean(v):.3f} ± {np.std(v):.3f}")
