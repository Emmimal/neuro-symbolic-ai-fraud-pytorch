"""
figures.py  —  All article figures

Static figures (no data needed):
    fig1_architecture()
    fig2_annealing()
    fig3_discretization()

Data-dependent figures (run after app.py):
    fig4_multiseed(detection_dict)
    fig5_fidelity(histories_dict)
    fig6_rules(rules_list, seed)
    fig7_score_distributions(probs, y_true, seed)

All axhline/transform bugs fixed.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

BLUE   = "#2563EB"
ORANGE = "#EA580C"
GREEN  = "#16A34A"
GRAY   = "#6B7280"
DARK   = "#111827"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})


# ── Fig 1 — Architecture diagram ─────────────────────────────────────────────

def fig1_architecture():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    def box(x, y, w, h, text, color, fs=9):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color,
                                   edgecolor="white", linewidth=1.5, zorder=3))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold", zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=DARK, lw=1.5))

    box(0.1, 1.5, 1.2, 1.0, "Input\n(30 feats)", GRAY)
    box(1.8, 2.3, 2.0, 0.9, "MLP\n3 layers\nbatch norm", BLUE)
    box(4.2, 2.3, 1.6, 0.9, "mlp_prob", BLUE)
    box(1.8, 0.8, 2.0, 0.9, "Learnable\nDiscretizer", GREEN)
    box(4.2, 0.8, 1.6, 0.9, "Rule\nLearner", GREEN)
    box(6.2, 0.8, 1.6, 0.9, "rule_prob", GREEN)
    box(8.1, 1.4, 1.7, 1.2, "α·mlp\n+\n(1-α)·rule\nfraud_prob", ORANGE)

    arrow(1.3, 2.0, 1.8, 2.75); arrow(3.8, 2.75, 4.2, 2.75); arrow(5.8, 2.75, 8.1, 2.0)
    arrow(1.3, 2.0, 1.8, 1.25); arrow(3.8, 1.25, 4.2, 1.25)
    arrow(5.8, 1.25, 6.2, 1.25); arrow(7.8, 1.25, 8.1, 1.8)

    ax.text(2.8, 3.4, "MLP Path",  fontsize=9, color=BLUE,  ha="center", style="italic")
    ax.text(2.8, 0.4, "Rule Path", fontsize=9, color=GREEN, ha="center", style="italic")
    ax.set_title("Hybrid Rule Learner — Parallel Architecture",
                 fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig("figures/fig1_architecture.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("Saved → figures/fig1_architecture.png")


# ── Fig 2 — Temperature annealing ────────────────────────────────────────────

def fig2_annealing():
    total = 80
    epochs = np.arange(total)
    tau    = 5.0 * (0.1 / 5.0) ** (epochs / (total - 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, tau, color=ORANGE, linewidth=2.5)
    ax.fill_between(epochs, tau, alpha=0.12, color=ORANGE)

    for ep, label in [(0, "Epoch 0\nFully soft"), (40, "Tightening"), (79, "Near-crisp")]:
        t = float(5.0 * (0.1/5.0) ** (ep/(total-1)))
        ax.axvline(x=ep, color=GRAY, linestyle=":", linewidth=1, alpha=0.7)
        ax.annotate(label, xy=(ep, t), xytext=(ep+2, t+0.6),
                    fontsize=8, arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))

    ax.set_xlabel("Epoch", fontsize=11); ax.set_ylabel("Temperature τ", fontsize=11)
    ax.set_title("Temperature Annealing: Rules Harden as τ Decreases", fontsize=11)
    ax.set_xlim(0, 79); ax.set_ylim(0, 5.8)
    plt.tight_layout()
    plt.savefig("figures/fig2_annealing.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → figures/fig2_annealing.png")


# ── Fig 3 — Feature discretization ───────────────────────────────────────────

def fig3_discretization():
    x = np.linspace(-4, 4, 500)
    thresholds = [-1.5, 0.0, 1.5]
    taus = [(5.0, BLUE, "τ = 5.0  (soft)"), (0.1, ORANGE, "τ = 0.1  (crisp)")]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    fig.suptitle("Learnable Discretizer — Three Thresholds per Feature",
                 fontsize=11, fontweight="bold")

    for i, (ax, thresh) in enumerate(zip(axes, thresholds)):
        for tau, color, label in taus:
            ax.plot(x, 1/(1+np.exp(-(x-thresh)/tau)),
                    color=color, linewidth=2.3, label=label)
        ax.axvline(x=thresh, color=GRAY, linestyle="--", linewidth=1.2, alpha=0.6)
        ax.set_xlabel("Feature value (standardized)", fontsize=9)
        ax.set_title(f"Threshold {i+1}  (θ = {thresh})", fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        if i == 0: ax.set_ylabel("Soft bit value", fontsize=9)
        if i == 1: ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig("figures/fig3_discretization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → figures/fig3_discretization.png")


# ── Fig 4 — Multi-seed bar chart ──────────────────────────────────────────────

def fig4_multiseed(results: dict):
    models  = ["pure_neural", "rule_learner"]
    labels  = ["Pure Neural\n(Article 1)", "Rule Learner\n(This article)"]
    colors  = [BLUE, ORANGE]
    metrics = ["f1", "pr_auc"]
    titles  = ["F1 Score", "PR-AUC"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Detection Performance Across 5 Seeds  |  mean ± std",
                 fontsize=12, fontweight="bold")

    x = np.arange(len(models))
    for ax, metric, title in zip(axes, metrics, titles):
        means = [np.mean(results[m][metric]) for m in models]
        stds  = [np.std(results[m][metric])  for m in models]
        bars  = ax.bar(x, means, width=0.5, color=colors, alpha=0.85,
                       yerr=stds, capsize=6,
                       error_kw={"elinewidth": 1.5, "capthick": 1.5})
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + std + 0.005,
                    f"{mean:.3f}\n±{std:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, min(1.05, max(means)+max(stds)+0.12))

    plt.tight_layout()
    plt.savefig("figures/fig4_multiseed_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → figures/fig4_multiseed_bar.png")


# ── Fig 5 — Rule fidelity vs epoch ───────────────────────────────────────────

def fig5_fidelity(histories_by_seed: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Val PR-AUC and Temperature vs Epoch",
                 fontsize=11, fontweight="bold")

    seed_colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(histories_by_seed)))
    for (seed, hist), color in zip(histories_by_seed.items(), seed_colors):
        prauc = hist.get("val_pr_auc", [])
        if prauc:
            ax1.plot(range(len(prauc)), prauc, color=color, alpha=0.7,
                     linewidth=1.3, label=f"Seed {seed}")

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val PR-AUC")
    ax1.set_title("Val PR-AUC per Epoch (all seeds)")
    ax1.set_ylim(0, 1.0)
    if len(histories_by_seed) <= 6:
        ax1.legend(fontsize=8)

    any_hist = next(iter(histories_by_seed.values()))
    tau_vals = any_hist.get("tau", [])
    if tau_vals:
        ax2.plot(range(len(tau_vals)), tau_vals, color=GREEN, linewidth=2.5)
        ax2.fill_between(range(len(tau_vals)), tau_vals, alpha=0.15, color=GREEN)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Temperature τ")
        ax2.set_title("Temperature Annealing Schedule")

    plt.tight_layout()
    plt.savefig("figures/fig5_fidelity_epoch.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → figures/fig5_fidelity_epoch.png")


# ── Fig 6 — Extracted rules (viral figure) ───────────────────────────────────

def fig6_rules(rules: list, seed: int = 42, max_rules: int = 6):
    if not rules:
        print("No rules to visualize — skipping fig6.")
        return

    rules_to_show = rules[:max_rules]
    n_rows = len(rules_to_show)
    fig_h  = max(3.0, 1.5 + n_rows * 1.0)

    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor(DARK)

    ax.text(0.5, 0.97, f"LEARNED FRAUD RULES  —  seed={seed}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=13, fontweight="bold", color="white", fontfamily="monospace")

    ax.text(0.5, 0.90,
            "Extracted post-training via weight thresholding  |  "
            "Rules were never hand-coded",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color=GRAY, style="italic")

    # Separator line via ax.plot (not axhline — avoids transform bug)
    ax.plot([0.03, 0.97], [0.86, 0.86], color=ORANGE, linewidth=2,
            transform=ax.transAxes, clip_on=False)

    y_start    = 0.80
    row_height = 0.75 / max(n_rows, 1)

    for i, rule in enumerate(rules_to_show):
        y = y_start - i * row_height

        ax.text(0.03, y,
                f"Rule {rule['rule_id']}  [conf={rule['confidence']:.2f}]",
                transform=ax.transAxes, fontsize=9.5, fontweight="bold",
                color=ORANGE, fontfamily="monospace")

        conds = "  AND  ".join(
            f"{c['feature']} {c['direction']} {c['threshold_str']}"
            for c in rule["conditions"]
        )
        ax.text(0.03, y - 0.055,
                f"  IF  {conds}  →  FRAUD",
                transform=ax.transAxes, fontsize=8.5, color="white",
                fontfamily="monospace")

        # Row separator
        if i < n_rows - 1:
            sep_y = y - 0.10
            ax.plot([0.02, 0.98], [sep_y, sep_y],
                    color="#374151", linewidth=0.8,
                    transform=ax.transAxes, clip_on=False)

    ax.text(0.5, 0.02,
            "Model never told which features to use  ·  "
            "Rules emerged from gradient descent alone  ·  "
            "github.com/Emmimal/neuro-symbolic-fraud-pytorch",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.5, color=GRAY, style="italic")

    plt.tight_layout()
    plt.savefig(f"figures/fig6_rules_seed{seed}.png",
                dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print(f"Saved → figures/fig6_rules_seed{seed}.png")


# ── Fig 7 — Score distributions ──────────────────────────────────────────────

def fig7_score_distributions(probs: np.ndarray, y_true: np.ndarray, seed: int = 42):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs[y_true == 0], bins=60, alpha=0.6, color=BLUE,
            label="Non-fraud", density=True)
    ax.hist(probs[y_true == 1], bins=30, alpha=0.7, color=ORANGE,
            label="Fraud", density=True)
    ax.set_xlabel("Predicted fraud probability", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Score Distributions — Rule Learner (seed={seed})", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"figures/fig7_score_dist_seed{seed}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → figures/fig7_score_dist_seed{seed}.png")


# ── Standalone: regenerate static figures ────────────────────────────────────

if __name__ == "__main__":
    print("Generating static figures...")
    fig1_architecture()
    fig2_annealing()
    fig3_discretization()

    results_path = "results/five_seed_results.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
        fig4_multiseed(data["detection"])
        fig5_fidelity(data["histories"])
        fig6_rules(data["rules_seed42"], seed=42)
        if "score_dist_seed42" in data:
            sd = data["score_dist_seed42"]
            fig7_score_distributions(
                np.array(sd["probs"]), np.array(sd["y_true"]), seed=42)
        print("All figures saved.")
    else:
        print("Run app.py first for data-dependent figures.")
