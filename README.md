# neuro-symbolic-ai-fraud-pytorch
Neural network that learns its own fraud rules via differentiable rule induction — no hand-coded rules required. PyTorch implementation with temperature annealing.

# Neuro-Symbolic AI Fraud Detection: Rule Learner

> A neural network that writes its own fraud rules — no hand-coding required.

**Article:** [How a Neural Network Learned Its Own Fraud Rules]([https://towardsdatascience.com/](https://towardsdatascience.com/how-a-neural-network-learned-its-own-fraud-rules-a-neuro-symbolic-ai-experiment/))   
**Series Part 1:** [Hybrid Neuro-Symbolic Fraud Detection: Guiding Neural Networks with Domain Rules](https://towardsdatascience.com/hybrid-neuro-symbolic-fraud-detection-guiding-neural-networks-with-domain-rules/)  
**Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · [CC-0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

---

## What This Does

Most neuro-symbolic systems inject rules written by humans. This experiment asks what rules emerge when the model is free to discover them.

After training on 284,807 transactions with 0.17% fraud rate, the model produced:

```
IF   V14 < −1.5σ
AND  V4  > +0.5σ
AND  V12 < −0.9σ
AND  V11 > +0.5σ
AND  V10 < −0.8σ
THEN Fraud                         [conf=0.95, seed=42]
```

The model was never told that V14 was important. It received 30 anonymized PCA features and a gradient signal — and independently rediscovered the feature most strongly correlated with fraud.

---

## Results (5 seeds)

| Model | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| Isolation Forest | 0.121 | 0.172 | 0.941 |
| Pure Neural (Part 1) | 0.804 ± 0.020 | 0.770 ± 0.024 | 0.946 ± 0.019 |
| **Rule Learner (this repo)** | **0.789 ± 0.032** | **0.721 ± 0.058** | **0.933 ± 0.029** |

| Rule Quality Metric | Result | Target |
|---|---|---|
| Fidelity (rules agree with MLP) | 0.993 ± 0.001 | > 0.85 |
| Coverage (fraud caught by ≥1 rule) | 0.811 ± 0.031 | > 0.70 |
| Simplicity (unique features/rule, active seeds) | 5–8 conditions | < 8 |

---

## Quickstart

**1. Clone**
```bash
git clone https://github.com/Emmimal/neuro-symbolic-ai-fraud-pytorch.git
cd neuro-symbolic-ai-fraud-pytorch
```

**2. Install dependencies**
```bash
pip install torch scikit-learn pandas numpy matplotlib seaborn
```

**3. Download the dataset**

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `rule-learner/`.

**4. Run**
```bash
python app.py
```

Runs 5 seeds · extracts rules · saves all figures to `figures/`  
Runtime: ~45 min CPU · ~8 min GPU

**Optional: run hyperparameter sweep first**
```bash
python sweep.py   # finds best λ values → saves best_params.json
python app.py     # picks up best_params.json automatically
```

---

## Code Structure

```
neuro-symbolic-ai-fraud-pytorch/
├── app.py            ← Master runner: 5 seeds, results, figures
├── models.py         ← LearnableDiscretizer + RuleLearner + HybridRuleLearner
├── losses.py         ← BCE + consistency + sparsity + confidence loss
├── train.py          ← Training loop with temperature annealing + early stopping
├── evaluate.py       ← Detection metrics + rule quality (fidelity/coverage/simplicity)
├── extract_rules.py  ← Post-training IF-THEN rule extraction with deduplication
├── sweep.py          ← Lambda grid sweep + n_rules ablation
├── figures.py        ← All article figures
├── data.py           ← Data loading, 70/15/15 stratified split
├── README.md
└── .gitignore        ← creditcard.csv excluded
```

---

## Architecture

```
Input ──→ MLP (3 layers, batch norm) ──────────────→ mlp_prob ──→ ╮
                                                                    α·mlp + (1-α)·rule → fraud_prob
Input ──→ LearnableDiscretizer → RuleLearner ──────→ rule_prob ──→ ╯

α = sigmoid(learnable scalar) · converges ≈ 0.88 on average
```

**Loss:**
```
L_total = L_BCE + 0.3·L_consistency + 0.25·L_sparsity + 0.01·L_confidence
```

---

## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `n_rules` | 4 | Number of learnable rules |
| `n_thresholds` | 3 | Soft threshold cuts per feature |
| `lambda_consist` | 0.3 | Consistency loss weight |
| `lambda_sparse` | 0.25 | Sparsity loss weight (L1 on rule weights) |
| `lambda_conf` | 0.01 | Confidence loss weight (kills noise rules) |
| `weight_threshold` | 0.50 | Min \|w\| for a condition to appear in extracted rule |
| `tau_start / tau_end` | 5.0 → 0.1 | Temperature annealing schedule |
| `total_epochs` | 80 | Max training epochs |
| `patience` | 15 | Early stopping on val PR-AUC |

---

## Extracting Rules

```python
from extract_rules import extract_rules, print_rules

rules = extract_rules(
    model,
    feature_names,
    X_train,
    weight_threshold=0.50,
    min_confidence=0.12
)
print_rules(rules)
```

---

## License

Code: MIT  
Dataset: [CC-0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

---

*Part 2 of a series on neuro-symbolic fraud detection.*  
*Part 1 (rule injection): [github.com/Emmimal/neuro-symbolic-fraud-pytorch](https://github.com/Emmimal/neuro-symbolic-fraud-pytorch)*  
