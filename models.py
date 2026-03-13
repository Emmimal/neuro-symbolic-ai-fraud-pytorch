"""
models.py  —  Neuro-Symbolic Rule Learner
Article: "How a Neural Network Learned Its Own Fraud Rules"

Architecture (parallel paths):
    Input ──> MLP (3 layers, BN) ──────────────> mlp_prob ──> ╮
                                                                α·mlp + (1-α)·rule
    Input ──> Discretizer ──> RuleLearner ────> rule_prob ──> ╯
"""

import torch
import torch.nn as nn


# ── Baseline MLP (identical to Article 1) ────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64),        nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)   # logits


# ── Learnable Discretizer ─────────────────────────────────────────────────────

class LearnableDiscretizer(nn.Module):
    """
    Soft binarization via learnable thresholds + temperature.

        b_{f,t} = sigmoid( (x_f - θ_{f,t}) / τ )

    High τ  → shallow sigmoid  (exploratory)
    Low  τ  → step function    (crisp, readable)
    """

    def __init__(self, n_features: int, n_thresholds: int = 3):
        super().__init__()
        self.n_features   = n_features
        self.n_thresholds = n_thresholds
        self.thresholds   = nn.Parameter(
            torch.randn(n_features, n_thresholds) * 0.5
        )

    @property
    def n_bits(self):
        return self.n_features * self.n_thresholds

    def forward(self, x, temperature=1.0):
        # x: [B, F]  →  bits: [B, F*T]
        x_exp = x.unsqueeze(-1)               # [B, F, 1]
        t_exp = self.thresholds.unsqueeze(0)  # [1, F, T]
        bits  = torch.sigmoid((x_exp - t_exp) / temperature)
        return bits.view(x.size(0), -1)

    def get_thresholds_numpy(self):
        return self.thresholds.detach().cpu().numpy()


# ── Rule Learner ──────────────────────────────────────────────────────────────

class RuleLearner(nn.Module):
    """
    Weighted conjunctions over binarized features.

        rule_r(x) = sigmoid( Σ_i  w_{r,i} · b_i  /  τ )

    Weight interpretation after tanh squashing:
        w >  0.65  →  feature must be HIGH
        w < -0.65  →  feature must be LOW
        |w| < 0.65 →  irrelevant to this rule
    """

    def __init__(self, n_bits: int, n_rules: int = 4):
        super().__init__()
        self.n_rules        = n_rules
        self.rule_weights   = nn.Parameter(torch.randn(n_rules, n_bits) * 0.1)
        self.rule_confidence = nn.Parameter(torch.ones(n_rules))

    def forward(self, bits, temperature=1.0):
        w         = torch.tanh(self.rule_weights)          # [R, n_bits]
        logits    = bits @ w.T                             # [B, R]
        rule_acts = torch.sigmoid(logits / temperature)    # [B, R]
        conf      = torch.softmax(self.rule_confidence, dim=0)  # [R]
        fraud_prob = (rule_acts * conf.unsqueeze(0)).sum(dim=1, keepdim=True)
        return fraud_prob, rule_acts

    def get_weights_numpy(self):
        return torch.tanh(self.rule_weights).detach().cpu().numpy()

    def get_confidence_numpy(self):
        return torch.softmax(self.rule_confidence, dim=0).detach().cpu().numpy()


# ── Hybrid Rule Learner ───────────────────────────────────────────────────────

class HybridRuleLearner(nn.Module):
    """
    Full parallel model. α is learnable (sigmoid-activated, starts at 0.5).
    """

    def __init__(self, input_dim: int, n_thresholds: int = 3, n_rules: int = 4):
        super().__init__()
        self.mlp          = MLP(input_dim)
        self.discretizer  = LearnableDiscretizer(input_dim, n_thresholds)
        self.rule_learner = RuleLearner(input_dim * n_thresholds, n_rules)
        self.alpha_raw    = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw)

    def forward(self, x, temperature=1.0):
        mlp_prob              = torch.sigmoid(self.mlp(x))
        bits                  = self.discretizer(x, temperature)
        rule_prob, rule_acts  = self.rule_learner(bits, temperature)
        final_prob            = self.alpha * mlp_prob + (1.0 - self.alpha) * rule_prob
        return final_prob, mlp_prob, rule_prob, rule_acts


# ── Temperature schedule ──────────────────────────────────────────────────────

def get_temperature(epoch, total_epochs, tau_start=5.0, tau_end=0.1):
    """Exponential decay: τ_start → τ_end over total_epochs."""
    progress = epoch / max(total_epochs - 1, 1)
    return tau_start * (tau_end / tau_start) ** progress
