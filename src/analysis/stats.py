"""
Statistics for message-utility experiments.

Everything here exists because binary-reward ablation at feasible k is a
low-SNR measurement problem. Point estimates without these are not reportable.

- paired_bootstrap_ci    : the only correct CI for "arm A vs arm B on the same items"
- cluster_bootstrap_ci   : same, but resampling problems, because messages
                           inside one trace are NOT independent observations
- required_k             : power calculation. Run BEFORE spending API budget.
- eb_shrink              : empirical-Bayes shrinkage; replaces hard significance gates
- benjamini_hochberg     : FDR control for REPORTED claims (never for selection)
- variance_decomposition : how much of the spread in delta is measurement noise
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    n: int

    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def paired_bootstrap_ci(a, b, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> Interval:
    """
    CI for mean(a - b) where a[i] and b[i] are the SAME item under two arms.

    Resamples items, not arms. This is the correct test for "selector A beats
    selector B on the same eval set", and it is strictly tighter than two
    independent CIs -- which is why unpaired CIs make real effects look null.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    if a.size == 0:
        return Interval(0.0, 0.0, 0.0, 0)

    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return Interval(float(d.mean()), float(lo), float(hi), int(d.size))


def cluster_bootstrap_ci(values, clusters, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> Interval:
    """
    CI for the mean of `values` when observations are grouped by `clusters`.

    Use this for message-level statistics: five messages from one debate are
    one problem's worth of evidence, not five. Resampling messages directly
    understates the CI by roughly sqrt(messages_per_trace).
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return Interval(0.0, 0.0, 0.0, 0)

    order: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        order.setdefault(c, []).append(i)
    groups = [values[idxs] for idxs in order.values()]

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for t in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        boots[t] = np.concatenate([groups[j] for j in pick]).mean()

    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return Interval(float(values.mean()), float(lo), float(hi), len(groups))


def se_of_delta(p_factual: float, p_ablated: float, k: int) -> float:
    """SE of a difference of two independent binomial rates at k samples each."""
    if k <= 0:
        return float("inf")
    return math.sqrt(
        max(p_factual * (1 - p_factual), 0.0) / k
        + max(p_ablated * (1 - p_ablated), 0.0) / k
    )


def required_k(effect: float, p_base: float = 0.5, power: float = 0.8, alpha: float = 0.05) -> int:
    """
    Samples per arm needed to detect a true effect of `effect`.

    Run this before buying API credit. If required_k(0.25) exceeds what you can
    afford, per-message significance testing is not an available method and you
    must use shrinkage (eb_shrink) or a surrogate instead.
    """
    if effect <= 0:
        return 10**9
    z_a = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _z(1 - alpha / 2)
    z_b = 0.8416212335729143 if abs(power - 0.8) < 1e-9 else _z(power)
    p1 = min(max(p_base, 0.0), 1.0)
    p0 = min(max(p_base - effect, 0.0), 1.0)
    var = p1 * (1 - p1) + p0 * (1 - p0)
    return int(math.ceil(((z_a + z_b) ** 2) * var / (effect**2)))


def _z(q: float) -> float:
    """Inverse normal CDF (Acklam rational approximation)."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0,1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    pl, ph = 0.02425, 1 - 0.02425
    if q < pl:
        t = math.sqrt(-2 * math.log(q))
        return (((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5]) / ((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1)
    if q > ph:
        t = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5]) / ((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1)
    t = q - 0.5
    r = t * t
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*t / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def eb_shrink(deltas, ses):
    """
    Empirical-Bayes (James-Stein style) shrinkage of noisy per-message effects.

    This REPLACES a hard significance gate as the selection rule. At k<=32 a
    per-message p-value is not informative enough to threshold on, but the
    posterior mean is still the right thing to rank by: high-variance estimates
    get pulled toward the pooled mean and stop winning the top-k by luck.

        tau2 = max(0, Var(delta) - mean(se^2))    # signal variance
        w_i  = tau2 / (tau2 + se_i^2)             # per-item reliability
        post = mu + w_i * (delta_i - mu)

    `signal_fraction` in the diagnostics is the headline number for a noise
    audit: the share of observed spread that is real rather than sampling noise.
    """
    d = np.asarray(deltas, dtype=float)
    s = np.asarray(ses, dtype=float)
    if d.size == 0:
        return [], {"tau2": 0.0, "mu": 0.0, "signal_fraction": 0.0, "n": 0}

    mu = float(d.mean())
    total_var = float(d.var(ddof=1)) if d.size > 1 else 0.0
    noise_var = float((s**2).mean())
    tau2 = max(total_var - noise_var, 0.0)

    w = tau2 / (tau2 + np.maximum(s**2, 1e-12))
    post = mu + w * (d - mu)

    diag = {
        "mu": mu,
        "tau2": tau2,
        "total_var": total_var,
        "noise_var": noise_var,
        "signal_fraction": (tau2 / total_var) if total_var > 0 else 0.0,
        "mean_weight": float(w.mean()),
        "n": int(d.size),
    }
    return [float(x) for x in post], diag


def benjamini_hochberg(pvals, alpha: float = 0.05):
    """BH-FDR. Use for claims you REPORT, never as the data-selection rule."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    keep = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = int(np.max(np.nonzero(passed)[0]))
        keep[order[: cutoff + 1]] = True
    return [bool(x) for x in keep]


def variance_decomposition(replicate_estimates: dict) -> dict:
    """
    Split observed variance in delta into prediction noise vs data noise.

    `replicate_estimates` maps message_id -> repeated estimates of the SAME
    delta (differing only in sampling seed).

    Report `noise_share`. If it is high, any selector built on these scores is
    ranking noise -- itself a publishable finding about the method class.
    """
    withins, means, reps = [], [], []
    for vals in replicate_estimates.values():
        v = np.asarray(vals, dtype=float)
        if v.size < 2:
            continue
        withins.append(float(v.var(ddof=1)))
        means.append(float(v.mean()))
        reps.append(v.size)
    if not withins:
        return {"error": "need >=2 replicates for >=1 message"}

    within = float(np.mean(withins))
    between = float(np.var(means, ddof=1)) if len(means) > 1 else 0.0
    r = float(np.mean(reps))
    signal = max(between - within / r, 0.0)
    total = signal + within
    return {
        "prediction_noise_var": within,
        "between_message_var": between,
        "estimated_signal_var": signal,
        "noise_share": (within / total) if total > 0 else 1.0,
        "n_messages": len(means),
        "mean_replicates": r,
    }
