"""
Utility estimator for causal effect estimation.

Provides functionality for estimating causal effects (Δ), confidence intervals,
and computing the noise-floor threshold τ for go/no-go decisions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass
class CausalEstimate:
    """Container for causal effect estimate with uncertainty."""
    delta: float  # Estimated causal effect
    ci_lower: float  # Lower bound of confidence interval
    ci_upper: float  # Upper bound of confidence interval
    p_value: float | None = None
    n_samples: int = 0


class UtilityEstimator:
    """
    Estimator for computing causal utility effects.

    Computes Δ (difference in outcomes), confidence intervals,
    and noise-floor threshold for decision making.
    """

    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 1000):
        """
        Initialize the utility estimator.

        Args:
            alpha: Significance level for confidence intervals.
            n_bootstrap: Number of bootstrap samples for CI estimation.
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap

    def estimate(
        self,
        treatment_outcomes: list[float],
        control_outcomes: list[float],
    ) -> CausalEstimate:
        """
        Estimate causal effect from treatment and control outcomes.

        Args:
            treatment_outcomes: Outcomes under treatment condition.
            control_outcomes: Outcomes under control condition.

        Returns:
            CausalEstimate with effect size and confidence interval.
        """
        treatment = np.array(treatment_outcomes)
        control = np.array(control_outcomes)

        # Point estimate: difference in means
        delta = float(np.mean(treatment) - np.mean(control))

        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ci(treatment, control)

        # Sample size
        n = len(treatment) + len(control)

        return CausalEstimate(
            delta=delta,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=n,
        )

    def _bootstrap_ci(
        self,
        treatment: np.ndarray,
        control: np.ndarray,
    ) -> tuple[float, float]:
        """
        Compute bootstrap confidence interval.

        Args:
            treatment: Treatment outcome array.
            control: Control outcome array.

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        n_treat = len(treatment)
        n_ctrl = len(control)
        delta_samples = []

        for _ in range(self.n_bootstrap):
            # Resample with replacement
            treat_sample = np.random.choice(treatment, size=n_treat, replace=True)
            ctrl_sample = np.random.choice(control, size=n_ctrl, replace=True)
            delta_samples.append(np.mean(treat_sample) - np.mean(ctrl_sample))

        delta_samples = np.array(delta_samples)
        lower = np.percentile(delta_samples, 100 * self.alpha / 2)
        upper = np.percentile(delta_samples, 100 * (1 - self.alpha / 2))

        return float(lower), float(upper)

    def compute_noise_floor_threshold(
        self,
        baseline_outcomes: list[float],
        noise_scale: float = 0.1,
    ) -> float:
        """
        Compute the noise-floor threshold τ for go/no-go decisions.

        The threshold is based on the variance of baseline outcomes
        plus expected noise from sampling.

        Args:
            baseline_outcomes: Baseline outcome measurements.
            noise_scale: Expected relative noise scale.

        Returns:
            Threshold value τ.
        """
        baseline = np.array(baseline_outcomes)
        variance = np.var(baseline)
        mean_abs = np.mean(np.abs(baseline))

        # τ = sqrt(variance) + noise_scale * mean(|baseline|)
        tau = np.sqrt(variance) + noise_scale * mean_abs

        return float(tau)

    def is_significant(
        self,
        estimate: CausalEstimate,
        threshold: float | None = None,
    ) -> bool:
        """
        Determine if the estimated effect is statistically significant.

        Args:
            estimate: The causal estimate to evaluate.
            threshold: Optional threshold τ for noise floor comparison.

        Returns:
            True if the effect is significant.
        """
        # Check if CI excludes zero
        excludes_zero = (estimate.ci_lower > 0) or (estimate.ci_upper < 0)

        if threshold is not None:
            # Also check against noise floor
            abs_delta = abs(estimate.delta)
            return excludes_zero and (abs_delta > threshold)

        return excludes_zero