"""
Script 02b: Noise floor estimation for go/no-go decisions.

Computes the noise-floor threshold τ to determine if causal effects
are statistically significant above measurement noise.
"""

import argparse
import json
from pathlib import Path
import numpy as np

from src.utility.estimator import UtilityEstimator


def main():
    parser = argparse.ArgumentParser(description="Noise floor estimation")
    parser.add_argument("--traces", type=str, required=True, help="Path to traces JSON")
    parser.add_argument("--counterfactuals", type=str, required=True, help="Path to counterfactual results")
    parser.add_argument("--output", type=str, required=True, help="Output path for noise floor results")
    parser.add_argument("--noise-scale", type=float, default=0.1, help="Expected noise scale")
    args = parser.parse_args()

    # Load traces and counterfactuals
    with open(args.traces) as f:
        traces_data = json.load(f)
    
    with open(args.counterfactuals) as f:
        cf_data = json.load(f)

    # Extract outcomes (placeholder - would use grading function)
    baseline_outcomes = []
    treatment_outcomes = []

    # Simple simulation: use message lengths as proxy for outcomes
    for trace_data in traces_data:
        for msg in trace_data.get("messages", []):
            if msg.get("round_type") == "solve":
                baseline_outcomes.append(len(msg.get("content", "")))

    # Estimate noise floor threshold
    estimator = UtilityEstimator()
    tau = estimator.compute_noise_floor_threshold(
        baseline_outcomes,
        noise_scale=args.noise_scale,
    )

    # Compute delta for counterfactuals
    # (Simplified - actual implementation would compare original vs counterfactual outcomes)
    deltas = []
    for cf in cf_data:
        # Placeholder: use trace length difference as delta
        orig_len = len(str(cf.get("original_trace_id", "")))
        cf_len = len(str(cf.get("counterfactual_trace", {})))
        deltas.append(cf_len - orig_len)

    # Determine if effects are significant
    significant_count = sum(1 for d in deltas if abs(d) > tau)
    
    result = {
        "noise_floor_threshold": tau,
        "n_baseline_samples": len(baseline_outcomes),
        "n_counterfactuals": len(deltas),
        "significant_effects": significant_count,
        "proportion_significant": significant_count / len(deltas) if deltas else 0,
        "go_decision": significant_count > len(deltas) * 0.1,  # >10% significant
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Noise floor threshold τ: {tau:.4f}")
    print(f"Significant effects: {significant_count}/{len(deltas)}")
    print(f"Go decision: {result['go_decision']}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()