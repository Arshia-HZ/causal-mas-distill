"""
Script 02b: Noise floor estimation for go/no-go decisions.

Computes the noise-floor threshold τ to determine if causal effects
are statistically significant above measurement noise.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backends.api import ApiBackend
from src.counterfactual.replay import noise_floor
from src.debate.schema import Trace
from src.utility.estimator import UtilityEstimator


def main():
    parser = argparse.ArgumentParser(description="Noise floor estimation")
    parser.add_argument("--traces", type=str, required=True, help="Path to traces JSON")
    parser.add_argument("--counterfactuals", type=str, required=True, help="Path to counterfactual utilities JSON")
    parser.add_argument("--output", type=str, required=True, help="Output path for noise floor results")
    parser.add_argument("--api-url", type=str, required=True, help="API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--model", type=str, default="deepseek-v3.2", help="Model name")
    parser.add_argument("--quantile", type=float, default=0.95, help="Quantile for noise floor threshold")
    parser.add_argument("--k", type=int, default=16, help="Samples per condition")
    parser.add_argument("--repeats", type=int, default=2, help="Number of placebo repeats")
    parser.add_argument("--cache-path", type=str, default="cache_nf.jsonl", help="Path to API cache")
    args = parser.parse_args()

    # Load traces and counterfactual utilities
    with open(args.traces) as f:
        traces_data = json.load(f)
    traces = [Trace.from_dict(t) for t in traces_data]
    
    with open(args.counterfactuals) as f:
        utilities_data = json.load(f)

    # Create backend
    backend = ApiBackend(
        base_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        cache_path=args.cache_path,
        supports_n=True,
        extra_body={"thinking": {"type": "disabled"}},  # hidden CoT breaks ablation
    )

    async def compute_noise():
        placebo_diffs = []
        for i, trace in enumerate(traces[:10]):  # Estimate on a subset for speed
            print(f"Estimating noise floor for trace {i+1}/{min(10, len(traces))}...")
            diffs = await noise_floor(trace, backend, k=args.k, repeats=args.repeats)
            placebo_diffs.extend(diffs)
        return placebo_diffs

    # Run placebo estimates
    placebo_diffs = asyncio.run(compute_noise())

    # Estimate noise floor threshold
    estimator = UtilityEstimator()
    tau = estimator.compute_noise_floor_threshold(
        placebo_diffs,
        quantile=args.quantile,
    )

    # Compute significance of the real counterfactuals
    deltas = [u.get("delta", 0.0) for u in utilities_data]
    
    # Determine if effects are significant
    positive_count = sum(delta > tau for delta in deltas)
    negative_count = sum(delta < -tau for delta in deltas)
    go_decision = (positive_count / len(deltas)) >= 0.10 if deltas else False
    
    result = {
        "noise_floor_threshold": tau,
        "n_placebo_samples": len(placebo_diffs),
        "n_counterfactuals": len(deltas),
        "positive_effects": positive_count,
        "negative_effects": negative_count,
        "proportion_positive": positive_count / len(deltas) if deltas else 0,
        "go_decision": go_decision,
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Noise floor threshold τ: {tau:.4f}")
    print(f"Positive effects: {positive_count}/{len(deltas)}")
    print(f"Go decision: {result['go_decision']}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()