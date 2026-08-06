"""
Script 02: Counterfactual replay for utility estimation.

Performs counterfactual analysis by regenerating downstream messages
when upstream messages are modified to estimate causal utility.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backends.api import ApiBackend
from src.counterfactual.replay import trace_utilities
from src.debate.schema import Trace


def main():
    parser = argparse.ArgumentParser(description="Counterfactual replay")
    parser.add_argument("--traces", type=str, required=True, help="Path to traces JSON")
    parser.add_argument("--output", type=str, required=True, help="Output path for counterfactual results")
    parser.add_argument("--api-url", type=str, required=True, help="API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash", help="Model name")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of traces to process")
    parser.add_argument("--k", type=int, default=16, help="Number of counterfactual samples per message")
    parser.add_argument("--cache-path", type=str, default="cache_cf.jsonl", help="Path to API cache")
    args = parser.parse_args()

    # Load traces
    with open(args.traces) as f:
        traces_data = json.load(f)
    traces = [Trace.from_dict(t) for t in traces_data[: args.sample_size]]

    # Create backend
    backend = ApiBackend(
        base_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        cache_path=args.cache_path,
        supports_n=False,  # DeepSeek chat API has no n parameter
        extra_body={"thinking": {"type": "disabled"}},  # hidden CoT breaks ablation
    )

    async def run_all():
        results = []
        for i, trace in enumerate(traces):
            print(f"Processing trace {i+1}/{len(traces)} ({trace.pid})...")
            # Calculate utility for each message
            utilities = await trace_utilities(trace, backend, k=args.k)
            for u in utilities:
                results.append(u.to_dict())
        return results

    # Perform counterfactual replay
    results = asyncio.run(run_all())

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Computed utilities for {len(results)} messages")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()