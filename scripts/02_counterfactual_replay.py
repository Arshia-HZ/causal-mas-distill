"""
Script 02: Counterfactual replay for utility estimation.

Performs counterfactual analysis by regenerating downstream messages
when upstream messages are modified.
"""

import argparse
import json
from pathlib import Path

from src.backends.api import APIBackend
from src.counterfactual.replay import CounterfactualReplay
from src.debate.schema import Trace


def main():
    parser = argparse.ArgumentParser(description="Counterfactual replay")
    parser.add_argument("--traces", type=str, required=True, help="Path to traces JSON")
    parser.add_argument("--output", type=str, required=True, help="Output path for counterfactual results")
    parser.add_argument("--api-url", type=str, required=True, help="API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Model name")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of traces to process")
    args = parser.parse_args()

    # Load traces
    with open(args.traces) as f:
        traces_data = json.load(f)
    traces = [Trace.from_dict(t) for t in traces_data[: args.sample_size]]

    # Create backend
    backend = APIBackend(
        base_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
    )

    # Create replayer
    replayer = CounterfactualReplay(backend=backend)

    # Perform counterfactual replay
    results = []
    for i, trace in enumerate(traces):
        print(f"Processing trace {i+1}/{len(traces)}...")

        # For each message, generate counterfactual
        for msg in trace.messages:
            if msg.parent_id is None:
                continue  # Skip root messages

            # Create counterfactual by varying the parent
            cf_trace = replayer.replay_from(trace, msg.mid)
            results.append({
                "original_trace_id": trace.trace_id,
                "modified_mid": msg.mid,
                "counterfactual_trace": cf_trace.to_dict(),
            })

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Generated {len(results)} counterfactual traces")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()