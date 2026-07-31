"""
Script 01: Generate debates using the teacher model.

Runs the debate harness to generate debate traces for selected problems.
"""

import argparse
import json
from pathlib import Path

from src.backends.api import APIBackend
from src.debate.harness import DebateHarness


def main():
    parser = argparse.ArgumentParser(description="Generate debates")
    parser.add_argument("--problems", type=str, required=True, help="Path to problems JSON")
    parser.add_argument("--output", type=str, required=True, help="Output path for traces")
    parser.add_argument("--api-url", type=str, required=True, help="API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="Model name")
    parser.add_argument("--max-rounds", type=int, default=3, help="Max debate rounds")
    parser.add_argument("--n-solutions", type=int, default=1, help="Solutions per problem")
    args = parser.parse_args()

    # Load problems
    with open(args.problems) as f:
        problems_data = json.load(f)
    problems = [p["problem"] for p in problems_data]

    # Create backend
    backend = APIBackend(
        base_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
    )

    # Create harness
    harness = DebateHarness(
        backend=backend,
        max_rounds=args.max_rounds,
    )

    # Generate debates
    all_traces = []
    for i, problem in enumerate(problems):
        print(f"Generating debate {i+1}/{len(problems)}...")
        traces = harness.run(problem, n_solutions=args.n_solutions)
        all_traces.extend(traces)

    # Save traces
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    traces_data = [t.to_dict() for t in all_traces]
    with open(output_path, "w") as f:
        json.dump(traces_data, f, indent=2)

    print(f"Generated {len(all_traces)} traces")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()