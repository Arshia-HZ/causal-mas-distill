"""
Script 01: Generate debates using the teacher model.

Runs the debate harness to generate debate traces for selected problems.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backends.api import ApiBackend
from src.debate.harness import DebateHarness


def main():
    parser = argparse.ArgumentParser(description="Generate debates")
    parser.add_argument("--problems", type=str, required=True, help="Path to problems JSON")
    parser.add_argument("--output", type=str, required=True, help="Output path for traces")
    parser.add_argument("--api-url", type=str, required=True, help="API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash", help="Model name")
    parser.add_argument("--max-rounds", type=int, default=3, help="Max debate rounds")
    parser.add_argument("--n-solutions", type=int, default=1, help="Solutions per problem")
    parser.add_argument("--cache-path", type=str, default="cache.jsonl", help="Path to API cache")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Output cap per message. The backend default of 768 "
                             "truncated 67%% of messages and cost 9.5%% of solvers "
                             "their final answer. The model context is 8192 total, "
                             "and the verifier must fit 5 messages plus its own "
                             "output, so this cannot go much above 1024.")
    args = parser.parse_args()

    # Load problems
    with open(args.problems) as f:
        problems_data = json.load(f)
        
    problems = []
    for i, p in enumerate(problems_data):
        problems.append({
            "pid": p.get("pid") or p.get("id") or str(i),
            "question": p.get("question") or p.get("problem") or "",
            "gold": p.get("gold") or p.get("answer") or "",
        })

    # Create backend
    backend = ApiBackend(
        base_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        cache_path=args.cache_path,
        max_tokens=args.max_tokens,
        supports_n=True,
        extra_body={"thinking": {"type": "disabled"}},  # hidden CoT breaks ablation
    )

    # Create harness
    harness = DebateHarness(
        backend=backend,
        max_rounds=args.max_rounds,
        max_tokens=args.max_tokens,
    )

    # Generate debates
    print(f"Generating debates for {len(problems)} problems...")
    
    def progress_callback(completed, total):
        if completed % 10 == 0 or completed == total:
            print(f"Generated {completed}/{total} traces...")
            
    all_traces = asyncio.run(
        harness.run_parallel(problems, n_solutions_per_problem=args.n_solutions, progress_callback=progress_callback)
    )

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