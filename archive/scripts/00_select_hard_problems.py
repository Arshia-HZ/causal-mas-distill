"""
Script 00: Select hard problems for debate generation.

Selects problems that are challenging enough to benefit from
the debate/iteration process.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Select hard problems for debate")
    parser.add_argument("--input", type=str, required=True, help="Input dataset path")
    parser.add_argument("--output", type=str, required=True, help="Output path for selected problems")
    parser.add_argument("--difficulty-threshold", type=float, default=0.5, help="Difficulty threshold")
    parser.add_argument("--max-problems", type=int, default=1000, help="Maximum number of problems")
    args = parser.parse_args()

    # Load dataset
    with open(args.input) as f:
        data = json.load(f)

    # Filter by difficulty
    selected = [
        item for item in data
        if item.get("difficulty", 1.0) >= args.difficulty_threshold
    ][: args.max_problems]

    # Save selected problems
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"Selected {len(selected)} problems out of {len(data)}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()