"""
Script 05: Evaluate trained models.

Evaluates trained models on test sets and compares against baselines.
"""

import argparse
import json
from pathlib import Path

from eval.run_eval import run_evaluation, evaluate_against_baselines


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline model for comparison")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    args = parser.parse_args()

    # Load test data
    with open(args.test_data) as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} test examples")

    # Run evaluation
    print(f"Evaluating model: {args.model}")
    metrics = run_evaluation(
        model_path=args.model,
        test_data=test_data,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
    )

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Correct: {metrics['n_correct']}/{metrics['n_examples']}")

    # Compare with baseline if provided
    if args.baseline:
        print(f"\nComparing with baseline: {args.baseline}")
        comparison = evaluate_against_baselines(
            trained_model_path=args.model,
            baseline_model=args.baseline,
            test_data=test_data,
        )

        print(f"Trained model accuracy: {comparison['trained_accuracy']:.4f}")
        print(f"Baseline accuracy: {comparison['baseline_accuracy']:.4f}")
        print(f"Improvement: {comparison['improvement']:.4f}")

        # Save comparison
        comparison_path = Path(args.output_dir) / "comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison saved to {comparison_path}")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()