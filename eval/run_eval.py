"""
Evaluation runner for trained models.

Runs evaluation on test sets and computes metrics
using math-verify for accurate grading.
"""

import json
from pathlib import Path
from typing import Any

from .grade import grade_batch, compute_accuracy


def run_evaluation(
    model_path: str,
    test_data: list[dict[str, Any]],
    output_dir: Path,
    batch_size: int = 8,
) -> dict[str, Any]:
    """
    Run evaluation on a test dataset.

    Args:
        model_path: Path to the trained model.
        test_data: List of test examples with 'problem' and 'answer'.
        output_dir: Directory for evaluation outputs.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary of evaluation metrics.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Generate predictions
    predictions = []
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i : i + batch_size]
        batch_preds = _generate_batch(model, tokenizer, batch)
        predictions.extend(batch_preds)

    # Grade predictions
    ground_truths = [ex["answer"] for ex in test_data]
    results = grade_batch(predictions, ground_truths)

    # Compute metrics
    accuracy = compute_accuracy(results)

    metrics = {
        "accuracy": accuracy,
        "n_examples": len(test_data),
        "n_correct": sum(1 for r in results if r),
        "results": results,
    }

    # Save results
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _generate_batch(
    model: Any,
    tokenizer: Any,
    batch: list[dict[str, Any]],
) -> list[str]:
    """
    Generate predictions for a batch of problems.

    Args:
        model: The model to use.
        tokenizer: The tokenizer.
        batch: Batch of test examples.

    Returns:
        List of generated predictions.
    """
    import torch

    prompts = [ex["problem"] for ex in batch]

    # Tokenize
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )

    # Decode
    predictions = []
    for i, output in enumerate(outputs):
        pred = tokenizer.decode(output, skip_special_tokens=True)
        # Extract answer (everything after the prompt)
        prompt = prompts[i]
        if pred.startswith(prompt):
            pred = pred[len(prompt) :].strip()
        predictions.append(pred)

    return predictions


def evaluate_against_baselines(
    trained_model_path: str,
    baseline_model: str,
    test_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compare trained model against baseline.

    Args:
        trained_model_path: Path to the trained model.
        baseline_model: Name/path of baseline model.
        test_data: Test examples.

    Returns:
        Comparison metrics.
    """
    # Evaluate trained model
    trained_metrics = run_evaluation(trained_model_path, test_data, Path("eval_trained"))

    # Evaluate baseline
    baseline_metrics = run_evaluation(baseline_model, test_data, Path("eval_baseline"))

    return {
        "trained_accuracy": trained_metrics["accuracy"],
        "baseline_accuracy": baseline_metrics["accuracy"],
        "improvement": trained_metrics["accuracy"] - baseline_metrics["accuracy"],
    }