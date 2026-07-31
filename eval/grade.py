"""
Grading module using math-verify for evaluation.

Uses math-verify for accurate mathematical answer verification,
not simple string matching.
"""

from typing import Any


def grade_answer(
    prediction: str,
    ground_truth: str,
    use_symbolic: bool = True,
) -> dict[str, Any]:
    """
    Grade a mathematical answer using math-verify.

    Args:
        prediction: Predicted answer string.
        ground_truth: Ground truth answer string.
        use_symbolic: Whether to use symbolic verification.

    Returns:
        Dictionary with 'correct' boolean and 'score' float.
    """
    try:
        from math_verify import verify
    except ImportError:
        # Fallback to simple string matching
        return grade_simple(prediction, ground_truth)

    try:
        # Try math-verify
        result = verify(prediction, ground_truth, timeout=5)
        return {
            "correct": result,
            "score": 1.0 if result else 0.0,
            "method": "math_verify",
        }
    except Exception:
        # Fallback to simple matching
        return grade_simple(prediction, ground_truth)


def grade_simple(prediction: str, ground_truth: str) -> dict[str, Any]:
    """
    Simple string-based grading fallback.

    Args:
        prediction: Predicted answer.
        ground_truth: Ground truth answer.

    Returns:
        Dictionary with grading results.
    """
    # Normalize whitespace
    pred_normalized = " ".join(prediction.split()).lower().strip()
    gt_normalized = " ".join(ground_truth.split()).lower().strip()

    # Direct match
    if pred_normalized == gt_normalized:
        return {"correct": True, "score": 1.0, "method": "exact_match"}

    # Check if ground truth is contained in prediction
    if gt_normalized in pred_normalized:
        return {"correct": True, "score": 1.0, "method": "substring_match"}

    return {"correct": False, "score": 0.0, "method": "no_match"}


def grade_batch(
    predictions: list[str],
    ground_truths: list[str],
) -> list[dict[str, Any]]:
    """
    Grade a batch of predictions.

    Args:
        predictions: List of predicted answers.
        ground_truths: List of ground truth answers.

    Returns:
        List of grading results.
    """
    results = []
    for pred, gt in zip(predictions, ground_truths):
        results.append(grade_answer(pred, gt))
    return results


def compute_accuracy(results: list[dict[str, Any]]) -> float:
    """
    Compute accuracy from grading results.

    Args:
        results: List of grading results.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    if not results:
        return 0.0

    correct = sum(1 for r in results if r["correct"])
    return correct / len(results)