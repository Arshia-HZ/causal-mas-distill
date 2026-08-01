"""
Math grading using math-verify.

Provides accurate math answer verification.
"""

from typing import Optional


def is_correct(pred: str, gold: str) -> bool:
    """
    Check if prediction is correct using math-verify.
    
    Falls back to simple string matching if math-verify unavailable.
    
    Args:
        pred: Predicted answer.
        gold: Ground truth answer.
        
    Returns:
        True if prediction matches gold.
    """
    # Normalize for basic comparison
    pred_norm = pred.strip().lower()
    gold_norm = gold.strip().lower()
    
    if pred_norm == gold_norm:
        return True
    
    # Try math-verify
    try:
        from math_verify import parse, verify
        return verify(parse(gold), parse(pred))
    except ImportError:
        pass
    
    return False


def grade_batch(
    predictions: list[str],
    gold_answers: list[str],
) -> list[bool]:
    """
    Grade a batch of predictions.
    
    Args:
        predictions: List of predicted answers.
        gold_answers: List of ground truth answers.
        
    Returns:
        List of boolean correctness values.
    """
    return [is_correct(p, g) for p, g in zip(predictions, gold_answers)]


def accuracy(predictions: list[str], gold_answers: list[str]) -> float:
    """
    Compute accuracy for a batch of predictions.
    
    Args:
        predictions: List of predicted answers.
        gold_answers: List of ground truth answers.
        
    Returns:
        Accuracy as a float between 0 and 1.
    """
    if len(predictions) != len(gold_answers):
        raise ValueError("predictions and gold_answers must have same length")
    
    if len(predictions) == 0:
        return 0.0
    
    results = grade_batch(predictions, gold_answers)
    return sum(results) / len(results)


if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("42", "42", True),
        ("42", "41", False),
        ("x = 42", "42", True),
        ("The answer is 42.", "42", True),
        ("3.14", "3.14", True),
        ("1/2", "0.5", True),
    ]
    
    print("Testing is_correct:")
    for pred, gold, expected in test_cases:
        result = is_correct(pred, gold)
        status = "✓" if result == expected else "✗"
        print(f"  {status} is_correct({pred!r}, {gold!r}) = {result} (expected {expected})")