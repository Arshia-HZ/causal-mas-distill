"""
Math answer extraction and grading.

Hard requirements:
- Never raise. A grader that throws kills an 8-hour unattended replay.
- Handle \\boxed{...}, which is the dominant format in MATH/AIME.
- Return False on unparseable input rather than accidentally matching.
"""

from __future__ import annotations

import re

_BOXED = re.compile(r"\\boxed\s*\{")
_ANSWER_IS = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is)?\s*[:=]?\s*\$?\\?\(?\s*([^\n$]+)",
    re.IGNORECASE,
)
_HASHES = re.compile(r"####\s*(.+)")


def extract_boxed(text: str) -> str | None:
    """Extract the LAST \\boxed{...} with correct brace matching."""
    if not text:
        return None
    last = None
    for m in _BOXED.finditer(text):
        i = m.end()  # just after the opening brace
        depth = 1
        buf = []
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(ch)
            i += 1
        if depth == 0:
            last = "".join(buf).strip()
    return last


def extract_answer(text: str) -> str | None:
    """
    Extract a final answer. Priority: \\boxed{} > #### > 'answer is'.

    Returns None when nothing is found. Never raises.
    """
    if not text:
        return None
    try:
        b = extract_boxed(text)
        if b:
            return b
        m = _HASHES.search(text)
        if m:
            return m.group(1).strip()
        # take the LAST 'answer is' occurrence, not the first
        matches = list(_ANSWER_IS.finditer(text))
        if matches:
            return matches[-1].group(1).strip().rstrip(".,;:").strip()
    except Exception:
        return None
    return None


def _normalize(s: str) -> str:
    """Conservative normalisation for the no-math_verify fallback path."""
    s = s.strip().rstrip(".").strip()
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("$", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace(" ", "")
    s = re.sub(r"^\\text\{(.*)\}$", r"\1", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # thousands separators
    s = s.lower()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def is_correct(pred: str | None, gold: str | None) -> bool:
    """
    Grade one prediction. Never raises; returns False on any failure.

    `pred` may be either a raw generation or an already-extracted answer.
    """
    if pred is None or gold is None:
        return False
    try:
        cand = extract_answer(pred) or pred
        try:
            from math_verify import parse, verify  # type: ignore

            return bool(verify(parse(str(gold)), parse(str(cand))))
        except ImportError:
            pass
        except Exception:
            pass  # fall through to string comparison
        return _normalize(str(cand)) == _normalize(str(gold))
    except Exception:
        return False


def grade_batch(preds: list[str], golds: list[str]) -> list[bool]:
    return [is_correct(p, g) for p, g in zip(preds, golds)]


def accuracy(preds: list[str], golds: list[str]) -> float:
    if len(preds) != len(golds):
        raise ValueError("preds and golds must have equal length")
    if not preds:
        return 0.0
    r = grade_batch(preds, golds)
    return sum(r) / len(r)


# Back-compat alias: eval/run_eval.py imports this name.
def compute_accuracy(results: list[bool]) -> float:
    if not results:
        return 0.0
    return sum(bool(r) for r in results) / len(results)
