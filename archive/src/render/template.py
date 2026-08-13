"""
Unified template renderer.

Every selector renders through this one template so that output FORMAT is
held constant and the only thing that varies across arms is WHICH messages
were selected. That is the format control for the whole experiment.

Two correctness requirements:
- Never use str.format on content containing LaTeX. `{\\frac{1}{2}}` raises
  KeyError/IndexError. Build by concatenation.
- The supervised target is the TEACHER's final answer, not the gold label.
  Training on gold turns every arm into answer-only SFT and erases the
  contrast you are trying to measure.
"""

from __future__ import annotations

from ..debate.schema import Trace

SLOTS = ("hypothesis", "disagreement", "error", "correction", "verify", "action")

# Role x position -> slot. Explicit, and independent of round numbering.
_ROLE_SLOT = {
    ("solver", 1): "hypothesis",
    ("critic", 1): "disagreement",
    ("solver", 2): "error",
    ("critic", 2): "correction",
    ("verifier", 1): "verify",
    ("verifier", 2): "verify",
    ("solver", 3): "action",
}


def assign_slots(trace: Trace, selected_mids: list[str]) -> dict[str, str]:
    """Map selected messages into named slots. Unselected slots stay empty."""
    slots = {s: "" for s in SLOTS}
    index = {m.mid: m for m in trace.messages}
    order = {m.mid: i for i, m in enumerate(trace.messages)}
    for mid in sorted(set(selected_mids), key=lambda x: order.get(x, 10**9)):
        m = index.get(mid)
        if m is None:
            continue
        slot = _ROLE_SLOT.get((m.role, m.round))
        if slot is None:
            slot = "verify" if m.role == "verifier" else "action"
        slots[slot] = (slots[slot] + "\n" + m.text).strip() if slots[slot] else m.text
    return slots


def render(trace: Trace, selected_mids: list[str], target: str | None = None) -> str:
    """
    Render one training example. Concatenation only -- no str.format.

    `target` defaults to the teacher's final answer. Pass gold explicitly only
    for an oracle arm, and label that arm as such.
    """
    slots = assign_slots(trace, selected_mids)
    answer = target if target is not None else (trace.final_answer or "")
    parts = ["Question: ", trace.question, "\n\n"]
    for s in SLOTS:
        parts.append("<" + s + ">" + slots[s] + "</" + s + ">\n")
    parts.append("\nAnswer: ")
    parts.append(answer)
    return "".join(parts)


def render_for_sft(
    traces: list[Trace],
    selected_by_trace_id: dict[str, list[str]],
) -> list[dict]:
    """
    Build SFT records. One record per trace, NOT one per message.

    `selected_by_trace_id` maps trace_id -> selected mids for that trace, so selection is
    per-trace and every arm yields the same number of examples.
    """
    out = []
    for t in traces:
        trace_id = getattr(t, "trace_id", getattr(t, "pid", ""))
        mids = selected_by_trace_id.get(trace_id, [])
        text = render(t, mids)
        prompt, _, completion = text.rpartition("\nAnswer: ")
        out.append(
            {
                "pid": t.pid,
                "trace_id": trace_id,
                "text": text,
                "prompt": prompt + "\nAnswer:",
                "completion": " " + completion,
                "n_selected": len(mids),
                "gold": t.gold,
            }
        )
    return out


class TemplateRenderer:
    """Thin OO wrapper kept for back-compat with scripts/03_build_datasets.py."""

    @staticmethod
    def render_for_sft(traces, selected_by_trace_id):
        return render_for_sft(traces, selected_by_trace_id)

    @staticmethod
    def render(trace, selected_mids, target=None):
        return render(trace, selected_mids, target)
