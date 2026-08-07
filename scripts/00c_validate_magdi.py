"""Script 00c: Validate MAGDi-converted traces through selection + rendering.

$0 dry-run of the consumption half of the pipeline. Proves, on real converted
debate data:
  1. every trace passes the terminal-sees-all invariant
  2. each selector returns sane PER-TRACE selections
  3. every arm renders exactly one SFT example per trace
  4. format holds: one 'Answer:' marker, prompt ends with '\\nAnswer:',
     non-empty completion

NOT part of the experiment. Causal deltas are zeros here (no replay on MAGDi),
so the causal arm trivially selects everything; the confidence arm uses real
MAGDi confidence_level scores.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.debate.schema import Trace
from src.counterfactual.replay import assert_terminal_sees_all
from src.selection.causal import CausalSelector
from src.selection.confidence import ConfidenceSelector
from src.selection.last_round_only import LastRoundOnlySelector
from src.selection.random_lenmatched import RandomLenMatchedSelector
from src.render.template import render_for_sft

BUDGET = 100_000


def check_examples(name, examples, n_traces):
    assert len(examples) == n_traces, (
        f"{name}: {len(examples)} examples != {n_traces} traces"
    )
    for ex in examples:
        tid = ex.get("trace_id", ex["pid"])
        assert ex["text"].count("Answer:") == 1, f"{name}/{tid}: bad marker count"
        assert ex["prompt"].endswith("\nAnswer:"), f"{name}/{tid}: bad prompt tail"
        assert ex["completion"].strip(), f"{name}/{tid}: empty completion"
    return sum(len(e["text"]) // 4 for e in examples)


def main():
    p = argparse.ArgumentParser(description="Validate MAGDi traces via selectors + renderer")
    p.add_argument("--traces", required=True)
    p.add_argument("--confidence", required=True)
    args = p.parse_args()
    
    traces = [Trace.from_dict(t) for t in json.load(open(args.traces))]
    conf = {
        (c["trace_id"], c["mid"]): c["confidence"]
        for c in json.load(open(args.confidence))
    }

    # ---- 1. schema invariant on every trace + basic stats
    for t in traces:
        assert_terminal_sees_all(t)
    print(f"[1] terminal-sees-all: PASS on {len(traces)} traces")
    n_msgs = [len(t.messages) for t in traces]
    lens = [len(m.text) for t in traces for m in t.messages]
    print(f"    messages/trace : min={min(n_msgs)} max={max(n_msgs)} "
          f"mean={sum(n_msgs)/len(n_msgs):.1f}")
    print(f"    text chars/msg : min={min(lens)} max={max(lens)} "
          f"mean={sum(lens)/len(lens):.0f}")

    # ---- 2. per-trace selection
    arms = {
        "causal(zeros)": CausalSelector().select(traces, {}, BUDGET),
        "confidence": ConfidenceSelector(min_confidence=0.85).select(traces, conf, BUDGET),
        "last_round_only": LastRoundOnlySelector().select(traces, {}, BUDGET),
        "random_lenmatched": RandomLenMatchedSelector().select(traces, {}, BUDGET),
    }
    for name, sel in arms.items():
        total = sum(len(v) for v in sel.values())
        missing = [t.trace_id for t in traces if t.trace_id not in sel]
        print(f"[2] {name:18s} covered={len(sel)}/{len(traces)} "
              f"msgs={total} missing={len(missing)}")

    # ---- 3+4. render + format
    print("[3] render + format:")
    for name, sel in arms.items():
        ex = render_for_sft(traces, sel)
        toks = check_examples(name, ex, len(traces))
        print(f"    {name:18s} examples={len(ex)} ~tokens={toks} OK")

    # ---- eyeball one rendered example
    ex = render_for_sft(traces, arms["confidence"])
    print("=" * 70)
    print(ex[0]["text"][:800])
    print("=" * 70)
    print("prompt tail:", repr(ex[0]["prompt"][-12:]),
          "| completion:", repr(ex[0]["completion"][:40]))
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
