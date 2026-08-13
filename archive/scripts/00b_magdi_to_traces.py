import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.debate.schema import Trace, Message
from eval.grade import is_correct

AGENTS = ("claude", "gpt4", "bard")

def item_to_trace(item, pid):
    question = (item.get("question") or "").strip()
    gold = str(item.get("gold_answer", "")).strip()
    if not question or not gold:
        return None
    trace = Trace(pid=pid, trace_id=f"{pid}:s0", question=question,
                  gold=gold, topology="magdi_3agent_debate")
    last_round = -1
    for r in range(20):
        if f"{AGENTS[0]}_output_{r}" not in item:
            break
        for agent in AGENTS:
            out = item.get(f"{agent}_output_{r}")
            if not isinstance(out, dict):
                continue
            text = (item.get(f"{agent}_exp_{r}") or out.get("reasoning", "")).strip()
            answer = str(item.get(f"{agent}_pred_{r}", out.get("answer", ""))).strip()
            parents = [] if r == 0 else [m.mid for m in trace.messages if m.round == r]
            trace.messages.append(Message(
                mid=f"r{r+1}.{agent}", round=r+1, role="solver",
                text=text, answer=answer, parents=parents))
        last_round = r
    if last_round < 0 or not trace.messages:
        return None
    final = str(item.get(f"majority_ans_{last_round}", "")).strip()
    trace.messages.append(Message(
        mid=f"r{last_round+2}.verifier", round=last_round+2, role="verifier",
        text=f"The final answer is \\boxed{{{final}}}.",
        answer=final, parents=[m.mid for m in trace.messages]))
    trace.final_answer = final
    trace.final_correct = is_correct(final, gold)
    return trace


def main():
    import argparse
    p = argparse.ArgumentParser(description="Convert MAGDi MAG JSON to CMD traces")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--confidence-out", default=None,
                   help="Also write real per-message confidence scores")
    args = p.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    if args.max_items:
        data = data[: args.max_items]

    traces, conf, skipped = [], [], 0
    for i, item in enumerate(data):
        pid = f"{args.dataset}_{i:04d}"
        t = item_to_trace(item, pid)
        if t is None:
            skipped += 1
            continue
        traces.append(t.to_dict())
        for m in t.messages:
            if m.role != "solver":
                continue
            agent = m.mid.split(".")[1]
            out = item.get(f"{agent}_output_{m.round - 1}", {})
            conf.append({
                "trace_id": t.trace_id, "pid": t.pid, "mid": m.mid,
                "delta": 0.0,  # placeholder so script 03 can read the file
                "confidence": float(out.get("confidence_level", 0.0)),
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(traces, f, indent=2)
    n_correct = sum(1 for t in traces if t["final_correct"])
    print(f"converted {len(traces)} traces (skipped {skipped}) -> {out}")
    print(f"majority-vote accuracy: {n_correct}/{len(traces)}")

    if args.confidence_out:
        cp = Path(args.confidence_out)
        cp.parent.mkdir(parents=True, exist_ok=True)
        with open(cp, "w") as f:
            json.dump(conf, f, indent=2)
        print(f"wrote {len(conf)} confidence scores -> {cp}")

if __name__ == "__main__":
    main()