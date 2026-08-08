"""
Script 02: Counterfactual replay for utility estimation.

Two estimands are available and they are NOT interchangeable:

  --estimand direct   Controlled direct effect. Removes m from the terminal
                      prompt only, holding every descendant of m fixed at its
                      factual text. On a solver/critic chain this blocks the
                      mediation path that carries essentially all of the
                      signal, so it returns ~0 for every non-terminal message
                      REGARDLESS of problem difficulty. Kept for the ablation
                      table; do not use it to select data.

  --estimand total    Removes m and REGENERATES every descendant. This is the
                      quantity you actually mean by "what did this message
                      contribute". Costs about 3-4x more per trace.

See tests/test_estimand.py for the executable demonstration that these differ
by 0.0 vs 1.0 on a trace where one critique is provably decisive.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backends.api import ApiBackend
from src.counterfactual.replay import trace_utilities
from src.counterfactual.estimands import trace_utilities_total
from src.debate.schema import Trace


def load_traces(path):
    """Accept either a JSON list or JSONL, since 01 writes jsonl."""
    text = Path(path).read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        return [Trace.from_dict(t) for t in json.loads(text)]
    out = []
    for line in text.splitlines():
        if line.strip():
            out.append(Trace.from_dict(json.loads(line)))
    return out


def main():
    parser = argparse.ArgumentParser(description="Counterfactual replay")
    parser.add_argument("--traces", type=str, required=True, help="Path to traces JSON or JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output path for counterfactual results")
    parser.add_argument("--api-url", type=str, required=True, help="API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--model", type=str, default="deepseek-v4-flash", help="Model name")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of traces to process")
    parser.add_argument("--k", type=int, default=32,
                        help="Counterfactual samples per message. Provider caps n per "
                             "REQUEST at 8; the backend chunks, so 32 and 64 are fine.")
    parser.add_argument("--estimand", choices=["direct", "total"], default="total",
                        help="'direct' is degenerate on chain topologies. See docstring.")
    parser.add_argument("--max-regen-depth", type=int, default=None,
                        help="Cap descendant regeneration depth (total estimand only).")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--trace-concurrency", type=int, default=4,
                        help="Traces processed in parallel.")
    parser.add_argument("--cache-path", type=str, default="cache_cf.jsonl", help="Path to API cache")
    args = parser.parse_args()

    traces = load_traces(args.traces)[: args.sample_size]
    print("loaded %d traces | estimand=%s | k=%d" % (len(traces), args.estimand, args.k))
    if args.estimand == "direct":
        print("NOTE: the direct estimand returns ~0 for every mediated message.")
        print("      That is a property of the estimand, not of your data.")

    backend = ApiBackend(
        base_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        cache_path=args.cache_path,
        supports_n=True,
        extra_body={"thinking": {"type": "disabled"}},  # hidden CoT breaks ablation
    )

    async def run_all():
        sem = asyncio.Semaphore(args.trace_concurrency)
        results = []
        done = [0]

        async def one(trace):
            async with sem:
                if args.estimand == "total":
                    utilities = await trace_utilities_total(
                        trace, backend, k=args.k,
                        temperature=args.temperature,
                        max_regen_depth=args.max_regen_depth,
                    )
                else:
                    utilities = await trace_utilities(trace, backend, k=args.k)
            done[0] += 1
            nz = sum(1 for u in utilities if abs(getattr(u, "delta", 0.0)) > 1e-9)
            print("  [%d/%d] %s  messages=%d  nonzero_delta=%d"
                  % (done[0], len(traces), trace.pid, len(utilities), nz), flush=True)
            return [u.to_dict() for u in utilities]

        chunks = await asyncio.gather(*[one(t) for t in traces])
        for c in chunks:
            results.extend(c)
        return results

    results = asyncio.run(run_all())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    nonzero = sum(1 for r in results if abs(r.get("delta", 0.0)) > 1e-9)
    degenerate = sum(1 for r in results if r.get("degenerate"))
    print("")
    print("computed utilities for %d messages" % len(results))
    print("  nonzero delta : %d  (%.1f%%)"
          % (nonzero, 100.0 * nonzero / max(len(results), 1)))
    print("  flagged degenerate : %d" % degenerate)
    if nonzero == 0:
        print("")
        print("ALL DELTAS ARE ZERO. If estimand=total this is real saturation,")
        print("not a measurement artifact: move to harder problems. If")
        print("estimand=direct, re-run with --estimand total first.")
    print("saved to %s" % output_path)


if __name__ == "__main__":
    main()
