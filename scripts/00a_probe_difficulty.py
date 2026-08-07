"""Script 00a: Probe single-solve pass rate; select problems with headroom.

The estimator needs p_factual in the mid-range. At ceiling (p~1.0) no message
can help; at floor (p~0.0) ablations can't hurt. Only problems the teacher
solves SOMETIMES give delta room to move. One API request per problem (n=k).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backends.api import ApiBackend
from src.debate.prompts import get_solve_prompt
from eval.grade import is_correct


async def probe(problems, backend, k, temperature):
    sem = asyncio.Semaphore(16)

    async def one(p):
        async with sem:
            msgs = [{"role": "user", "content": get_solve_prompt(p["question"])}]
            samples = await backend.generate(msgs, n=k, temperature=temperature)
            hits = sum(1 for s in samples if is_correct(s, p["gold"]))
            p["pass_rate"] = hits / len(samples) if samples else 0.0
            return p

    return await asyncio.gather(*[one(dict(p)) for p in problems])


def keep(p, lo, hi):
    return lo <= p["pass_rate"] <= hi


def main():
    ap = argparse.ArgumentParser(description="Select problems by measured pass rate")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--api-url", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default="deepseek-v3.2")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--keep-min", type=float, default=0.1)
    ap.add_argument("--keep-max", type=float, default=0.9)
    ap.add_argument("--max-problems", type=int, default=100)
    ap.add_argument("--cache-path", default="cache_probe.jsonl")
    args = ap.parse_args()

    problems = json.load(open(args.input))
    backend = ApiBackend(
        base_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
        cache_path=args.cache_path,
    )
    out = asyncio.run(probe(problems, backend, args.k, 0.7))

    dist = {"floor(0)": 0, "mid": 0, "ceiling(1)": 0}
    for p in out:
        dist[
            "floor(0)"
            if p["pass_rate"] == 0
            else "ceiling(1)"
            if p["pass_rate"] == 1
            else "mid"
        ] += 1
    print(f"pass-rate distribution over {len(out)} problems: {dist}")

    kept = [p for p in out if keep(p, args.keep_min, args.keep_max)]
    kept.sort(key=lambda p: abs(p["pass_rate"] - 0.5))
    kept = kept[: args.max_problems]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(kept, open(out_path, "w"), indent=2)
    print(f"kept {len(kept)} -> {out_path}")
    print(f"first pass_rates: {[round(p['pass_rate'], 3) for p in kept[:20]]}")


if __name__ == "__main__":
    main()
