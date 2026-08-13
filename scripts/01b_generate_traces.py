#!/usr/bin/env python3
"""
Generate debate traces with HETEROGENEOUS agents and MULTI-KEY rotation.

Replaces scripts/01_generate_debates.py for the v3 run.

Why a new file instead of patching the old one:
  - the old script hardcodes one ApiBackend and one model
  - the old script imports src.debate.harness (self-refinement, not debate)
Keep the old file for the protocol comparison; use this one for the real run.

KEY DESIGN POINT
----------------
solver and verifier stay on deepseek-v3.2 so that data/probed_all.json and the
probe cache (= Arm A) remain valid. Only the CRITIC changes model. A critic
that is a different model from the solver does not share the solver's blind
spots, which is the whole reason the old critic had recall 0.10.

USAGE
-----
  python scripts/01b_generate_traces.py \\
    --problems data/probed_all.json \\
    --output data/traces_v3.jsonl \\
    --solver-model deepseek-v3.2 \\
    --critic-model gpt-oss-120b \\
    --verifier-model deepseek-v3.2 \\
    --n-solutions 3 --max-rounds 3 --max-tokens 1024 \\
    --cache-path /content/drive/MyDrive/cmd/v3/cache_traces.jsonl

Credentials come from the environment only. Never pass a key on the command
line: Colab echoes cell commands into the notebook output, and notebooks get
committed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.backends.multikey import build_role_backends  # noqa: E402
from src.debate.harness_debate import DebateHarness, critic_flag_rate  # noqa: E402


def load_problems(path: str, limit: int = 0, band: str = "all",
                  min_p: float = -1.0, max_p: float = 2.0,
                  seed: int = 0) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        raw = raw.get("problems", list(raw.values()))

    out = []
    for i, p in enumerate(raw):
        pr = p.get("pass_rate", p.get("p", None))
        pr = float(pr) if pr is not None else None
        if pr is not None:
            if band == "headroom" and not (0.0 < pr < 1.0):
                continue
            if band == "nonceiling" and pr >= 1.0:
                continue
            if not (min_p <= pr <= max_p):
                continue
        out.append({
            "pid": p.get("pid") or p.get("id") or str(i),
            "question": p.get("question") or p.get("problem") or "",
            "gold": p.get("gold") or p.get("answer") or "",
            "pass_rate": pr,
        })
    out = [p for p in out if p["question"] and p["gold"]]
    if limit and limit < len(out):
        random.Random(seed).shuffle(out)
        out = out[:limit]
    return out


def already_done(path: str) -> set[str]:
    """pids already present in the output file, so a disconnect is resumable."""
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["pid"])
            except Exception:  # noqa: BLE001
                continue
    return done


async def main_async(a: argparse.Namespace) -> int:
    problems = load_problems(a.problems, a.limit, a.band, a.min_p, a.max_p, a.seed)
    print("problems loaded : %d" % len(problems))

    if a.resume:
        done = already_done(a.output)
        if done:
            problems = [p for p in problems if p["pid"] not in done]
            print("already in output: %d -> %d remaining" % (len(done), len(problems)))
    if not problems:
        print("nothing to do")
        return 0

    role_models = {
        "solver": a.solver_model,
        "critic": a.critic_model or a.solver_model,
        "verifier": a.verifier_model or a.solver_model,
    }
    print("roles           : %s" % role_models)
    if role_models["critic"] == role_models["solver"]:
        print("WARN: critic and solver are the SAME model. This is "
              "self-critique, not multi-agent debate. Measured critic recall "
              "in that configuration was 0.10.")

    backends = build_role_backends(
        role_models,
        base_url=a.api_url,
        cache_path=a.cache_path,
        api_keys_env=a.api_keys_env,
        concurrency_per_key=a.concurrency_per_key,
        max_tokens=a.max_tokens,
        extra_body=json.loads(a.extra_body) if a.extra_body else None,
    )

    harness = DebateHarness(
        backend=backends["solver"],
        role_backends=backends,
        max_rounds=a.max_rounds,
        temperature=a.temperature,
        max_tokens=a.max_tokens,
        critic_persona=a.critic_persona,
    )

    pathlib.Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (a.resume and os.path.exists(a.output)) else "w"
    fh = open(a.output, mode, encoding="utf-8")

    t0 = time.time()
    written = 0
    chunk = max(1, a.chunk)
    try:
        for i in range(0, len(problems), chunk):
            part = problems[i:i + chunk]
            traces = await harness.run_parallel(
                part, n_solutions=a.n_solutions, concurrency=a.problem_concurrency
            )
            for tr in traces:
                fh.write(json.dumps(tr.to_dict(), ensure_ascii=False) + "\n")
                written += 1
            fh.flush()
            el = time.time() - t0
            seen = min(i + chunk, len(problems))
            rate = seen / el if el else 0.0
            eta = (len(problems) - seen) / rate if rate else 0.0
            print("[%d/%d problems] %d traces | %.0fs elapsed | ETA %.0f min"
                  % (seen, len(problems), written, el, eta / 60.0), flush=True)
            if traces:
                print("    critic dispute rate so far: %.3f"
                      % critic_flag_rate(traces), flush=True)
    finally:
        fh.close()
        for be in set(backends.values()):
            try:
                be.close()
            except Exception:  # noqa: BLE001
                pass

    print("\nwrote %d traces -> %s in %.1f min"
          % (written, a.output, (time.time() - t0) / 60.0))
    print("NEXT: scripts/00g_diagnose_signal.py BEFORE building any dataset.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problems", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--api-url", default=os.environ.get(
        "API_URL", "https://api.generalcompute.com/v1"))
    p.add_argument("--api-keys-env", default="GC_API_KEYS",
                   help="env var holding comma separated API keys")
    p.add_argument("--solver-model", default="deepseek-v3.2")
    p.add_argument("--critic-model", default=None,
                   help="different model = real multi-agent. e.g. gpt-oss-120b")
    p.add_argument("--verifier-model", default=None)
    p.add_argument("--critic-persona", default="adversarial",
                   choices=["default", "adversarial"])
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--n-solutions", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--cache-path", default="cache_traces_v3.jsonl")
    p.add_argument("--concurrency-per-key", type=int, default=8)
    p.add_argument("--problem-concurrency", type=int, default=16)
    p.add_argument("--chunk", type=int, default=25,
                   help="flush to disk every N problems")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--band", default="all",
                   choices=["all", "headroom", "nonceiling"])
    p.add_argument("--min-p", type=float, default=-1.0)
    p.add_argument("--max-p", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extra-body", default=None,
                   help='JSON, e.g. \'{"thinking": {"type": "disabled"}}\'')
    p.add_argument("--resume", action="store_true",
                   help="skip pids already present in --output and append")
    return p


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(build_parser().parse_args())))
