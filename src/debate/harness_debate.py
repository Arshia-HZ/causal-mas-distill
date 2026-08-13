"""
A real multi-agent debate harness.

WHY THIS FILE EXISTS
--------------------
The original src/debate/harness.py issues every non-terminal call as a single
stateless user turn:

    messages = [{"role": "user", "content": critique_prompt}]

No system prompt, no role identity, no transcript. The critic sees only the
immediately preceding solution. That protocol is sequential self-refinement,
not debate, and self-refinement is already known to fail on math reasoning
(Huang et al., ICLR 2024). Measured on 399 traces it produced a critic flag
rate of 8% and a recall of 0.10 on genuinely wrong solutions.

Keep the original file. Rename it harness_selfrefine.py and select between the
two with a --protocol flag. The protocol comparison at matched budget is a
result in its own right.

WHAT CHANGED
------------
1. Role system prompts are prepended to every call (ROLE_SYSTEM in prompts.py).
2. The critic and the reviser both receive the FULL running transcript.
3. Every generation carries a unique cache_nonce, so two seeds that happen to
   emit identical text no longer collapse onto one cached downstream draw.
4. No seed padding. The original did seeds.append(seeds[len(seeds) % distinct])
   when the provider returned fewer than n completions, manufacturing duplicate
   seeds. This version returns fewer traces and warns.
5. critic_persona="adversarial" swaps in a stronger critic system prompt so you
   can run role conditioning as an explicit ablation arm.

CONTEXT BUDGET WARNING
----------------------
The teacher has an 8192-token TOTAL context (prompt + completion). Passing the
full transcript grows the prompt every round. _fit_transcript enforces a
character budget by dropping the middle of over-long individual messages. Keep
the critic capped at 200 words (prompts.py v3 does this) or you will hit the
limit on round 3.
"""

from __future__ import annotations

import asyncio

from ..backends.base import Backend
from .prompts import (
    CRITIC_SYSTEM,
    CRITIC_SYSTEM_ADVERSARIAL,
    ROLE_SYSTEM,
    VERIFIER_SYSTEM,
    get_critique_prompt,
    get_revision_prompt,
    get_solve_prompt,
    get_verify_prompt,
    parse_verdict,
)
from .schema import Message, Trace

try:
    from eval.grade import extract_answer, is_correct
except ImportError:
    from ...eval.grade import extract_answer, is_correct


# Total characters of transcript we are willing to put in a prompt. The teacher
# context is 8192 tokens; math text runs near 3 chars/token, so 8192 tokens is
# roughly 24000 chars. Leave room for the instruction block and the completion.
TRANSCRIPT_CHAR_BUDGET = 12000


def _clip(text: str, limit: int) -> str:
    """Drop the middle of an over-long message, keeping head and tail."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return text[:head] + "\n[...truncated...]\n" + text[-tail:]


def render_transcript(messages: list[Message], budget: int = TRANSCRIPT_CHAR_BUDGET) -> str:
    """
    Render the running debate as labelled turns. Roles are visible, so the
    reader knows who said what -- that is the whole point of the rewrite.
    """
    if not messages:
        return "(no messages yet)"
    per_msg = max(400, budget // len(messages))
    parts = []
    for m in messages:
        parts.append("[%s | %s]\n%s" % (m.role.upper(), m.mid, _clip(m.text, per_msg)))
    return "\n\n".join(parts)


class DebateHarness:
    """Solver / Critic / Verifier debate with a shared transcript."""

    def __init__(
        self,
        backend: Backend,
        max_rounds: int = 3,
        temperature: float = 0.7,
        max_tokens: int | None = 1024,
        critic_persona: str = "default",
        role_backends: dict | None = None,
    ):
        self.backend = backend
        # role -> Backend. Missing roles fall back to `backend`.
        # This is what makes the system genuinely multi-agent: the critic can
        # be a DIFFERENT model from the solver, so its errors are not the same
        # errors the solver already made.
        self.role_backends = role_backends or {}
        self.max_rounds = max_rounds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.critic_persona = critic_persona

    def _be(self, role: str):
        return self.role_backends.get(role, self.backend)

    def _system_for(self, role: str) -> str:
        if role == "critic" and self.critic_persona == "adversarial":
            return CRITIC_SYSTEM_ADVERSARIAL
        return ROLE_SYSTEM.get(role, ROLE_SYSTEM["solver"])

    async def _gen(self, role: str, user_prompt: str, nonce: str,
                   max_tokens: int | None = None) -> str:
        """One generation with an explicit role system turn and a unique nonce."""
        messages = [
            {"role": "system", "content": self._system_for(role)},
            {"role": "user", "content": user_prompt},
        ]
        out = await self._be(role).generate(
            messages,
            n=1,
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            cache_nonce=nonce,
        )
        return out[0] if out else ""

    async def run(self, pid: str, question: str, gold: str,
                  n_solutions: int = 1) -> list[Trace]:
        """
        Draw n_solutions independent round-1 solutions in ONE request, then run
        an independent debate on each.

        If the provider returns fewer distinct completions than requested, we
        run fewer traces. We do NOT pad by duplicating a seed: duplicated seeds
        produce per-problem scores that are all-0 or all-N, which is the
        fingerprint of the cache-collapse bug this project already paid for.
        """
        seed_msgs = [
            {"role": "system", "content": self._system_for("solver")},
            {"role": "user", "content": get_solve_prompt(question)},
        ]
        seeds = await self._be("solver").generate(
            seed_msgs,
            n=n_solutions,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            cache_nonce="seed|%s" % pid,
        )
        seeds = [s for s in seeds if s and s.strip()]
        distinct = len({s.strip() for s in seeds})
        if distinct < len(seeds):
            print("WARN %s: %d/%d round-1 solutions are identical; "
                  "keeping distinct ones only" % (pid, len(seeds) - distinct, len(seeds)))
            seen, uniq = set(), []
            for s in seeds:
                if s.strip() not in seen:
                    seen.add(s.strip())
                    uniq.append(s)
            seeds = uniq
        if not seeds:
            print("WARN %s: no round-1 solutions returned" % pid)
            return []

        tasks = [
            self._run_single_trace(pid, question, gold, i, seed)
            for i, seed in enumerate(seeds)
        ]
        return list(await asyncio.gather(*tasks))

    async def _run_single_trace(self, pid: str, question: str, gold: str,
                                solution_index: int, seed_solution: str) -> Trace:
        trace_id = "%s:s%d" % (pid, solution_index)
        trace = Trace(pid=pid, trace_id=trace_id, question=question, gold=gold,
                      topology="debate_solver_critic_verifier")

        # Round 1: the seed solution, already generated.
        mid = "r1.solver"
        trace.messages.append(Message(
            mid=mid, round=1, role="solver", text=seed_solution,
            answer=extract_answer(seed_solution), parents=[],
        ))

        # Alternating critique / revision. Both see the full transcript.
        for rnd in range(1, self.max_rounds):
            transcript = render_transcript(trace.messages)
            parents = [m.mid for m in trace.messages]

            cmid = "r%d.critic" % rnd
            ctext = await self._gen(
                "critic",
                get_critique_prompt(question, transcript=transcript),
                nonce="%s|%s" % (trace_id, cmid),
            )
            trace.messages.append(Message(
                mid=cmid, round=rnd, role="critic", text=ctext,
                answer=extract_answer(ctext), parents=parents,
            ))

            transcript = render_transcript(trace.messages)
            parents = [m.mid for m in trace.messages]

            smid = "r%d.solver" % (rnd + 1)
            stext = await self._gen(
                "solver",
                get_revision_prompt(question, transcript=transcript),
                nonce="%s|%s" % (trace_id, smid),
            )
            trace.messages.append(Message(
                mid=smid, round=rnd + 1, role="solver", text=stext,
                answer=extract_answer(stext), parents=parents,
            ))

        # Verifier reads everything and decides.
        vmid = "r%d.verifier" % (self.max_rounds + 1)
        vparents = [m.mid for m in trace.messages]
        vtext = await self._verify_with_retry(question, trace, vmid)
        trace.messages.append(Message(
            mid=vmid, round=self.max_rounds + 1, role="verifier", text=vtext,
            answer=extract_answer(vtext), parents=vparents,
        ))

        final = extract_answer(vtext)
        if not final:
            # Fall back to the last solver answer rather than losing the trace.
            for m in reversed(trace.messages):
                if m.role == "solver" and m.answer:
                    final = m.answer
                    break
        trace.final_answer = final
        trace.final_correct = bool(final) and is_correct(final, gold)
        return trace

    async def _verify_with_retry(self, question: str, trace: Trace, vmid: str) -> str:
        """
        Shrink the transcript on context-overflow errors instead of losing the
        trace. Two problems were lost to overflow in the previous run.
        """
        overflow = ("max_len", "context", "too long", "maximum context",
                    "only supports", "reduce the length")
        budget = TRANSCRIPT_CHAR_BUDGET
        last_err = None
        for attempt in range(4):
            transcript = render_transcript(trace.messages, budget=budget)
            try:
                return await self._gen(
                    "verifier",
                    get_verify_prompt(question, transcript),
                    nonce="%s|%s|a%d" % (trace.trace_id, vmid, attempt),
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                if not any(t in str(e).lower() for t in overflow):
                    raise
                budget = int(budget * 0.65)
        print("WARN %s: verifier failed after retries: %s" % (trace.trace_id, last_err))
        return ""

    async def run_parallel(self, problems: list[dict], n_solutions: int = 1,
                           concurrency: int = 8) -> list[Trace]:
        sem = asyncio.Semaphore(concurrency)
        all_traces: list[Trace] = []

        async def _process(problem):
            async with sem:
                try:
                    return await self.run(
                        problem["pid"], problem["question"], problem["gold"],
                        n_solutions=n_solutions,
                    )
                except Exception as e:  # noqa: BLE001
                    print("FAILED %s: %s: %s" % (problem["pid"], type(e).__name__, e))
                    return []

        for result in await asyncio.gather(*[_process(p) for p in problems]):
            all_traces.extend(result)
        return all_traces


def critic_flag_rate(traces: list[Trace]) -> float:
    """
    Diagnostic. Fraction of critic messages whose verdict line is DISPUTE.
    The old harness scored 0.08 here. If the rewrite does not move this number,
    the rewrite did not work and you should not spend money regenerating.
    """
    n = d = 0
    for t in traces:
        for m in t.messages:
            if m.role == "critic":
                n += 1
                if parse_verdict(m.text) == "dispute":
                    d += 1
    return d / n if n else 0.0
