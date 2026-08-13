"""
Prompts for the debate harness.

Prompt version: rcr_v3  (2026-08-12)

CHANGES FROM v1/v2 -- READ BEFORE RUNNING
-----------------------------------------
These strings are part of the API cache key. Changing them invalidates every
cached generation. Use a NEW cache file (e.g. cache_debates_v3.jsonl).

1. Role system prompts now exist AND are actually used by harness.py. In v1
   RCR_SYSTEM_PROMPT was defined but never imported, so no agent was ever told
   what role it played.
2. The critic re-derives the answer independently BEFORE reading for errors.
   The v1 critic read the solution first, which anchored it: measured flag rate
   was 8% and recall on genuinely wrong solutions was 0.10.
3. Critiques are capped at ~200 words. v1 critic messages averaged 6175 chars,
   longer than the solutions they reviewed, which pushed the verifier prompt
   toward the 8192-token total context limit. Two problems died from overflow.
4. Every critique ends with a machine-parseable verdict line.
5. The revision prompt now REQUIRES a boxed final answer. In v1, 115 of 1197
   solver messages had no extractable answer, all in rounds 2-3, none in round 1.
"""

PROMPT_VERSION = "rcr_v3"

# --------------------------------------------------------------------------
# System prompts. One per role. harness.py prepends these.
# --------------------------------------------------------------------------

SOLVER_SYSTEM = (
    "You are the Solver in a multi-agent mathematical reasoning team. "
    "Your job is to produce a complete, correct, step-by-step solution. "
    "You will be reviewed by an independent Critic, so make every step "
    "checkable. Always end with the final answer in \\boxed{}."
)

CRITIC_SYSTEM = (
    "You are the Critic in a multi-agent mathematical reasoning team. "
    "You are a separate agent from the Solver and you do not trust its work. "
    "Your job is NOT to agree. Your job is to independently determine the "
    "correct answer and then report precisely where, if anywhere, the Solver "
    "diverged from it. Agreeing when the solution is wrong is the worst "
    "outcome; it is worse than a false alarm. Be terse."
)

CRITIC_SYSTEM_ADVERSARIAL = (
    "You are the Critic in a multi-agent mathematical reasoning team, and you "
    "have been assigned the adversarial role. Assume the Solver has made a "
    "mistake until you have proved to yourself that it has not. You must solve "
    "the problem independently first. Only after your own derivation may you "
    "read the Solver's work. Agreeing with a wrong solution is the worst "
    "outcome. Be terse."
)

VERIFIER_SYSTEM = (
    "You are the Verifier in a multi-agent mathematical reasoning team. "
    "You have read the full transcript of the debate. Your job is to decide "
    "the final answer. Weigh the arguments; do not simply defer to the last "
    "speaker. Output a short justification and then the final answer in "
    "\\boxed{}."
)

# Back-compat alias so any old import site keeps working.
RCR_SYSTEM_PROMPT = SOLVER_SYSTEM

ROLE_SYSTEM = {
    "solver": SOLVER_SYSTEM,
    "critic": CRITIC_SYSTEM,
    "verifier": VERIFIER_SYSTEM,
}

# --------------------------------------------------------------------------
# User prompts
# --------------------------------------------------------------------------

RCR_SOLVE_PROMPT = """Solve the following problem.

Problem:
{question}

Work step by step. Keep the derivation tight; do not restate the problem.
End your response with the final answer in \\boxed{{}}."""

RCR_CRITIQUE_PROMPT = """Problem:
{question}

Transcript so far:
{transcript}

Your task, in this exact order:

STEP 1. Before reading the Solver's work again, derive the answer yourself.
Write at most three lines of your own reasoning and state the answer you get.

STEP 2. Compare your answer to the Solver's most recent answer.

STEP 3. If they differ, identify the FIRST step in the Solver's work where it
diverges from a correct derivation. Quote that step. Say what is wrong with it.
If they agree, state the single step you consider least well justified, so the
Solver knows where to add rigour.

Hard limits:
- At most 200 words total.
- Do not rewrite the full solution.
- Your last line must be exactly one of:
    VERDICT: AGREE
    VERDICT: DISPUTE <one short clause naming the faulty step>"""

RCR_REVISION_PROMPT = """Problem:
{question}

Transcript so far:
{transcript}

The Critic has responded to your solution. Decide whether the objection is
correct.

- If the Critic is right, fix the specific step it identified and carry the
  correction through to the end.
- If the Critic is wrong, say so in one sentence, explain briefly why, and keep
  your answer.

Do not restart from scratch and do not restate the problem.

MANDATORY: your response must end with your complete current final answer in
\\boxed{{}}, even if it is unchanged from before. A response without a boxed
answer is discarded."""

RCR_VERIFY_PROMPT = """Problem:
{question}

Full transcript:
{transcript}

Decide the final answer. In at most four lines, say which position the evidence
supports and why. Then give the final answer in \\boxed{{}}."""

# --------------------------------------------------------------------------
# Accessors
# --------------------------------------------------------------------------


def get_solve_prompt(question: str) -> str:
    return RCR_SOLVE_PROMPT.format(question=question)


def get_critique_prompt(question: str, transcript: str = "", solution: str = "") -> str:
    """
    `transcript` is the running debate transcript. `solution` is accepted only
    for backward compatibility with the v1 call signature; if `transcript` is
    empty it is used instead.
    """
    return RCR_CRITIQUE_PROMPT.format(
        question=question, transcript=transcript or solution
    )


def get_revision_prompt(question: str, transcript: str = "", solution: str = "",
                        critique: str = "") -> str:
    if not transcript:
        transcript = (solution + "\n\n" + critique).strip()
    return RCR_REVISION_PROMPT.format(question=question, transcript=transcript)


def get_verify_prompt(question: str, transcript: str) -> str:
    return RCR_VERIFY_PROMPT.format(question=question, transcript=transcript)


def parse_verdict(critique_text: str) -> str:
    """Return 'agree', 'dispute', or 'unparsed'."""
    for line in reversed((critique_text or "").strip().splitlines()):
        s = line.strip().upper()
        if s.startswith("VERDICT:"):
            if "DISPUTE" in s:
                return "dispute"
            if "AGREE" in s:
                return "agree"
    return "unparsed"
