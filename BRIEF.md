# Project brief — read this first

You are patching a research codebase for a Master's thesis in AI. You have no
prior context. This document is complete: everything you need is here.

Do not optimise, refactor, or "improve" anything not asked for. This is a
scientific measurement pipeline. Silent behaviour changes destroy results that
cost real money to produce.

---

## 1. The one-sentence hypothesis

**At a strictly matched training-token budget, does supervised fine-tuning on
multi-agent debate transcripts produce a better small model than supervised
fine-tuning on plain correct solutions?**

Arm A = train on correct solutions found by sampling the teacher repeatedly.
Arm B = train on full debate transcripts for the same problems.

If B beats A, deliberation contains transferable signal beyond the final
answer. If B ties or loses, it does not. Both outcomes are publishable; the
experiment must be built so the answer is trustworthy either way.

---

## 2. Vocabulary

- **Teacher**: a large API model, `deepseek-v3.2`, served OpenAI-compatible at
  `https://api.generalcompute.com/v1`. Context limit 8192 tokens TOTAL
  (prompt + completion). Not 32k. This limit has already broken runs.
- **Student**: `Qwen/Qwen2.5-1.5B-Instruct`, trained locally with LoRA on a
  free Colab T4 or Kaggle P100.
- **Trace**: one debate on one problem. Currently 6 messages:
  `r1.solver -> r1.critic -> r2.solver -> r2.critic -> r3.solver -> r4.verifier`.
- **Seed**: an independent debate on the same problem. 3 seeds per problem.
- **Rejection sampling (RS)**: sample the teacher N times, keep completions
  whose final answer matches the gold answer. Also called STaR or RFT. This is
  the standard, strong baseline this thesis must beat.
- **pass rate `p`**: fraction of N independent samples that were correct.

---

## 3. What has already been measured (do not re-derive)

On 785 problems from the MATH dataset, teacher sampled 32 times each:

| result | value |
|---|---|
| solved 32/32 (ceiling) | 629 (80.1%) |
| solved 0/32 (floor) | 22 (2.8%) |
| in between | 134 (17.1%) |

Debates were then run on 133 of those 134, 3 seeds each = **399 traces**.

| measurement | value |
|---|---|
| debate final-answer accuracy | 0.787 |
| critic says "there is an error" | 8% of the time |
| critic recall on genuinely wrong solutions | 0.10 |
| problems the debate solved at least once | 92.5% |
| problems RS solves at the same generation budget | 98.8% |

**Key established fact:** an 18-generation debate (3 seeds x 6 messages)
achieves the answer coverage of roughly **3 independent samples**. The 15
non-seed generations are conditioned on the seed and add almost no diversity.

So the debate is already known to be a WORSE way to FIND answers. The open
question, and the entire point of this experiment, is whether the transcripts
are nonetheless a BETTER way to TEACH.

---

## 4. Traps. Read this section twice.

These are the ways this experiment silently produces a wrong answer.

**T1 — Token matching must use the student tokenizer.**
Never `len(text) // 4`. Load `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")`
and count real tokens. LaTeX tokenises near 3 chars/token, so the heuristic
is off by ~25% and it is off by a DIFFERENT amount for each arm, because
critic messages have different character statistics than solutions.

**T2 — Match on loss-bearing tokens, not total tokens.**
SFT computes loss only on the completion, not the prompt. Match the arms on
the number of **completion tokens**. If you match total tokens, Arm B looks
artificially small because its prompts are longer.

**T3 — The problem sets must be identical.**
Use only problems where BOTH arms have at least one correct example. If Arm A
covers 130 problems and Arm B covers 123, and you compare them, you are
measuring coverage, not teaching quality. That question is already answered
(RS wins). Intersect the sets first.

**T4 — Matched tokens implies unmatched example counts.**
A debate transcript is roughly 6x longer than a solution. At equal completion
tokens, Arm A will have ~6x more training examples than Arm B. That is the
honest primary comparison, but it confounds "content" with "number of gradient
updates". So run BOTH:
  - **Primary: matched completion tokens** (unequal example counts)
  - **Secondary: matched example counts** (unequal tokens)
If the two disagree, report both. Do not pick the flattering one.

**T5 — Three training seeds minimum.**
LoRA SFT on ~1000 examples has run-to-run standard deviation of 1-2 accuracy
points. A single-seed 2-point difference is noise. Train each arm with seeds
0, 1, 2 and report mean with a bootstrap confidence interval.

**T6 — The supervised target is the TEACHER text, never the gold label.**
If you train on the gold answer string, every arm collapses to answer-only SFT
and the contrast you are measuring disappears.

**T7 — Deduplicate near-identical problems.**
213 of the 399 traces come from problems the teacher solves 32/32. Those
traces are near-duplicates of each other and will dominate any budget. Report
results split by problem difficulty as well as pooled.

**T8 — Held-out evaluation only.**
The evaluation set must contain no problem used to build either training set.

**T9 — Never call `updatePage`-style destructive rewrites on cached data.**
The API disk cache keys on the exact request payload. Changing `max_tokens`,
the model name, or a prompt string invalidates every cached entry and silently
costs money on the next run. If you change a prompt, say so loudly.

---

## 5. The experiment, precisely

```
PHASE 0 (free, uses only data already on disk)
  build Arm A from the existing probe cache
  build Arm B from the existing 399 traces
  intersect problems, match completion tokens
  train 2 arms x 3 seeds, evaluate
  -> first read on the hypothesis, zero API spend

PHASE 1 (only if Phase 0 is inconclusive or negative)
  the current harness is NOT a debate (see section 7)
  fix it, regenerate traces, re-run Phase 0

PHASE 2 (only if Arm B wins)
  causal attribution: which messages carry the advantage
```

Do Phase 0 first. It costs nothing and may answer the question.

---

## 6. Repo map

```
src/backends/api.py        OpenAI-compatible client with a JSONL disk cache
src/debate/prompts.py      prompt strings
src/debate/harness.py      runs the debate loop, builds the trace DAG
src/debate/schema.py       Message and Trace dataclasses
src/counterfactual/replay.py     leave-one-out ablation (direct effect)
src/counterfactual/estimands.py  total effect with descendant regeneration
src/selection/*.py         message selectors for a later experiment
src/distill/sft.py         LoRA SFT wrapper over trl
eval/grade.py              answer extraction and exact-match grading
scripts/00a..00i           data preparation and diagnostics
scripts/01_generate_debates.py   produces traces.jsonl
scripts/04_train.py        trains the student
scripts/05_evaluate.py     evaluates the student
```

Key data structures in `src/debate/schema.py`:

```python
Message(mid, round, role, text, answer=None, parents=[])
Trace(pid, trace_id, question, gold, messages=[],
      final_answer=None, final_correct=False, topology="...")
```

`eval/grade.py` exposes `extract_answer(text)` and `is_correct(pred, gold)`.
Use them. Do not write a new answer matcher; a hand-rolled one already caused
a false conclusion in this project by mis-grading `2,\!880` against `2880`.

---

## 7. Why the current harness is not a debate (context for Phase 1)

Every non-terminal call in `harness.py` currently looks like this:

```python
messages = [{"role": "user", "content": critique_prompt}]
out = await backend.generate(messages, n=1, ...)
```

A single stateless user turn. Consequences:

- `RCR_SYSTEM_PROMPT` exists in `prompts.py` but is never imported. **No agent
  is ever told it is a solver, a critic, or a verifier.** The roles exist only
  as substrings in the `mid` field.
- The critic sees only the immediately preceding solution. Not the question
  history, not the previous critique, not the sibling seeds.
- No participant except the verifier ever sees the transcript.

So the architecture is: sample, self-critique, self-revise, self-critique,
self-revise, summarise. That is **sequential self-refinement**, not debate.
Self-refinement is already known to fail on reasoning tasks (Huang et al.,
"Large Language Models Cannot Self-Correct Reasoning Yet", ICLR 2024). The
measured 8% flag rate and 0.10 critic recall are a faithful replication of
that known result, not a new finding.

Phase 1 fixes this. Phase 0 does not need it.

---

## 8. ROADMAP — do these in order

Each task lists: what to create, the exact contract, and the acceptance test.
Do not proceed to the next task until the acceptance test passes.

### TASK 1 — `scripts/10_build_arm_a.py` (Arm A source data, zero API cost)

The teacher was already sampled 32 times per problem during difficulty
probing. Those completions are sitting in the JSONL disk cache. They ARE
rejection samples. Recover them instead of paying to regenerate.

Cache format: one JSON object per line, `{"k": "<32-hex>", "v": ["completion", ...]}`.
The key is `sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False))[:32]`
where payload is:

```python
{"m": messages, "n": take, "t": temperature, "mt": max_tokens,
 "mo": model, "c": cache_nonce}
```

Read `src/backends/api.py` for the authoritative `key_of` and chunking logic
(requests larger than `max_n_per_request=8` are split into chunks and the
nonce becomes `f"{cache_nonce or ''}|chunk{ci}"`). Reconstruct the payload for
each problem exactly as `scripts/00a_probe_difficulty.py` built it, hash it,
look up the cache, and collect the completions.

`scripts/00e_recover_probe_offline.py` already does this reconstruction and
reported `recovered 785/785 (100.0%)`. **Read it and reuse its logic.**

CLI:
```
--cache-path PATH     (required) the probe cache jsonl
--problems PATH       (required) probed_all.json
--model STR           default deepseek-v3.2
--max-tokens INT      default 768   # MUST match what the probe actually used
--temperature FLOAT   default 0.7
--n-probe INT         default 32
--out PATH            default data/arm_a_pool.jsonl
```

Output, one JSON object per line:
```json
{"pid": "math_0093", "question": "...", "gold": "...",
 "solutions": ["<full completion text>", ...]}
```
where `solutions` contains ONLY completions for which
`eval.grade.is_correct(text, gold)` is true.

**Acceptance test:** prints recovered/total and the number of problems with at
least one correct solution. If recovery is below 90%, stop and report; do not
silently fall back to calling the API.

### TASK 2 — `scripts/11_build_ab_datasets.py` (the matched datasets)

This is the scientific core. Get it exactly right.

Inputs: `data/arm_a_pool.jsonl` from Task 1, `data/traces.jsonl` (a JSON array
of 399 trace dicts), and the student tokenizer.

Steps, in order:

1. Load both. Build `pids_a` = problems with >=1 correct solution.
   Build `pids_b` = problems with >=1 trace where `final_correct` is true.
2. `pids = sorted(pids_a & pids_b)`. Print the three set sizes. **This
   intersection is trap T3.**
3. Split `pids` into train and eval by hashing the pid for determinism, e.g.
   eval if `int(hashlib.md5(pid.encode()).hexdigest(), 16) % 5 == 0`. Roughly
   80/20. Print both sizes. Eval pids are excluded from both training sets.
4. Build candidate examples.
   - Arm A example: prompt = the question rendered through the student chat
     template; completion = one correct teacher solution verbatim.
   - Arm B example: prompt = the same question, same template; completion =
     the full debate transcript from a trace whose `final_correct` is true,
     rendered as described below.
5. Arm B transcript rendering. Concatenate messages in trace order:
   ```
   [solver]
   <text>

   [critic]
   <text>

   ...

   [final answer]
   <verifier text>
   ```
   Skip messages whose text is empty. Do not include the gold answer (trap T6).
6. Count completion tokens with the real tokenizer:
   ```python
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
   n = len(tok(text, add_special_tokens=False)["input_ids"])
   ```
   **Never estimate with `len(text)//4` (trap T1).**
7. Matched-token budget. Let `budget` be a CLI argument, default 400000
   completion tokens. For each arm independently, iterate problems in a fixed
   shuffled order (seeded), take one example per problem round-robin, and stop
   when the budget is reached. Round-robin matters: it keeps problem coverage
   as equal as possible between arms instead of letting a few long traces eat
   the budget.
8. Emit `data/sft_arm_a.jsonl` and `data/sft_arm_b.jsonl`, each line
   `{"pid":..., "prompt":..., "completion":...}`.
9. Also emit the matched-EXAMPLE-COUNT variant (trap T4) as
   `data/sft_arm_a_eqn.jsonl` / `data/sft_arm_b_eqn.jsonl`, where both arms
   get `min(len(a), len(b))` examples.
10. Emit `data/eval_problems.json` with the held-out pids, questions, golds.
11. Print a summary table: arm, n_examples, total completion tokens, mean
    completion tokens, n_problems_covered. **The two token totals must be
    within 1% of each other. Assert this and fail loudly if not.**

### TASK 3 — training

Use the existing `scripts/04_train.py` (it already accepts `--seed` and writes
to `<output-dir>/seed<N>`). If it does not read the jsonl format above, adapt
the loader only — do not change the training hyperparameters.

Hyperparameters are in `configs/student_qwen2.5_1.5b.yaml`: Qwen2.5-1.5B-Instruct,
LoRA r=16 alpha=32 dropout=0.05, lr 2e-4, 3 epochs, batch 4, grad-accum 2,
max_seq_length 4096, bf16. **Identical for both arms.** The only thing that
varies is the dataset file.

Loss must be computed on the completion only. `src/distill/sft.py` already
imports `DataCollatorForCompletionOnlyLM` from `trl`. Verify it is actually
used and that the response template matches the chat template you rendered in
Task 2. A mismatch here silently trains on the prompt too and invalidates
everything.

Run 6 jobs:
```
for arm in a b; do for seed in 0 1 2; do
  python scripts/04_train.py --dataset data/sft_arm_${arm}.jsonl \
    --config configs/student_qwen2.5_1.5b.yaml \
    --output-dir ckpt/arm_${arm} --seed ${seed}
done; done
```
Each run is roughly 40-70 minutes on a free T4. Checkpoint to Google Drive or
the Hugging Face Hub, never to Colab local disk — sessions are wiped.

### TASK 4 — `scripts/12_eval_ab.py`

Load each of the 6 checkpoints, generate on `data/eval_problems.json` with
greedy decoding (`temperature=0`, `do_sample=False`, `max_new_tokens=1024`),
grade with `eval.grade.is_correct`, and report:

```
arm  seed  accuracy
A    0     0.xxx
...

arm  mean   95% CI (paired bootstrap over problems)
A    0.xxx  [.., ..]
B    0.xxx  [.., ..]
delta B-A   0.xxx  [.., ..]
```

`src/analysis/stats.py` already provides `paired_bootstrap_ci(a, b, n_boot=10000,
alpha=0.05, seed=0)`. Use it. Pair on problem id, pooling the three seeds per
arm by averaging each problem's per-seed correctness first.

Also print the same table split by teacher difficulty (`p == 1.0` vs `p < 1.0`
from `probed_all.json`), because 80% of problems are at teacher ceiling and
the pooled number hides the interesting stratum (trap T7).

**Decision rule, commit to it before looking:**
- delta CI entirely above 0 -> deliberation carries transferable signal. Go to Phase 2.
- CI spans 0 and |delta| < 0.02 -> null result. Go to Phase 1 (fix the harness) and repeat.
- CI entirely below 0 -> debate transcripts are worse training data. Report it; that is a real finding against STaR-style assumptions.

---

## 9. PHASE 1 — make the harness a real debate

Only do this if Phase 0 is null or negative. Two files change.

### 9.1 `src/debate/prompts.py`

A corrected version is supplied in `patches/src/debate/prompts.py`. It adds:

- **Per-role system prompts.** Each participant is told who it is.
- **A critic that re-derives before it reviews.** The current critic reads the
  solution and then looks for errors, which anchors it on the solution's
  reasoning — this is why it agrees 92% of the time. The new one solves the
  problem independently FIRST, then compares, then locates the first divergent
  step. It ends with a machine-parseable line:
  `VERDICT: AGREE` or `VERDICT: DISPUTE <one-line reason>`.
- **A 200-word cap on critiques.** Critic messages currently average 6,175
  characters — longer than the solutions they review — which pushes the
  verifier prompt to ~6,500 tokens against an 8,192 total context. Two
  problems already died from context overflow.
- **A reviser that must restate the final answer in `\boxed{}` every round.**
  115 of 1,197 solver messages currently contain no extractable answer, all of
  them in rounds 2 and 3, zero in round 1. The revision prompt never asks for
  one.

**Changing these strings invalidates the entire API cache.** Say so in your
commit message. Use a new cache file, e.g. `cache_debates_v3.jsonl`.

### 9.2 `src/debate/harness.py`

A corrected version is supplied in `patches/src/debate/harness.py`. Changes:

1. **Role system prompts** are prepended to every call.
2. **Shared transcript.** The critic and the reviser both receive the full
   running transcript, not just the previous message. This is the difference
   between debate and self-refinement.
3. **`cache_nonce` on every call.** The disk cache keys on the payload. Two
   seeds that happen to produce identical text currently produce identical
   downstream payloads and the cache collapses them into one. Every generation
   now carries `f"{trace_id}|{mid}"`.
4. **No seed padding.** The old code did
   `seeds.append(seeds[len(seeds) % distinct])` when the provider returned
   fewer than `n` completions, which manufactures duplicate seeds — the exact
   bug that was already paid for once. It now returns fewer traces instead and
   prints a warning.
5. **Optional adversarial persona** via `critic_persona`, so you can run
   role-conditioned debate and plain self-refinement as two arms and report
   both.

**Keep the old harness.** Rename it `harness_selfrefine.py` and expose it
behind a `--protocol {selfrefine,debate}` flag on
`scripts/01_generate_debates.py`. The comparison between the two protocols at
matched budget is itself a result worth reporting.

---

## 10. PHASE 2 — CRN matched resampling (attribution)

Only do this if Arm B wins. It answers: WHICH messages carry the advantage.

### The bug being fixed

`src/counterfactual/estimands.py::trace_utilities_total` currently computes:

```python
p_fact, _, _ = await _p_correct_with_regen(trace, set(), ...)   # dropped = empty
```

With `dropped=set()`, the descendant set is empty, so `order` is empty and
**no intermediate message is regenerated** — only the terminal is resampled.
But `p_abl` drops message `m` AND regenerates every descendant of `m`.

So the reported effect is not "with m minus without m". It is:

> (factual transcript, terminal resampled)
> minus
> (m removed AND r2.critic, r2.solver, r3.solver all freshly resampled)

Resampling alone flips the final answer on 17-30% of traces. That variance
loads entirely onto the ablated arm, and it grows with the number of
descendants. Therefore **early messages will show the largest apparent effect
purely as an artefact**, and the pipeline will "discover" that the first
message matters most. That conclusion would be manufactured by the estimator.

### The fix: matched conditions plus common random numbers

For each message `m` with descendant set `D(m)`, run TWO conditions that
regenerate exactly the same node set:

```
p_keep : regenerate D(m), with m PRESENT in every downstream prompt
p_drop : regenerate D(m), with m ABSENT from every downstream prompt
delta  = p_keep - p_drop
```

Both conditions resample identically, so resampling variance cancels.

**Common random numbers (CRN)** further reduce variance. Run `k` paired
"worlds" indexed `w = 0..k-1`. Within world `w`, the cache nonce for node
`mid` must be **identical across the two conditions**:

```python
nonce = f"crn|{trace_id}|w{w}|{mid}"
```

Note what is NOT in the nonce: the identity of the ablated message, and the
condition label. That is deliberate. With a temperature-sampling backend and
a deterministic disk cache, an identical nonce and an identical payload return
the identical completion. So whenever removing `m` does not actually change a
downstream node's prompt, that node returns the SAME text in both conditions,
and the only difference that survives to the terminal is the one caused by
`m`. The paired difference then has far lower variance than two independent
estimates, which matters enormously at k=32 where the raw standard error is
about 0.125.

The supplied patch is a NEW module, `patches/src/counterfactual/crn.py`. It
does not modify `estimands.py`; it imports `UtilityResult` and
`default_node_prompt` from it and adds:

```python
async def message_utility_crn(trace, backend, mid, k=32, ...) -> UtilityResult
async def trace_utilities_total_crn(trace, backend, k=32, ...) -> list[UtilityResult]
async def placebo_check(trace, backend, k=8) -> dict
```

Switch `scripts/02_counterfactual_replay.py` to call
`trace_utilities_total_crn` when `--estimand total_crn` is passed. Leave the
old code path reachable so the two estimands can be compared on the same
traces; that comparison is the evidence that the old one was biased.

It returns `estimand="total_crn"` and records `p_factual=p_keep`,
`p_ablated=p_drop`, plus a paired standard error computed from the per-world
difference vector rather than from the two marginals:

```python
d = [keep_w - drop_w for w in worlds]        # each element in {-1, 0, 1}
se = stdev(d) / sqrt(k)
```

This paired SE is the correct one and it is typically 2-4x smaller than the
unpaired `sqrt(p(1-p)/k + p(1-p)/k)` the old code used.

### Sanity check you must run before trusting any number

Call `placebo_check(trace, backend)`. It runs the same world twice with no
ablation at all. Under CRN the measured difference must be **exactly 0.0** in
every world, because every prompt is byte-identical and the cache returns
identical text. If it is not exactly zero, the nonces are wrong or the cache
is not being hit, and every attribution number is noise. Run this before
spending a cent on attribution.

---

## 11. Other defects found in review — fix these regardless of phase

| # | File | Defect | Fix |
|---|---|---|---|
| 1 | `scripts/03_build_datasets.py` | Passes the SAME causal-delta dict to every selector. `confidence.py:55` and `prm.py:55` just read that dict, so the "confidence" and "PRM" baselines are the causal selector with a different threshold. | Each selector must compute its own score. Confidence needs token logprobs; PRM needs a real process reward model (`Qwen2.5-Math-PRM-7B` runs 4-bit on a T4). Until fixed you have 2 arms, not 4. |
| 2 | `scripts/03_build_datasets.py` | Calls `selector.select(traces, utilities, budget)` with 3 args, so `ses=None`, so `any(s>0 for s in se_list)` is False, so empirical-Bayes shrinkage **never runs**. | Pass the per-message `se` dict as the 4th argument. |
| 3 | `scripts/03_build_datasets.py` | `selector_class()` takes no arguments, so every `configs/selection/*.yaml` is ignored. | Load the YAML and pass it as kwargs. |
| 4 | `src/counterfactual/replay.py` | `assert_parents_invariant()` only checks that parents exist. Its docstring promises it verifies every non-parent is absent from the rendered prompt. It does not. | Either implement the check against the rendered prompt or delete the function; a no-op guard is worse than none. |
| 5 | `src/selection/base.py` | `_estimate_tokens = len(text)//4` | Use the Qwen tokenizer. |
| 6 | `src/utility/surrogate.py` | Entire file is a stub: `train()` is `pass`, `_model_predict` returns `0.5`, `_heuristic_predict` returns hand-tuned constants. Docstring calls it "A key efficiency contribution". | Delete it. A real ContextCite-style surrogate already exists in `estimands.ablation_surrogate`. |
| 7 | `scripts/02b_noise_floor.py` | `go_decision = positive_count/len(deltas) >= 0.10`, an arbitrary hard-coded rule, computed on `traces[:10]`. | Make the threshold a CLI argument and justify it, or drop the go/no-go entirely. |
| 8 | `scripts/01_generate_debates.py:25`, `scripts/02b_noise_floor.py` | `--model` defaults to `deepseek-v4-flash`, which does not exist on the endpoint. | Default to `deepseek-v3.2`. |
| 9 | `configs/teacher_api_deepseek.yaml` | `max_tokens: 768`, but traces were generated at 1024. `00c_headroom_audit.py` reads this file, so audit arms would silently mismatch the traces. | Set 1024. Corrected file supplied in `patches/configs/`. |
| 10 | `src/debate/prompts.py` | Header says `Prompt version: rcr_v2` but the content is byte-identical to v1. | Fixed by the supplied patch. |
| 11 | `src/render/template.py` | `_ROLE_SLOT` has no entry for `("verifier", 4)`, which is the actual verifier round; it survives only via a fallback. A critic at round 3 would be mapped to the `action` slot, which is wrong. | Key slots on role and position-in-trace, not on the round number. |

---

## 12. Definition of done for Phase 0

- [ ] `data/arm_a_pool.jsonl` exists, recovery rate printed and above 90%
- [ ] `data/sft_arm_a.jsonl` and `data/sft_arm_b.jsonl` exist
- [ ] completion-token totals of the two arms are within 1%
- [ ] both arms cover the identical set of training problems
- [ ] `data/eval_problems.json` shares no pid with either training set
- [ ] 6 checkpoints trained (2 arms x 3 seeds), identical hyperparameters
- [ ] `12_eval_ab.py` prints per-seed accuracies, pooled means, a paired
      bootstrap CI on the difference, and the same table split by teacher
      difficulty

---

## 13. PROMPT VERSION LOCK — read before installing prompts.py

The API disk cache keys on a hash of the exact prompt string. Two consequences:

**A. Installing the v3 `prompts.py` invalidates every cached generation.**
Use a new cache file (`cache_debates_v3.jsonl`). Do not overwrite the old one.

**B. The v1 and v3 function signatures are NOT compatible.**

| | v1 | v3 |
|---|---|---|
| `RCR_SOLVE_PROMPT` placeholder | `{problem}` | `{question}` |
| `get_critique_prompt` | `(problem, previous_solution)` | `(question, transcript="", solution="")` |
| `get_revision_prompt` | `(problem, previous_solution, critique)` | `(question, transcript="", solution="", critique="")` |

The v3 versions accept the old keyword names so most call sites keep working,
but a **positional** call `get_revision_prompt(q, prev, crit)` maps `prev` to
`transcript` and `crit` to `solution`, which is silently wrong.

So do this:

1. Copy the current `src/debate/prompts.py` to `src/debate/prompts_v1.py`
   unchanged.
2. Copy the current `src/debate/harness.py` to `src/debate/harness_selfrefine.py`
   and change its import to `from .prompts_v1 import ...`.
3. Install the v3 `prompts.py` and the new `harness_debate.py`.
4. Give `scripts/01_generate_debates.py` a `--protocol {selfrefine,debate}`
   flag that selects between the two harnesses.

**Most important:** `scripts/10_build_arm_a.py` hardcodes the v1 solve prompt
as `LEGACY_SOLVE_PROMPT` because it must reproduce cache keys written under v1.
Do not refactor that constant into an import. If you do, recovery silently
drops to 0% and it will look like the cache was lost.

---

## 14. What is in patches/

```
patches/
  src/debate/prompts.py             v3. Role system prompts, re-deriving critic,
                                    200-word cap, mandatory boxed answer,
                                    parse_verdict() helper.
  src/debate/harness_debate.py      Real debate. Shared transcript, role system
                                    turns, unique cache_nonce per generation,
                                    no seed padding, adversarial critic option,
                                    critic_flag_rate() diagnostic.
  src/counterfactual/crn.py         Matched-condition CRN estimator plus the
                                    mandatory placebo_check().
  configs/teacher_api_deepseek.yaml max_tokens 768 -> 1024, LOCKED markers.
  scripts/10_build_arm_a.py         Recover rejection samples from the probe
                                    cache. Zero API calls.
  scripts/11_build_ab_datasets.py   Matched-token and matched-example datasets,
                                    real tokenizer, intersected problem sets,
                                    held-out split, 1% assertion.
  scripts/12_eval_ab.py             Greedy eval, 3 seeds per arm, paired
                                    bootstrap, difficulty-stratified table.
```

All files pass `ast.parse`. None of them have been executed against real data
— there is no GPU, no network, and no `transformers` in the environment they
were written in. Treat them as reviewed drafts: read them before running, and
expect to fix import paths.

## 15. Order of operations, one screen

```
# Phase 0 -- free, answers the question
python scripts/10_build_arm_a.py --cache-path <probe cache> \
    --problems data/probed_all.json --out data/arm_a_pool.jsonl
python scripts/11_build_ab_datasets.py --arm-a-pool data/arm_a_pool.jsonl \
    --traces data/traces.jsonl --probed data/probed_all.json \
    --budget-tokens 400000 --outdir data
for arm in a b; do for s in 0 1 2; do
  python scripts/04_train.py --dataset data/sft_arm_${arm}.jsonl \
      --config configs/student_qwen2.5_1.5b.yaml \
      --output-dir ckpt/arm_${arm} --seed ${s}
done; done
python scripts/12_eval_ab.py --eval data/eval_problems.json \
    --arm-a ckpt/arm_a/seed0 ckpt/arm_a/seed1 ckpt/arm_a/seed2 \
    --arm-b ckpt/arm_b/seed0 ckpt/arm_b/seed1 ckpt/arm_b/seed2 \
    --probed data/probed_all.json --out results/ab_eval.json

# repeat the last two steps on sft_arm_{a,b}_eqn.jsonl for the
# matched-example-count variant
```
