"""
Debate harness for running the RCR loop.

Own implementation of the debate loop that builds the DAG structure
for downstream counterfactual analysis.
"""

from typing import Any

from ..backends.base import Backend
from .prompts import get_critique_prompt, get_revision_prompt, get_solve_prompt
from .schema import Message, Trace

try:
    from eval.grade import extract_answer, is_correct
except ImportError:
    from ...eval.grade import extract_answer, is_correct


class DebateHarness:
    """
    Harness for running the RCR (Reason to Criticize and Revise) debate loop.

    This implementation builds a complete DAG of the debate process,
    enabling downstream counterfactual analysis.
    """

    def __init__(
        self,
        backend: Backend,
        max_rounds: int = 3,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ):
        """
        Initialize the debate harness.

        Args:
            backend: Backend for model inference.
            max_rounds: Maximum number of debate rounds.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens per generation.
        """
        self.backend = backend
        self.max_rounds = max_rounds
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def run(self, pid: str, question: str, gold: str, n_solutions: int = 1) -> list[Trace]:
        """
        Run the debate loop for a problem.

        Args:
            pid: Problem ID.
            question: The problem statement.
            gold: Ground truth answer.
            n_solutions: Number of independent debate traces to generate.

        Returns:
            List of debate traces.
        """
        import asyncio
        tasks = [self._run_single_trace(pid, question, gold, i) for i in range(n_solutions)]
        return await asyncio.gather(*tasks)

    async def _run_single_trace(self, pid: str, question: str, gold: str, solution_index: int) -> Trace:
        """Run a single debate trace for a problem."""
        trace_id = f"{pid}:s{solution_index}"
        trace = Trace(pid=pid, trace_id=trace_id, question=question, gold=gold, topology="solver_critic_verifier")

        # Round 1: Initial solution
        solve_prompt = get_solve_prompt(question)
        messages = [{"role": "user", "content": solve_prompt}]
        solutions = await self.backend.generate(messages, n=1, temperature=self.temperature, max_tokens=self.max_tokens)

        if not solutions:
            return trace

        last_solution = solutions[0]
        solve_msg = Message(
            mid="r1.solver",
            round=1,
            role="solver",
            text=last_solution,
            answer=extract_answer(last_solution),
            parents=[],
        )
        trace.messages.append(solve_msg)

        # Subsequent rounds: critique and revision
        for round_num in range(2, self.max_rounds + 1):
            prev_solver_mid = f"r{round_num-1}.solver"
            
            # Critique round
            critique_prompt = get_critique_prompt(question, last_solution)
            messages = [{"role": "user", "content": critique_prompt}]
            critiques = await self.backend.generate(messages, n=1, temperature=self.temperature, max_tokens=self.max_tokens)

            if critiques:
                critique_msg = Message(
                    mid=f"r{round_num-1}.critic",
                    round=round_num-1,
                    role="critic",
                    text=critiques[0],
                    answer=None,
                    parents=[prev_solver_mid],
                )
                trace.messages.append(critique_msg)

                # Revision round
                revision_prompt = get_revision_prompt(question, last_solution, critiques[0])
                messages = [{"role": "user", "content": revision_prompt}]
                revisions = await self.backend.generate(messages, n=1, temperature=self.temperature, max_tokens=self.max_tokens)

                if revisions:
                    last_solution = revisions[0]
                    revision_msg = Message(
                        mid=f"r{round_num}.solver",
                        round=round_num,
                        role="solver",
                        text=last_solution,
                        answer=extract_answer(last_solution),
                        parents=[prev_solver_mid, critique_msg.mid],
                    )
                    trace.messages.append(revision_msg)
                else:
                    break
            else:
                break

        # Terminal verifier node (required for direct-effect estimation)
        from src.counterfactual.replay import render_verifier_messages
        messages = render_verifier_messages(trace, exclude=())
        
        verifiers = await self.backend.generate(messages, n=1, temperature=self.temperature, max_tokens=self.max_tokens)
        
        if verifiers:
            final_text = verifiers[0]
        else:
            final_text = f"The final answer is \\boxed{{{extract_answer(last_solution) or last_solution}}}."

        # The verifier sees all previous messages
        active_messages = [m.mid for m in trace.messages]
        
        verifier_msg = Message(
            mid=f"r{self.max_rounds+1}.verifier",
            round=self.max_rounds+1,
            role="verifier",
            text=final_text,
            answer=extract_answer(final_text),
            parents=active_messages,
        )
        trace.messages.append(verifier_msg)

        # Set final answer
        trace.final_answer = verifier_msg.answer or extract_answer(last_solution)
        trace.final_correct = is_correct(trace.final_answer, gold)
        return trace

    async def run_parallel(
        self,
        problems: list[dict],
        n_solutions_per_problem: int = 1,
        progress_callback: Any = None,
    ) -> list[Trace]:
        """
        Run the debate loop for multiple problems in parallel.

        Args:
            problems: List of problem dicts with 'pid', 'question', 'gold'.
            n_solutions_per_problem: Number of traces per problem.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of all debate traces.
        """
        import asyncio
        all_traces = []
        total = len(problems) * n_solutions_per_problem
        completed = 0

        async def _process(problem):
            nonlocal completed
            try:
                traces = await self.run(
                    problem["pid"], 
                    problem["question"], 
                    problem["gold"], 
                    n_solutions_per_problem
                )
            except Exception as e:
                # One overflowing/failing problem must not kill the whole run.
                print(f"  FAILED {problem['pid']}: {type(e).__name__}: {e}", flush=True)
                traces = []
            completed += n_solutions_per_problem
            if progress_callback:
                progress_callback(completed, total)
            return traces

        tasks = [_process(p) for p in problems]
        results = await asyncio.gather(*tasks)
        for traces in results:
            all_traces.extend(traces)

        return all_traces