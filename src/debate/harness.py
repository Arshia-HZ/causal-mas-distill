"""
Debate harness for running the RCR loop.

Own implementation of the debate loop that builds the DAG structure
for downstream counterfactual analysis.
"""

from typing import Any

from ..backends.base import Backend
from .prompts import get_critique_prompt, get_revision_prompt, get_solve_prompt
from .schema import MessageRole, RoundType, Trace


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

    def run(self, problem: str, n_solutions: int = 1) -> list[Trace]:
        """
        Run the debate loop for a problem.

        Args:
            problem: The problem statement.
            n_solutions: Number of independent debate traces to generate.

        Returns:
            List of debate traces.
        """
        traces = []
        for _ in range(n_solutions):
            trace = self._run_single_trace(problem)
            traces.append(trace)
        return traces

    def _run_single_trace(self, problem: str) -> Trace:
        """Run a single debate trace for a problem."""
        trace = Trace(problem=problem)

        # Round 0: Initial solution
        solve_prompt = get_solve_prompt(problem)
        messages = [{"role": "user", "content": solve_prompt}]
        solutions = self.backend.generate(messages, n=1, temperature=self.temperature, max_tokens=self.max_tokens)

        if solutions:
            solve_msg = trace.add_message(
                content=solutions[0],
                role=MessageRole.ASSISTANT,
                round_type=RoundType.SOLVE,
            )
            last_solution = solutions[0]
            last_solution_msg_id = solve_msg.mid
        else:
            return trace

        # Subsequent rounds: critique and revision
        for round_num in range(1, self.max_rounds + 1):
            # Critique round
            critique_prompt = get_critique_prompt(last_solution)
            messages = [{"role": "user", "content": critique_prompt}]
            critiques = self.backend.generate(messages, n=1, temperature=self.temperature, max_tokens=self.max_tokens)

            if critiques:
                critique_msg = trace.add_message(
                    content=critiques[0],
                    role=MessageRole.CRITIQUE,
                    round_type=RoundType.CRITIQUE,
                    parent_id=last_solution_msg_id,
                )

                # Revision round
                revision_prompt = get_revision_prompt(problem, last_solution, critiques[0])
                messages = [{"role": "user", "content": revision_prompt}]
                revisions = self.backend.generate(messages, n=1, temperature=self.temperature, max_tokens=self.max_tokens)

                if revisions:
                    revision_msg = trace.add_message(
                        content=revisions[0],
                        role=MessageRole.REVISION,
                        round_type=RoundType.REVISION,
                        parent_id=critique_msg.mid,
                    )
                    last_solution = revisions[0]
                    last_solution_msg_id = revision_msg.mid
                else:
                    break
            else:
                break

        # Set final answer to last solution
        trace.final_answer = last_solution
        return trace

    def run_parallel(
        self,
        problems: list[str],
        n_solutions_per_problem: int = 1,
        progress_callback: Any = None,
    ) -> list[Trace]:
        """
        Run the debate loop for multiple problems in parallel.

        Args:
            problems: List of problem statements.
            n_solutions_per_problem: Number of traces per problem.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of all debate traces.
        """
        all_traces = []
        total = len(problems) * n_solutions_per_problem
        completed = 0

        for problem in problems:
            for _ in range(n_solutions_per_problem):
                trace = self._run_single_trace(problem)
                all_traces.append(trace)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        return all_traces