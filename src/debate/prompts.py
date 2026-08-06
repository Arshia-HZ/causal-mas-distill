"""
Project-authored prompts for an RCR-style deliberation protocol.

Conceptually inspired by reason–critique–revise methods.
No third-party prompt text is vendored.

Prompt version: rcr_v2
"""

RCR_SYSTEM_PROMPT = """You are a helpful AI assistant engaging in a critical thinking exercise.

Your task is to either generate a thorough critique of a given solution OR provide a revised solution based on feedback.

Follow these guidelines:
1. Be specific and point out concrete errors or weaknesses
2. Provide actionable suggestions for improvement
3. If revising, incorporate the feedback while maintaining what was correct
4. Be rigorous but constructive
"""

RCR_CRITIQUE_PROMPT = """You are helping with a problem. Here is the original problem:

{problem}

Below is a proposed solution:

{previous_solution}

Now, critically analyze this solution. Identify:
1. Any factual errors or incorrect reasoning
2. Missing steps or incomplete reasoning
3. Areas that could be improved or optimized
4. Any assumptions that may not be valid

Provide a clear, structured critique that will help improve the solution.
"""

RCR_REVISION_PROMPT = """You are helping with a problem. Here is the original problem:

{problem}

Here is a proposed solution:
{previous_solution}

Here is feedback/critique on that solution:
{critique}

Based on this feedback, provide an improved version of the solution.
Address all the points raised in the critique while maintaining what was correct in the original.
"""

RCR_SOLVE_PROMPT = """Solve the following problem step by step. Show your reasoning:

{problem}
"""


def get_critique_prompt(problem: str, previous_solution: str) -> str:
    """Get the critique prompt with the problem and previous solution filled in."""
    return RCR_CRITIQUE_PROMPT.format(
        problem=problem, 
        previous_solution=previous_solution
    )

def get_revision_prompt(problem: str, previous_solution: str, critique: str) -> str:
    """Get the revision prompt with all fields filled in."""
    return RCR_REVISION_PROMPT.format(
        problem=problem,
        previous_solution=previous_solution,
        critique=critique
    )

def get_solve_prompt(problem: str) -> str:
    """Get the initial solve prompt with the problem filled in."""
    return RCR_SOLVE_PROMPT.format(problem=problem)