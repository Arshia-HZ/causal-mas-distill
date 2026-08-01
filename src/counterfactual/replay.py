"""
Counterfactual replay for causal utility estimation.

Core function: message_utility()
- Ablates a message from the debate
- Regenerates all topologically downstream messages
- Computes utility as the change in outcome probability
"""

import asyncio
import re
from typing import Any

from ..debate.schema import Trace, Message, descendants


def extract_answer(text: str) -> str | None:
    """Extract answer from text using common patterns."""
    # Try to find answer after "Answer:" or "The answer is"
    patterns = [
        r"(?:The )?answer(?:\s+is)?:?\s*(.+)",
        r"####\s*(.+)",
        r"\$\$?\s*(.+?)\s*\$\$?$",
        r"\(([^)]+)\)$",  # Answer in parentheses at end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.strip(), re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    
    return None


def is_correct(pred: str, gold: str) -> bool:
    """
    Check if prediction is correct using math-verify if available.
    
    Falls back to simple string matching.
    """
    # Normalize
    pred_norm = pred.strip().lower()
    gold_norm = gold.strip().lower()
    
    if pred_norm == gold_norm:
        return True
    
    # Try math-verify
    try:
        from math_verify import parse, verify
        return verify(parse(gold), parse(pred))
    except ImportError:
        pass
    
    return False


def build_prompt(trace: Trace, node: str, ctx: dict[str, Message]) -> list[dict]:
    """
    Build prompt for regenerating a message with ablated context.
    
    Args:
        trace: Original trace.
        node: Node ID to regenerate.
        ctx: Context with ablated message removed.
        
    Returns:
        List of message dicts for API call.
    """
    msg = trace.get_message(node)
    if msg is None:
        raise ValueError(f"Unknown node: {node}")
    
    messages = []
    
    # System prompt
    messages.append({
        "role": "system",
        "content": f"You are a {msg.role}. Provide a response to the question."
    })
    
    # User: problem
    messages.append({
        "role": "user",
        "content": trace.question
    })
    
    # Add context messages (excluding ablated message and its downstream)
    for m in trace.messages:
        if m.mid in ctx:
            role = "assistant" if m.role in ("solver", "critic", "verifier") else "user"
            messages.append({
                "role": role,
                "content": m.text
            })
    
    return messages


def final_answer_of(ctx: dict[str, Message]) -> str | None:
    """Get final answer from context."""
    # Find last message with an answer
    for mid in sorted(ctx.keys(), reverse=True):
        msg = ctx[mid]
        if msg.answer:
            return msg.answer
    return None


def replace(msg: Message, **kwargs) -> Message:
    """Create a new Message with updated fields."""
    return Message(
        mid=msg.mid,
        round=kwargs.get("round", msg.round),
        role=kwargs.get("role", msg.role),
        text=kwargs.get("text", msg.text),
        answer=kwargs.get("answer", msg.answer),
        parents=kwargs.get("parents", msg.parents),
    )


async def message_utility(
    trace: Trace,
    mid: str,
    backend: Any,
    k: int = 16,
    temperature: float = 0.7
) -> float:
    """
    Compute causal utility of a message via counterfactual replay.
    
    Args:
        trace: Original debate trace.
        mid: Message ID to ablate.
        backend: API backend for generation.
        k: Number of samples for estimation.
        temperature: Sampling temperature.
        
    Returns:
        Causal utility (Δ) as proportion correct improvement.
    """
    # Get downstream messages
    downstream = descendants(trace, mid)
    
    correct = 0
    for _ in range(k):
        # Build context without the ablated message
        ctx = {
            m.mid: m for m in trace.messages
            if m.mid != mid and m.mid not in downstream
        }
        
        # Regenerate each downstream message in topological order
        downstream_list = sorted(downstream)
        for node_mid in downstream_list:
            node_msg = trace.get_message(node_mid)
            if node_msg is None:
                continue
                
            msgs = build_prompt(trace, node_mid, ctx)
            txt = (await backend.generate(msgs, n=1, temperature=temperature))[0]
            ans = extract_answer(txt)
            
            ctx[node_mid] = replace(node_msg, text=txt, answer=ans)
        
        # Check if counterfactual is correct
        final_ans = final_answer_of(ctx)
        if final_ans and is_correct(final_ans, trace.gold):
            correct += 1
    
    return correct / k


class CounterfactualReplay:
    """Manager for counterfactual replay operations."""
    
    def __init__(self, backend: Any):
        self.backend = backend
    
    async def replay_from(
        self,
        trace: Trace,
        mid: str,
        k: int = 16,
        temperature: float = 0.7
    ) -> Trace:
        """
        Replay a trace with a specific message ablated.
        
        Returns:
            New trace with ablated message and regenerated downstream.
        """
        util = await message_utility(trace, mid, self.backend, k, temperature)
        
        # For now, just return the utility value
        # Full trace reconstruction would require storing all intermediate states
        return util
    
    async def compute_utilities(
        self,
        traces: list[Trace],
        k: int = 16,
        temperature: float = 0.7
    ) -> dict[str, float]:
        """
        Compute utilities for all messages in traces.
        
        Returns:
            Dict mapping mid to utility value.
        """
        utilities = {}
        
        for trace in traces:
            for msg in trace.messages:
                if msg.round > 0:  # Skip root message
                    util = await message_utility(trace, msg.mid, self.backend, k, temperature)
                    utilities[f"{trace.pid}:{msg.mid}"] = util
        
        return utilities