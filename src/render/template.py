"""
Unified template renderer for all selectors.

Every selector goes through this template. No exceptions.
This eliminates the format confound structurally.
"""

from ..debate.schema import Trace, Message


# Universal skeleton template - one template for ALL selectors
SKEL = """Question: {q}

<hypothesis>{hypothesis}</hypothesis>
<disagreement>{disagreement}</disagreement>
<error>{error}</error>
<correction>{correction}</correction>
<verify>{verify}</verify>
<action>{action}</action>

Answer: {answer}"""


def assign_slots(trace: Trace, selected_mids: list[str]) -> dict[str, str]:
    """
    Assign selected messages to slots in the template.
    
    Slots: hypothesis, disagreement, error, correction, verify, action
    
    Role-to-slot mapping for solver_critic_verifier topology:
    - solver.hypothesis -> hypothesis
    - critic.disagreement -> disagreement
    - solver.error -> error
    - critic.correction -> correction
    - verifier.verify -> verify
    - solver.action -> action
    
    Args:
        trace: Debate trace.
        selected_mids: List of message IDs to include.
        
    Returns:
        Dict mapping slot names to content (empty string if unfilled).
    """
    # Default empty slots
    slots = {
        "hypothesis": "",
        "disagreement": "",
        "error": "",
        "correction": "",
        "verify": "",
        "action": "",
        "q": trace.question,
        "answer": trace.gold,
    }
    
    # Build message lookup
    mid_to_msg = {m.mid: m for m in trace.messages}
    
    for mid in selected_mids:
        msg = mid_to_msg.get(mid)
        if msg is None:
            continue
            
        # Map role+round to slot
        if msg.role == "solver":
            if msg.round == 1:
                slots["hypothesis"] = msg.text
            elif msg.round == 2:
                slots["error"] = msg.text
            elif msg.round == 3:
                slots["action"] = msg.text
        elif msg.role == "critic":
            if msg.round == 1:
                slots["disagreement"] = msg.text
            elif msg.round == 2:
                slots["correction"] = msg.text
        elif msg.role == "verifier":
            slots["verify"] = msg.text
    
    return slots


def render(trace: Trace, selected_mids: list[str]) -> str:
    """
    Render a trace into the unified template format.
    
    Args:
        trace: Debate trace.
        selected_mids: List of message IDs to include.
        
    Returns:
        Formatted string using the universal skeleton.
    """
    slots = assign_slots(trace, selected_mids)
    return SKEL.format(**slots)


def render_batch(traces: list[Trace], selected_mids_list: list[list[str]]) -> list[str]:
    """
    Render multiple traces.
    
    Args:
        traces: List of debate traces.
        selected_mids_list: List of message ID lists (one per trace).
        
    Returns:
        List of formatted strings.
    """
    return [render(t, mids) for t, mids in zip(traces, selected_mids_list)]


class TemplateRenderer:
    """Template renderer with caching for efficiency."""
    
    def __init__(self):
        self._cache = {}
    
    def render(self, trace: Trace, selected_mids: list[str]) -> str:
        """Render with optional caching."""
        cache_key = f"{trace.pid}:{','.join(sorted(selected_mids))}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = render(trace, selected_mids)
        self._cache[cache_key] = result
        return result
    
    def clear_cache(self):
        """Clear the render cache."""
        self._cache.clear()