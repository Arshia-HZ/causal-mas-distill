"""
Debate schema with Message DAG structure.

Each message has stable, deterministic ID (mid) and parent references
enabling counterfactual ablation.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    """
    A single message in the debate DAG.
    
    Attributes:
        mid: Message ID in format "r{round}.{author}" (e.g., "r1.solver").
             Stable and deterministic for ablation.
        round: Debate round number (1-indexed).
        role: Author role - solver, critic, or verifier.
        text: Full message content.
        answer: Extracted answer (if any).
        parents: List of message IDs visible to this author.
                 Enables topological downstream analysis.
    """
    mid: str
    round: int
    role: str  # solver | critic | verifier
    text: str
    answer: Optional[str] = None
    parents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "mid": self.mid,
            "round": self.round,
            "role": self.role,
            "text": self.text,
            "answer": self.answer,
            "parents": self.parents,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        """Create from dictionary."""
        return cls(
            mid=d["mid"],
            round=d["round"],
            role=d["role"],
            text=d["text"],
            answer=d.get("answer"),
            parents=d.get("parents", []),
        )


@dataclass
class Trace:
    """
    Complete debate trace for a single problem.
    
    Attributes:
        pid: Problem ID.
        trace_id: Unique identifier for this trace (e.g. pid:s0).
        question: The problem statement.
        gold: Ground truth answer.
        messages: List of messages in the debate (in order).
        final_answer: Final answer from the debate.
        final_correct: Whether final answer is correct.
        topology: Topology type (e.g., "solver_critic_verifier").
    """
    pid: str
    trace_id: str
    question: str
    gold: str
    messages: list[Message] = field(default_factory=list)
    final_answer: Optional[str] = None
    final_correct: bool = False
    topology: str = "solver_critic_verifier"

    def get_message(self, mid: str) -> Optional[Message]:
        """Get message by ID."""
        for msg in self.messages:
            if msg.mid == mid:
                return msg
        return None

    def get_round_messages(self, round_num: int) -> list[Message]:
        """Get all messages from a specific round."""
        return [m for m in self.messages if m.round == round_num]

    def get_role_messages(self, role: str) -> list[Message]:
        """Get all messages from a specific role."""
        return [m for m in self.messages if m.role == role]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "pid": self.pid,
            "trace_id": self.trace_id,
            "question": self.question,
            "gold": self.gold,
            "messages": [m.to_dict() for m in self.messages],
            "final_answer": self.final_answer,
            "final_correct": self.final_correct,
            "topology": self.topology,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        """Create from dictionary."""
        return cls(
            pid=d["pid"],
            trace_id=d.get("trace_id", d["pid"]),  # fallback for old data
            question=d["question"],
            gold=d["gold"],
            messages=[Message.from_dict(m) for m in d.get("messages", [])],
            final_answer=d.get("final_answer"),
            final_correct=d.get("final_correct", False),
            topology=d.get("topology", "solver_critic_verifier"),
        )


def descendants(trace: Trace, mid: str) -> set[str]:
    """
    Get all message IDs that are topologically downstream of mid.
    
    Args:
        trace: The debate trace.
        mid: Message ID to find descendants for.
        
    Returns:
        Set of message IDs that depend on the given message.
    """
    mid_to_msg = {m.mid: m for m in trace.messages}
    downstream = set()
    queue = [mid]
    
    while queue:
        current = queue.pop()
        # Find messages that have current in their parents
        for msg in trace.messages:
            if msg.mid not in downstream and current in msg.parents:
                downstream.add(msg.mid)
                queue.append(msg.mid)
    
    return downstream


def topo_order(trace: Trace, mids: set[str]) -> list[str]:
    """
    Topological sort of message IDs based on parent relationships.
    
    Since trace.messages is topologically sorted by construction,
    we can just sort based on their index in the trace.
    """
    order = {m.mid: i for i, m in enumerate(trace.messages)}
    return sorted(mids, key=lambda mid: order.get(mid, float("inf")))