"""
Schema definitions for debate traces.

Provides dataclasses for representing messages and traces in the debate loop,
with stable message IDs and parent references for building the DAG structure.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


class MessageRole(str, Enum):
    """Enumeration of message roles in the debate."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    CRITIQUE = "critique"
    REVISION = "revision"


class RoundType(str, Enum):
    """Types of rounds in the debate."""
    SOLVE = "solve"
    CRITIQUE = "critique"
    REVISION = "revision"


@dataclass
class Message:
    """
    A single message in the debate trace.

    Attributes:
        mid: Stable message ID (UUID).
        parent_id: ID of the parent message (None for root).
        role: Role of the message sender.
        content: The message content.
        round: Debate round number.
        round_type: Type of round this message belongs to.
        created_at: Timestamp of creation.
        metadata: Additional metadata.
    """
    role: MessageRole
    content: str
    round: int = 0
    round_type: RoundType = RoundType.SOLVE
    parent_id: str | None = None
    mid: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "mid": self.mid,
            "parent_id": self.parent_id,
            "role": self.role.value,
            "content": self.content,
            "round": self.round,
            "round_type": self.round_type.value,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create message from dictionary representation."""
        return cls(
            mid=data["mid"],
            parent_id=data.get("parent_id"),
            role=MessageRole(data["role"]),
            content=data["content"],
            round=data.get("round", 0),
            round_type=RoundType(data.get("round_type", "solve")),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Trace:
    """
    A complete debate trace for a problem.

    Attributes:
        problem: The original problem statement.
        messages: List of messages in chronological order.
        final_answer: The final answer (if determined).
        utility: Estimated utility of this trace.
    """
    problem: str
    messages: list[Message] = field(default_factory=list)
    final_answer: str | None = None
    utility: float | None = None
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def add_message(
        self,
        content: str,
        role: MessageRole,
        round_type: RoundType,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """
        Add a message to the trace.

        Args:
            content: The message content.
            role: Role of the message sender.
            round_type: Type of round.
            parent_id: ID of parent message.
            metadata: Additional metadata.

        Returns:
            The created message.
        """
        round_num = self.messages[-1].round + 1 if self.messages else 0
        if round_type == RoundType.SOLVE:
            round_num = 0

        message = Message(
            content=content,
            role=role,
            round=round_num,
            round_type=round_type,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        self.messages.append(message)
        return message

    def get_messages_by_round(self, round_num: int) -> list[Message]:
        """Get all messages from a specific round."""
        return [m for m in self.messages if m.round == round_num]

    def get_children(self, mid: str) -> list[Message]:
        """Get all messages that are direct children of the given message."""
        return [m for m in self.messages if m.parent_id == mid]

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary representation."""
        return {
            "trace_id": self.trace_id,
            "problem": self.problem,
            "messages": [m.to_dict() for m in self.messages],
            "final_answer": self.final_answer,
            "utility": self.utility,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trace":
        """Create trace from dictionary representation."""
        messages = [Message.from_dict(m) for m in data.get("messages", [])]
        trace = cls(
            problem=data["problem"],
            messages=messages,
            final_answer=data.get("final_answer"),
            utility=data.get("utility"),
            trace_id=data.get("trace_id", str(uuid.uuid4())),
        )
        return trace