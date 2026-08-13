"""
Action token handling for supervised special tokens.

Demoted from head to supervised special token - handles special
token identification for debate actions.
"""

from typing import Any


# Special tokens for debate actions
ACTION_TOKENS = {
    "solve": "<|solve|>",
    "critique": "<|critique|>",
    "revise": "<|revise|>",
    "final_answer": "<|final_answer|>",
    "end_turn": "<|end_turn|>",
}


class ActionTokenHandler:
    """
    Handler for action tokens in the debate system.

    Demoted from head to supervised special token.
    This is a simple tokenizer wrapper for action tokens.
    """

    def __init__(self, tokenizer: Any):
        """
        Initialize the action token handler.

        Args:
            tokenizer: Base tokenizer from HuggingFace.
        """
        self.tokenizer = tokenizer

        # Add special tokens to tokenizer
        special_tokens = {"additional_special_tokens": list(ACTION_TOKENS.values())}
        self.tokenizer.add_special_tokens(special_tokens)

    def encode_action(self, action: str, text: str) -> list[int]:
        """
        Encode an action with associated text.

        Args:
            action: Action type (solve, critique, revise, etc.).
            text: Text content for the action.

        Returns:
            List of token IDs.
        """
        action_token = ACTION_TOKENS.get(action, "")
        full_text = f"{action_token}{text}"
        return self.tokenizer.encode(full_text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs.

        Returns:
            Decoded text string.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def get_action(self, token_ids: list[int]) -> str | None:
        """
        Extract the action from token IDs.

        Args:
            token_ids: List of token IDs.

        Returns:
            Action type string or None.
        """
        decoded = self.tokenizer.decode(token_ids[:10], skip_special_tokens=False)

        for action, token in ACTION_TOKENS.items():
            if token in decoded:
                return action

        return None