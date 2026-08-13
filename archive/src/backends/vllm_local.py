"""
vLLM backend for local GPU inference.

Provides the same interface as ApiBackend but runs models locally
using vLLM for efficient inference on Kaggle or local machines.
"""

from typing import Any

from .base import Backend


class VLLMBackend(Backend):
    """Local vLLM-based backend for GPU inference."""

    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        dtype: str = "auto",
    ):
        """
        Initialize vLLM backend.

        Args:
            model_name_or_path: HuggingFace model name or local path.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use.
            max_model_len: Maximum model context length.
            dtype: Data type for model weights.
        """
        self.model_name_or_path = model_name_or_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self._llm: Any = None

    def _ensure_initialized(self) -> None:
        """Lazy initialization of vLLM engine."""
        if self._llm is None:
            from vllm import LLM

            self._llm = LLM(
                model=self.model_name_or_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
            )

    async def generate(
        self,
        messages: list[dict],
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        cache_nonce: str | None = None,
    ) -> list[str]:
        """
        Generate completions using local vLLM engine.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            n: Number of completions to generate.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            cache_nonce: Optional string to differentiate cache keys.

        Returns:
            List of generated completion strings.
        """
        self._ensure_initialized()

        # Convert messages to prompt string
        prompt = self._messages_to_prompt(messages)

        # Generate with vLLM in a thread so we don't block the async loop
        import asyncio
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            n=n,
            temperature=temperature if temperature > 0 else 0.001,
            max_tokens=max_tokens or 2048,
        )

        outputs = await asyncio.to_thread(self._llm.generate, [prompt], sampling_params)

        # Extract completions
        return [output.outputs[i].text for output in outputs for i in range(n)]

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert message list to prompt string."""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        prompt += "Assistant:"
        return prompt