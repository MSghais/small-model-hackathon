from typing import Protocol


class InferenceBackend(Protocol):
    def load(self) -> None:
        """Load model weights into memory."""

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from a single prompt."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate a reply from a chat message history."""
