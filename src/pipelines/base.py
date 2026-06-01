from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class PipelineResponse:
    """Standard response object every pipeline must return."""
    output: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    cost_usd: float
    metadata: Optional[dict] = None


class BasePipeline(ABC):
    """
    Every LLM pipeline must implement this interface.
    This means you can swap Groq for Gemini for any other model
    without changing a single line of eval code.
    """

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def _call_api(self, prompt: str) -> PipelineResponse:
        """Subclasses implement this with their specific API call."""
        pass

    def run(self, prompt: str) -> PipelineResponse:
        """
        Public method called by the eval system.
        Wraps _call_api with timing and error handling.
        """
        start = time.time()
        try:
            response = self._call_api(prompt)
            return response
        except Exception as e:
            # Return a failed response instead of crashing the eval run
            return PipelineResponse(
                output=f"ERROR: {str(e)}",
                model=self.model,
                input_tokens=0,
                output_tokens=0,
                latency_seconds=time.time() - start,
                cost_usd=0.0,
                metadata={"error": True, "error_message": str(e)}
            )