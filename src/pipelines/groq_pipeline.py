import time
from groq import Groq
from .base import BasePipeline, PipelineResponse

# Groq pricing per 1M tokens (as of 2025, free tier)
GROQ_PRICING = {
    "llama-3.1-8b-instant":  {"input": 0.05,  "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "mixtral-8x7b-32768":    {"input": 0.24,  "output": 0.24},
}


class GroqPipeline(BasePipeline):
    """
    Pipeline wrapper for Groq-hosted models.
    Supports llama-3.1-8b-instant and other Groq models.
    """

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant",
                 temperature: float = 0.0, max_tokens: int = 1024):
        super().__init__(model, temperature, max_tokens)
        self.client = Groq(api_key=api_key)

    def _call_api(self, prompt: str) -> PipelineResponse:
        start = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        latency = time.time() - start
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        # Calculate cost
        pricing = GROQ_PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        return PipelineResponse(
            output=response.choices[0].message.content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=round(latency, 3),
            cost_usd=round(cost, 6),
            metadata={"provider": "groq"}
        )